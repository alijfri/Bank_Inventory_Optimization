
import numpy as np
import pandas as pd
from .replenishment_logic import should_trigger_replenishment
from .callback_logic import should_trigger_callback
#from demand_SS_preprocess import next_t_on_weekday 
from inputs import CONFIG,get_trip_config,num_notes_in_bag
from transportation import Transportation
import sys
from .helper import enforce_cap_keep_new, _aggregate_rep_arrival_today, _callback_load_today,handle_callback_on_day
from inputs import get_trip_config ,optimization_policy,start_date,finish_date,opt_finish_date,opt_start_date






cal_rhc=pd.bdate_range(opt_start_date, opt_finish_date, freq="B")
# Get indices for the optimization window (handle non-B-days gracefully)
# _start_idx = _cal.get_indexer([pd.Timestamp(rhc_start_date)], method="bfill")[0]
# _end_idx   = _cal.get_indexer([pd.Timestamp(rhc_finish_date)], method="ffill")[0]
_cal = pd.bdate_range(start_date, finish_date, freq="B")
def simulate_inventory(network ,rdp_id,_start_idx,_end_idx,step,last_callback_day_dict=None):
    _opt_len = _end_idx - _start_idx + 1
    print(f"Simulating inventory for RDP {rdp_id} from index {_start_idx} to {_end_idx} (length {_opt_len})", flush=True)
    transport = Transportation(path_transport)
        #  🔵 pipeline replenishments that arrive AFTER _end_idx
    #  key: (denom, rdp_id, note_type) -> total qty arriving after horizon
    pipeline_rep_next = {}
    # 🔵 NEW: pipeline callbacks that leave AFTER _end_idx
    pipeline_cb_next = {}   # key: rdp_id -> {unfit_bags, unfit_value, fit_removals{(denom,'FIT'): qty}}
    overcap_log = {rdp.id: [] for rdp in network.get_rdp()}
    rep_log, callback_log = [], []
    
    # 🔹 If no state passed in, start fresh; otherwise reuse
    if last_callback_day_dict is None:
        last_callback_day_dict = {}

    # ✅ New: structured logs
    # event_logs = {
    #     'lost_sales': [],       # [{day, rdp, denom, note_type, lost}]
    #     'overcap': [],          # [{day, rdp, value}]
    #     'replenishment': [],    # [{day, rdp, denom, note_type, qty, cost}]
    #     'callback': []  ,        # [{day, rdp, value, bags, cost, type}]
    #     "inventory": [],
    # }
        # ✅ Structured logs: per-denom vs total vs unfit
    event_logs = {
        # inventory
        "inventory_per_denom": [],  # NEW/FIT per denom
        "inventory_unfit": [],      # UNFIT per RDP/day (no denom)
        "inventory_total": [],      # totals per RDP/day

        # lost sales
        "lost_sales_per_denom": [],
        "lost_sales_total": [],

        # replenishments
        "replenishment_per_denom": [],
        "replenishment_total": [],  # trip-based totals

        # callbacks
        "callback_total": [],

        # overcapacity
        "overcap_total": [],        # total value over cap
        "overcap_per_denom_fit": [],# excess FIT per denom (optional)
    }


    lost_sales_log = {
                    rdp.id: {(denom, note_type): [] for denom in [5, 10, 20, 50, 100] for note_type in ['NEW', 'FIT']}
                    for rdp in network.get_rdp()
                }
    # Per-RDP start offsets and effective lengths
    # start_idx = {r.id: _start_idx_for(r, None) for r in network.get_rdp()}
    # local_len = {r.id: max(0, len(r.calendar_index) - start_idx[r.id]) for r in network.get_rdp()}
    # 1) Precompute windowed aggregates for the optimization horizon
    unfit_value_win = {}  # (rdp_id, t) -> float
    unfit_bags_win  = {}  # (rdp_id, t) -> float

    for rdp in network.get_rdp():
        if rdp.id != rdp_id: continue
        i = rdp.id
        #print(f"example of real demand data for fit notes denom 5 for RDP {i}: {rdp.get_net_demand(5,'FIT')[_start_idx:_end_idx+1]}", flush=True)
        for t in range(len(cal_rhc)):
            src_idx = _start_idx + t
            # if src_idx > _end_idx:
            #     break  # outside the opt window

            total_val = 0.0
            total_bag = 0.0
            for rdc in rdp.get_rdcs():
                # Guard against any shorter arrays
                if 0 <= src_idx < len(rdc.unfit_value):
                    total_val += float(rdc.unfit_value[src_idx])
                if 0 <= src_idx < len(rdc.unfit_bag):
                    total_bag += float(rdc.unfit_bag[src_idx])

            unfit_value_win[(i, t)] = total_val
            unfit_bags_win[(i, t)]  = total_bag
        # print("len of unfit_value",len(unfit_value_win), len(unfit_bags_win), flush=True)
        # print(len(cal_rhc), flush=True)
        full_len = len(next(iter(rdp.get_rdcs())).unfit_value)

        full_unfit_value = [0.0] * full_len
        full_unfit_bags  = [0.0] * full_len

        for rdc in rdp.get_rdcs():
            for t in range(full_len):
                full_unfit_value[t] += float(rdc.unfit_value[t])
                full_unfit_bags[t]  += float(rdc.unfit_bag[t])
    # just the wanted chunk:
    series_unfit_value=full_unfit_value[_start_idx:_end_idx+1]
    series_unfit_bags=full_unfit_bags[_start_idx:_end_idx+1]

    idx_demand=0


    for t in range((step-1)*20,(step-1)*20+_opt_len):
        #print(f"beginning and end of range {(step-1)*20} to {step*_opt_len}")
        day = _cal[_start_idx + idx_demand] 
        #src_idx = _start_idx + t      # <-- global day index in _cal
        print(f"===== SIM DAY {t} =====", flush=True)
        for rdp in network.get_rdp():

            if rdp.id != rdp_id: continue
            print(f"Date: {day.date()}, Weekday:{day.weekday()}", flush=True)
            # Step 0: Update unfit inventory based on new deposits and possible callback
            full_unfit_bags=(rdc.unfit_value for rdc in rdp.get_rdcs())
            full_unfit_value=(rdc.unfit_bag for rdc in rdp.get_rdcs())
            new_bags  = series_unfit_bags[idx_demand]   #unfit_bags_win[(rdp.id, t)]
            new_value = series_unfit_value[idx_demand] #unfit_value_win[(rdp.id, t)]

            # We'll check later if callback will happen today
            rdp.unfit_inventory.update_day_unfit(t, new_bags, new_value, do_callback=False)  # Will override to True later if needed
            # Printing daily unfit inventory level and number of bags
            print(f"[{rdp.id}] Day {t}: Unfit inventory updated: {rdp.unfit_inventory.unfit_value_inventory[t]} value, {rdp.unfit_inventory.unfit_bag_inventory[t]} bags", flush=True)
            # 🔵 Log unfit inventory (no denom)
            unfit_val_t = float(rdp.unfit_inventory.unfit_value_inventory[t])
            unfit_bags_t = float(rdp.unfit_inventory.unfit_bag_inventory[t])
            event_logs["inventory_unfit"].append(
                {
                    "day": t,
                    "day_idx_global": int(_start_idx + t),
                    "date": str(day.date()),
                    "rdp": rdp.id,
                    "note_type": "UNFIT",
                    "unfit_value": unfit_val_t,
                    "unfit_bags": unfit_bags_t,
                }
            )
            # ---- Step 1: Fulfil demand, update lost sales, log inventory per denom ----
            total_lost_today = 0.0
            total_on_hand_notes_today = 0.0
            total_on_hand_value_today = 0.0
            # Phase 2: fulfill demand and update lost sales, etc.
            for (denom, note_type), inv in rdp.inventory_managers.items():
                series_full = rdp.get_net_demand(denom, note_type)
                demand_series = series_full[_start_idx:_end_idx+1]  # ← the OPT CHUNK only
                # if denom==5 and note_type=='FIT':
                # demand_series might be empty
                if len(demand_series) > idx_demand:
                    demand = demand_series[idx_demand]
                else:
                    demand = 0.0
                

                if note_type == 'FIT':
                    backup_inv = rdp.inventory_managers[(denom, 'NEW')]
                    inv.update_day(t, demand, backup_inventory_manager=backup_inv, denom=denom)
                else:
                    inv.update_day(t, demand)
                
                lost = float(inv.lost_sales[t])
                lost_sales_log[rdp.id][(denom, note_type)].append(lost)
                





                total_lost_today += lost
                #lost_sales_log[rdp.id][(denom, note_type)][t] = lost
                on_hand_t = float(inv.get_on_hand(t))
                total_on_hand_notes_today += on_hand_t
                total_on_hand_value_today += denom * on_hand_t
                # per-denom inventory log (NEW/FIT)
                event_logs["inventory_per_denom"].append(
                    {
                        "day": t,
                        "day_idx_global": int(_start_idx + t),
                        "date": str(day.date()),
                        "rdp": rdp.id,
                        "denom": denom,
                        "note_type": note_type,   # NEW or FIT
                        "on_hand": on_hand_t,
                        "inventory_position": float(inv.get_inventory_position(t)),
                        "s": int(round(rdp.opt_s[(denom, note_type)])),
                        "S": int(round(rdp.opt_S[(denom, note_type)])),
                    }
                )
                

                # 🔴 Log lost sales per denom and type
                if lost > 0:
                    print("💀 we have lost sales for",denom,"note_type",note_type, flush=True)
                    # event_logs['lost_sales'].append({
                    #     'day': t,'date': str(day.date()) ,'rdp': rdp.id,
                    #     'denom': denom, 'note_type': note_type,
                    #     'lost_sales': lost
                    # })
                    event_logs["lost_sales_per_denom"].append(
                        {
                            "day": t,
                            "day_idx_global": int(_start_idx + t),
                            "date": str(day.date()),
                            "rdp": rdp.id,
                            "denom": denom,
                            "note_type": note_type,
                            "lost_sales": lost,
                        }
                    )
            
            # 🔴 Log total lost sales today
            event_logs["lost_sales_total"].append(
                {
                    "day": t,
                    "day_idx_global": int(_start_idx + t),
                    "date": str(day.date()),
                    "rdp": rdp.id,
                    "lost_sales": total_lost_today,
                }
            )

            # Step 2: Overcapacity check per denomination
            total_value = (
                rdp.unfit_inventory.get_unfit_inventory(t) +
                sum(denom * inv.get_on_hand(t) for (denom, _), inv in rdp.inventory_managers.items())
            )

            # cap_limit =int(rdp.get_capacity())
            # overcap_notes = {}
            # overcap_flag = total_value > cap_limit
            # if overcap_flag:
            #     print(
            #         f"[{rdp.id}] Day {t}: Overcapacity! Total value: {total_value}, "
            #         f"Cap limit: {cap_limit}",
            #         flush=True,
            #     )

            #     # overcapacity per denom for FIT only
            #     for (denom, note_type), inv in rdp.inventory_managers.items():
            #         if note_type != "FIT":
            #             continue
            #         current_on_hand = inv.get_on_hand(t)
            #         if current_on_hand > int(round(rdp.opt_S[(denom, note_type)])):
            #             excess = current_on_hand - int(round(rdp.opt_S[(denom, note_type)])) +0.0*int(round(rdp.opt_S[(denom, note_type)]))
            #             #excess = np.max([0, excess])
            #             if excess > 0:
            #                 overcap_notes[(denom, "FIT")] = excess
            #                 print(
            #                     f"[{rdp.id}] Day {t}: Overcapacity detected for FIT "
            #                     f"notes of denom {denom}. Current on hand: "
            #                     f"{current_on_hand}, S: {int(round(rdp.opt_S[(denom, note_type)]))}, Excess: {excess}",
            #                     flush=True,
            #                 )
            #                 event_logs["overcap_per_denom_fit"].append(
            #                     {
            #                         "day": t,
            #                         "day_idx_global": int(_start_idx + t),
            #                         "date": str(day.date()),
            #                         "rdp": rdp.id,
            #                         "denom": denom,
            #                         "note_type": "FIT",
            #                         "excess_notes": float(excess),
            #                     }
            #                 )

            #     overcap_value = max(0, total_value - cap_limit)
            #     overcap_event = {
            #         "day": t,
            #         "day_idx_global": int(_start_idx + t),
            #         "date": str(day.date()),
            #         "rdp": rdp.id,
            #         "value": float(overcap_value),
            #     }
            #     overcap_log[rdp.id].append(overcap_event)
            #     event_logs["overcap_total"].append(overcap_event)
            cap_limit = int(rdp.get_capacity())
            target_ratio = 0.90
            target_value = int(target_ratio * cap_limit)

            overcap_notes = {}

            # Only trigger if we are above the 80% target
            overcap_flag = total_value > cap_limit
            if overcap_flag:
                overcap_value = total_value - target_value

                print(
                    f"[{rdp.id}] Day {t}: Overcapacity! Total value: {total_value} "
                    f"(Cap: {cap_limit}) ",
                    flush=True
                )

                # --- 1) Build candidate pool: FIT notes above S ---
                for (denom, note_type), inv in rdp.inventory_managers.items():
                    if note_type != "FIT":
                        continue

                    current_on_hand = inv.get_on_hand(t)
                    S_dn = int(round(rdp.opt_S[(denom, note_type)]))

                    if current_on_hand > S_dn:
                        excess = current_on_hand - S_dn
                        if excess > 0:
                            overcap_notes[(denom, "FIT")] = excess

                            print(
                                f"[{rdp.id}] Day {t}: Overcapacity candidate for FIT "
                                f"notes of denom {denom}. Current on hand: {current_on_hand}, "
                                f"S: {S_dn}, Excess: {excess}",
                                flush=True,
                            )


                # --- 2) Decide how many notes to actually remove from candidates ---
                # total "removable" value if we used all excess
                candidate_value = sum(
                    excess * denom
                    for (denom, note_type), excess in overcap_notes.items()
                )

                if candidate_value > 0:
                    # fraction of the candidate pool we need to remove
                    frac = min(1.0, overcap_value / candidate_value)

                    removed_notes = {}  # (denom, "FIT") -> notes actually removed

                    for (denom, note_type), excess in overcap_notes.items():
                        n_remove = int(round(excess * frac))
                        if n_remove <= 0:
                            continue
                        removed_notes[(denom, "FIT")] = n_remove
                        event_logs["overcap_per_denom_fit"].append(
                                {
                                    "day": t,
                                    "day_idx_global": int(_start_idx + t),
                                    "date": str(day.date()),
                                    "rdp": rdp.id,
                                    "denom": denom,
                                    "note_type": "FIT",
                                    "excess_notes": float(n_remove),
                                }
                            )
                        # What we actually do to adjust inventory:
                        print(
                            f"[{rdp.id}] Day {t}: Removing proportionally {n_remove} notes of denom {denom} "
                            f"to address overcapacity.",
                            flush=True,
                        )
                

                    overcap_notes=removed_notes
                    print("Adjusted overcap_notes after proportional removal:", overcap_notes, flush=True)

                    # overcap_value here is "value we tried to remove"
                    overcap_event = {
                        "day": t,
                        "day_idx_global": int(_start_idx + t),
                        "date": str(day.date()),
                        "rdp": rdp.id,
                        "value": float(overcap_value),
                    }
                    overcap_log[rdp.id].append(overcap_event)
                    event_logs["overcap_total"].append(overcap_event)









            # --- Step 3: Check triggers ---
            do_rep = should_trigger_replenishment(rdp,day,t, CONFIG, optimization_policy)
            do_callback = should_trigger_callback(rdp, day,t, CONFIG, last_callback_day_dict)
            #print("Last callback day dict:", last_callback_day_dict, flush=True)
            if do_rep:
                print(f"[{rdp.id}] Day {t}: Replenishment triggered.", flush=True)
            if do_callback:
                print(f"[{rdp.id}] Day {t}: Callback triggered.", flush=True)

                   
            # --- Step 4: Aggregate quantities ---
            max_value = get_trip_config(rdp.id)["max_value"]

            proposed = {}  # {(denom, note_type): qty}
            # 1) Decide raw (pre-cap) orders
            
            for (denom, note_type), inv in rdp.inventory_managers.items():
                current_position = inv.get_inventory_position(t)
                target_S = int(round(rdp.opt_S[(denom, note_type)]))
                # If you truly want 20% above s threshold for FIT trigger, use the next line; else keep inv.s
                # threshold_s = int(np.ceil(1.2 * inv.s))
                threshold_s = int(round(rdp.opt_s[(denom, note_type)]))
 
                qty = 0
                if do_rep:
                    if optimization_policy == 'joint_ss':
                        if current_position < target_S:
                            qty = max(0, target_S - current_position)
                    elif optimization_policy == 'separate_ss':
                        if current_position < threshold_s:
                            qty = target_S - current_position
                if qty > 0:
                    proposed[(denom, note_type)] = qty
            proposed=enforce_cap_keep_new(proposed, max_value)  # Ensure we don't exceed cap while keeping NEW fully
            # 2) Place orders for proposed quantities
            note_summary = {}
            total_value = 0
            total_notes = 0
            for (denom, note_type), qty in proposed.items():
                if qty <= 0:
                    continue
                inv = rdp.inventory_managers[(denom, note_type)]
                inv.place_order(t, qty)
                note_summary[(denom, note_type)] = qty
                print(f"[{rdp.id}] Day {t}: Placed order for {qty} of denom {denom}, type {note_type},(s,S)=({int(round(rdp.opt_s[(denom, note_type)]))},{int(round(rdp.opt_S[(denom, note_type)]))})", flush=True)
                total_value += denom * qty
                total_notes += qty
                event_logs["replenishment_per_denom"].append(
                    {
                        "day": t,
                        "day_idx_global": int(_start_idx + t),
                        "date": str(day.date()),
                        "rdp": rdp.id,
                        "denom": denom,
                        "note_type": note_type,
                        "qty_notes": float(qty),
                        "value": float(denom * qty),
                    }
                )

                # Global day index of order AND arrival
                lead_time_days = CONFIG[rdp.id]['lead_time']
                order_idx = t
                arrival_idx = order_idx + lead_time_days
                if arrival_idx > (step-1)*20+_opt_len:
                    print(f"arrival_idx {arrival_idx} beyond end idx {(step-1)*20+_opt_len}", flush=True)
                    print(f"Planned order for denom {denom}, type {note_type}, qty {qty} will arrive after horizon", flush=True)
                    key = (denom, rdp.id, note_type)
                    pipeline_rep_next[key] = pipeline_rep_next.get(key, 0) + qty

            callback_value = callback_bags = 0
            if do_callback:
                callback_bags = rdp.unfit_inventory.unfit_bag_inventory[t]
                callback_value = rdp.unfit_inventory.unfit_value_inventory[t]
                removal_day = t + 1  # assume callback leaves next day

                if removal_day >= (step-1)*20+_opt_len:
                    print(f"callback removal day {removal_day} beyond end idx {(step-1)*20+_opt_len}", flush=True)
                    print(f"Planned callback for RDP {rdp.id} will leave after horizon", flush=True)

                    # ✅ Initialize entry for this RDP if not created yet
                    cb_state = pipeline_cb_next.setdefault(
                        rdp.id,
                        {
                            "unfit_bags": 0.0,
                            "unfit_value": 0.0,
                            "fit_removals": {},  # (denom, 'FIT') -> qty
                        },
                    )

                    cb_state["unfit_bags"]  += float(callback_bags)
                    cb_state["unfit_value"] += float(callback_value)

                    for (denom, note_type), qty in overcap_notes.items():
                        if note_type == 'FIT':
                            cb_state["fit_removals"][(denom, 'FIT')] = \
                                cb_state["fit_removals"].get((denom, 'FIT'), 0.0) + float(qty)

                print("pipeline_cb_next:", pipeline_cb_next)
                handle_callback_on_day(
                    rdp=rdp,
                    t=t,
                    callback_bags=callback_bags,
                    callback_value=callback_value,
                    overcap_notes=overcap_notes
                )
                print(f"callback unfit bags:{callback_bags}, callback unfit value:{callback_value}, overcap_notes:{overcap_notes}", flush=True)

            #print(f"checking the total notes and value at day {t}: {total_notes}, {total_value}", flush=True)
            # --- Step 5: Transport cost ---
            aoc = network.get_aoc_for_rdp(rdp.id)
            # --- Day-of-execution transport cost & logging (calendar-based) ---
            rep_notes_today, rep_value_today = _aggregate_rep_arrival_today(rdp, t)
            cb_bags_today, cb_value_today    = _callback_load_today(rdp, t)
            
            if (rep_notes_today > 0) or (cb_bags_today > 0):
                round_trip = (rep_notes_today > 0) and (cb_bags_today > 0)

                aoc = network.get_aoc_for_rdp(rdp.id)


                # single log entry per RDP/day
                if rep_notes_today > 0:
                    cost = transport.repcall_cost(
                    origin=aoc.name,
                    destination=rdp.id,
                    num_notes=rep_notes_today,
                    value_notes=rep_value_today,
                    callback_value=cb_value_today,
                    callback_bags=cb_bags_today,
                    activity=("replenishment"),
                    round_trip=round_trip
                )
                    ev_rep = {
                        "day": t,
                        "day_idx_global": int(_start_idx + t),
                        "date": str(day.date()),
                        "rdp": rdp.id,
                        "notes": float(rep_notes_today),
                        "value": float(rep_value_today),
                        "cost": float(cost),
                        "round_trip": round_trip,
                    }
                    rep_log.append(ev_rep)
                    event_logs["replenishment_total"].append(ev_rep)

                if cb_bags_today > 0:

                    #sum of fit notes being removed today

                    cost = transport.repcall_cost(
                    origin=aoc.name,
                    destination=rdp.id,
                    num_notes=total_notes,
                    value_notes=total_value,
                    callback_value=cb_value_today,
                    callback_bags=cb_bags_today,
                    activity=("callback"),
                    round_trip=round_trip
                )
                    ev_cb = {
                        "day": t,
                        "day_idx_global": int(_start_idx + t),
                        "date": str(day.date()),
                        "rdp": rdp.id,
                        "bags": float(cb_bags_today),
                        "value": float(cb_value_today),
                        "cost": float(cost),
                        "round_trip": round_trip,
                    }
                    callback_log.append(ev_cb)
                    event_logs["callback_total"].append(ev_cb)
            # ---- Inventory TOTAL row per RDP/day ----
            total_value_all = (
                total_on_hand_value_today + rdp.unfit_inventory.get_unfit_inventory(t)
            )
            event_logs["inventory_total"].append(
                {
                    "day": t,
                    "day_idx_global": int(_start_idx + t),
                    "date": str(day.date()),
                    "rdp": rdp.id,
                    "on_hand_notes": float(total_on_hand_notes_today),
                    "on_hand_value": float(total_on_hand_value_today),
                    "unfit_value": float(unfit_val_t),
                    "unfit_bags": float(unfit_bags_t),
                    "total_value_all": float(total_value_all),
                    "capacity": float(cap_limit),
                    "overcap_flag": int(overcap_flag),
                }
            )
        # END RDP LOOP
        # now we want to pass the last days unarrived replenishments and callbacks to the next day
        # Collect the last days inventory levels to pass to next day in optimization model
        # After the daily loop
        idx_demand+=1
    last_t = _opt_len - 1

    # Snapshot of end-of-window state
    end_inventory = {}
    end_unfit = {}

    for rdp in network.get_rdp():
        if rdp.id != rdp_id:
            continue

        for (denom, note_type), inv in rdp.inventory_managers.items():
            end_inventory[(denom, rdp.id, note_type)] = inv.get_on_hand(t)

        end_unfit[rdp.id] = rdp.unfit_inventory.get_unfit_inventory(t)
    print(f"End inventory example: {list(end_inventory.items())[:5]}")
    print(f"End unfit example: {list(end_unfit.items())[:5]}")
    print(f"Pipeline replenishments to next step: {list(pipeline_rep_next.items())[:5]}")
    print(f"total inventory values at end plus unfit: {sum(denom*qty for (denom, _, _), qty in end_inventory.items()) + sum(end_unfit.values())}", flush=True)

    return (
        rep_log,
        callback_log,
        lost_sales_log,
        overcap_log,
        event_logs,
        end_inventory,
        end_unfit,
        pipeline_rep_next,
        pipeline_cb_next,
        last_callback_day_dict,
    )

    # return rep_log, callback_log, lost_sales_log, overcap_log, event_logs

