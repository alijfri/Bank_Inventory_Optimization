def should_trigger_replenishment(rdp, day,t, CONFIG, policy):
    denoms = [5, 10, 20, 50, 100]
    rep_day = CONFIG[rdp.id]['rep_plan_day']
    is_rep_day = (day.weekday() == rep_day)


    if not is_rep_day:
        return False
    print(f"Repelishment Policy is: {policy}")
    # --- 2) Planned rules on rep day ---
    for denom in denoms:
        inv_new = rdp.inventory_managers.get((denom, 'NEW'))
        inv_fit = rdp.inventory_managers.get((denom, 'FIT'))

        # Rule A: NEW below s
        # if inv_new and inv_new.get_inventory_position(t) <= inv_new.s:
        #     return True

        if policy == 'combined':
            inv_new_pos = inv_new.get_inventory_position(t) if inv_new else 0
            inv_fit_pos = inv_fit.get_inventory_position(t) if inv_fit else 0
            total_inventory = inv_new_pos + inv_fit_pos
            total_s = rdp.get_s_fit_and_new(denom)  # your combined s
            if total_inventory <= total_s:
                return True

        elif policy == 'separate_ss':
            inv_new_pos = inv_new.get_inventory_position(t) if inv_new else 0
            inv_fit_pos = inv_fit.get_inventory_position(t) if inv_fit else 0
            ss_new = int(round(rdp.opt_s[(denom, 'NEW')]))
            ss_fit = int(round(rdp.opt_s[(denom, 'FIT')]))
            if inv_fit_pos + inv_new_pos <= ss_fit + ss_new:
                print(f"Inv level for denom {denom}, New+FIT: {inv_new_pos + inv_fit_pos} (ss: {ss_new + ss_fit})")
                return True

            # if inv_new_pos <= ss_new or inv_fit_pos <= ss_fit:
            #     print(f"Inv level for denom {denom}, note type NEW: {inv_new_pos} (ss: {ss_new}), FIT: {inv_fit_pos} (ss: {ss_fit})")
            #     return True
            if inv_new_pos <= ss_new:
                print(f"Inv level for denom {denom}, note type NEW: {inv_new_pos}<= (ss: {ss_new}), FIT: {inv_fit_pos} (ss: {ss_fit})")
                return True
        elif policy == "joint_ss":
            inv_new_pos = inv_new.get_inventory_position(t) if inv_new else 0
            inv_fit_pos = inv_fit.get_inventory_position(t) if inv_fit else 0
            ss_new = int(round(rdp.opt_s[(denom, 'NEW')]))  
            ss_fit = int(round(rdp.opt_s[(denom, 'FIT')]))
            if inv_fit_pos + inv_new_pos <= ss_fit + ss_new:
                print(f"Inv level for denom {denom}, New+FIT: {inv_new_pos + inv_fit_pos} (ss: {ss_new + ss_fit})")
                return True

            # if inv_new_pos <= ss_new or inv_fit_pos <= ss_fit:
            #     print(f"Inv level for denom {denom}, note type NEW: {inv_new_pos} (ss: {ss_new}), FIT: {inv_fit_pos} (ss: {ss_fit})")
            #     return True
            if inv_new_pos <= ss_new:
                print(f"Inv level for denom {denom}, note type NEW: {inv_new_pos}<= (ss: {ss_new}), FIT: {inv_fit_pos} (ss: {ss_fit})")
                return True

        else:
            raise ValueError(f"Unknown replenishment policy: {policy}")

    return False
