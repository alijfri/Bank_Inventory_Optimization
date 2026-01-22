
import numpy as np
from collections import deque

class InventoryManager:
    def __init__(
        self,
        location_id,
        capacity,
        s, S,
        rep_arrival_day,
        lead_time,
        callback_leaves_day,
        AOC_received_day,
        horizon,
        denomination,
        note_type,
        #dow,
        initial_inventory,                      # <-- NEW: initial inventory value for day 0
    ):
        self.location_id = location_id
        self.capacity = capacity
        self.s = s
        self.S = S
        self.rep_arrival_day = rep_arrival_day
        self.lead_time = lead_time
        self.callback_leaves_day = callback_leaves_day
        self.AOC_received_day = AOC_received_day

        # horizon & calendar
        self.horizon = int(horizon)
        #self.dow = np.asarray(dow, dtype=int)
        # if len(self.dow) != self.horizon:
        #     raise ValueError(f"[{location_id}] len(dow)={len(self.dow)} != horizon={self.horizon}")

        # state
        self.on_hand = np.zeros(self.horizon, dtype=float)
        
        self.inventory_position = np.zeros(self.horizon, dtype=float)

        # UNFIT-only state (safe to exist for all types)
        self.unfit_value_inventory = np.zeros(self.horizon, dtype=float)
        self.unfit_bag_inventory   = np.zeros(self.horizon, dtype=float)

        self.orders_placed     = np.zeros(self.horizon, dtype=float)
        self.callback__unfit_placed = np.zeros(self.horizon, dtype=float)
        self.incoming_orders   = [deque() for _ in range(self.horizon)]
        self.lost_sales        = np.zeros(self.horizon, dtype=float)

        self.denom = denomination
        self.note_type = note_type.upper()

        # scheduled removals (overcap / unfit leave days)
        self.scheduled_removal             = np.zeros(self.horizon, dtype=float)  # FIT overcap removal
        self.scheduled_unfit_bag_removal   = np.zeros(self.horizon, dtype=float)
        self.scheduled_unfit_value_removal = np.zeros(self.horizon, dtype=float)
        self.initial_inventory = initial_inventory
        #self.on_hand[0] = int(0.6 * self.S)  # Initial inventory, can be adjusted
    # ---- helpers -------------------------------------------------------------

    # def _next_t_on_weekday(self, start_t, target_wd):
    #     """Return smallest t' > start_t with dow == target_wd, else None."""
    #     for i in range(start_t + 1, self.horizon):
    #         if self.dow[i] == target_wd:
    #             return i
    #     return None

    # ---- daily updates -------------------------------------------------------

    def update_day(self, t, demand, backup_inventory_manager=None, denom=None, debug_denom=None):
        """
        - Receive shipments due today
        - Fulfill demand (use NEW as backup if this is FIT)
        - Apply scheduled removals for today
        - Update inventory position
        """
        if t >= self.horizon:
            return
        if t==0:
            self.on_hand[0]= self.initial_inventory
        if t > 0:
            self.on_hand[t] = self.on_hand[t - 1]
            if self.denom == 50:  # ✅ only for denom 5
                print(f"[{self.location_id}] 📊 Denom {self.denom}, Type: {self.note_type}, Day {t}: On hand BEFORE update = {self.on_hand[t]}", flush=True)

        # shipments arriving today
        if self.incoming_orders[t]:
            self.on_hand[t] += sum(self.incoming_orders[t])
            #if self.denom == 100:  # ✅ only for denom 5
            print(f"[{self.location_id}] 🚚 Receiving {list(self.incoming_orders[t]) } of denom {self.denom} note type {self.note_type}")

        # demand fulfillment
        fulfilled = min(demand, self.on_hand[t])
        self.on_hand[t] -= fulfilled
        remaining = demand - fulfilled

        # backup from NEW when this inv is FIT
        if remaining > 0 and backup_inventory_manager is not None:
            use_denom = self.denom if denom is None else denom
            backup_fulfilled = min(remaining, backup_inventory_manager.on_hand[t])
            if backup_fulfilled > 0:
                backup_inventory_manager.on_hand[t] -= backup_fulfilled
                fulfilled += backup_fulfilled
                remaining -= backup_fulfilled
                print(f"[{self.location_id}] 🔄 Substituted {backup_fulfilled} of denom {use_denom} "
                      f"from NEW for FIT demand on day {t}, remaining: {remaining}")

        # lost sales
        self.lost_sales[t] = remaining

        # apply scheduled removals *after* demand processing, same-day effect
        if self.scheduled_removal[t] > 0:
            self.on_hand[t] = max(0.0, self.on_hand[t] - self.scheduled_removal[t])

        # inventory position = on-hand + all future on-order
        on_order_future = 0.0
        for day_deque in self.incoming_orders[t+1:]:
            for qty in day_deque:
                on_order_future += qty
        self.inventory_position[t] = self.on_hand[t] + on_order_future
        if self.denom == 50:  # ✅ only for denom 5
            print(f"[{self.location_id}] 📦 Day {t}: On hand EOD = {self.on_hand[t]},Inv position = {self.inventory_position[t]}", flush=True)
            print("Denom",self.denom,"Note_type: ",self.note_type,"Demand: ",demand)

        # # optional focused debug
        # if debug_denom is not None and self.denom == debug_denom:
        #     print(f"[{self.location_id}] Denom {self.denom} {self.note_type} | t={t} "
        #           f"Demand={demand} OnHand={self.on_hand[t]} InvPos={self.inventory_position[t]}")

    def place_order(self, t, order_qty):
        """
        Place an order at time t. Arrives on the next calendar index whose weekday
        equals self.rep_arrival_day. If none remains within horizon, it’s dropped.
        """
        if t >= self.horizon or order_qty <= 0:
            return

        self.orders_placed[t] = order_qty
        if order_qty>0:
            order_t=t
        #arrival_leadtime=(order_t+self.lead_time)%5
        #arrival_t = self._next_t_on_weekday(t, self.rep_arrival_day)
        #arrival_t = self._next_t_on_weekday(t, arrival_leadtime)
        arrival_t=order_t+ self.lead_time
        print(f"[{self.location_id}] 🛒 Placing order of {order_qty} at t={t} for arrival at t={arrival_t} (lead time={self.lead_time} days)")
        #print(f"arrival_leadtime: {arrival_leadtime}, order_t: {order_t}, arrival_t: {arrival_t}, t: {t}")
        if arrival_t is None:
            # no feasible arrival within horizon
            print(f"[{self.location_id}] ⚠️ No arrival day (wd={self.rep_arrival_day}) available after t={t}")
            return
        if arrival_t >= self.horizon:
            print(f"[{self.location_id}] ⚠️ Arrival t={arrival_t} exceeds horizon")
            return
        self.incoming_orders[arrival_t].append(order_qty)

    # ---- getters -------------------------------------------------------------

    def get_on_hand(self, t):
        return 0.0 if t >= self.horizon else self.on_hand[t]

    def get_inventory_position(self, t):
        return 0.0 if t >= self.horizon else self.inventory_position[t]

    def get_unfit_inventory(self, t):
        if self.note_type != 'UNFIT':
            raise ValueError("get_unfit_inventory() called on non-UNFIT inventory manager")
        return 0.0 if t >= self.horizon else self.unfit_value_inventory[t]

    # ---- unfit updates -------------------------------------------------------

    def update_day_unfit(self, t, new_bags_today, new_value_today, do_callback=False):
        """
        UNFIT manager only:
        - carry over previous day
        - add today's new unfit deposits
        - apply scheduled removals for today
        - if do_callback: zero out
        """
        if t >= self.horizon:
            return

        if t > 0:
            self.unfit_bag_inventory[t]   = self.unfit_bag_inventory[t - 1]
            self.unfit_value_inventory[t] = self.unfit_value_inventory[t - 1]

        self.unfit_bag_inventory[t]   += float(new_bags_today)
        self.unfit_value_inventory[t] += float(new_value_today)

        # scheduled removal (leave day)
        if self.scheduled_unfit_bag_removal[t] or self.scheduled_unfit_value_removal[t]:
            before_b, before_v = self.unfit_bag_inventory[t], self.unfit_value_inventory[t]

            self.unfit_bag_inventory[t]   = max(0.0, self.unfit_bag_inventory[t]   - self.scheduled_unfit_bag_removal[t])
            self.unfit_value_inventory[t] = max(0.0, self.unfit_value_inventory[t] - self.scheduled_unfit_value_removal[t])
            print(f"[{self.location_id}] 🟣 UNFIT removal @t={t}: bags {before_b}→{self.unfit_bag_inventory[t]}, "f"value {before_v}→{self.unfit_value_inventory[t]}")

        if do_callback:
            self.unfit_bag_inventory[t] = 0.0
            self.unfit_value_inventory[t] = 0.0
