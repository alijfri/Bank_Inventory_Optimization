def should_trigger_rebalancing(rdp, t, CONFIG, policy):
    denoms = [5, 10, 20, 50, 100]
    rep_day = CONFIG[rdp.id]['rep_plan_day']
    is_rep_day = (rdp.calendar_index[t].weekday() == rep_day)
    rdp.get_adjacent_rdps()
    # --- 1) Emergency rule on NON rep days: only FIT dangerously low (<= 50% s) ---
    if not is_rep_day:

        return False

    # --- 2) Planned rules on rep day ---
    for denom in denoms:
        inv_new = rdp.inventory_managers.get((denom, 'NEW'))
        inv_fit = rdp.inventory_managers.get((denom, 'FIT'))

        # Rule A: NEW below s
        if inv_new and inv_new.get_inventory_position(t) <= inv_new.s:
            return False


        if inv_fit and inv_fit.get_inventory_position(t) <= inv_fit.s:
            if rdp.get_adjacent_rdps():
                for adj in rdp.get_adjacent_rdps():
                    adj_inv_fit = adj.inventory_managers.get((denom, 'FIT'))
                    if adj_inv_fit and adj_inv_fit.get_inventory_position(t) > adj_inv_fit.S:
                        print(f"[Rebalancing Day] {rdp.id} t={t} denom={denom}: FIT <= s and adjacent {adj.id} has FIT > S -> trigger rebalancing")
                        return True
            return False

        else:
            raise ValueError(f"Unknown replenishment policy: {policy}")

    return False
