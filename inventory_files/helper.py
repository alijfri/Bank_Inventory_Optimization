import numpy as np
import pandas as pd
from inputs import num_notes_in_bag
def enforce_cap_keep_new(proposed, max_value):
    # 1) total and NEW value
    total_value = sum(den * q for (den, nt), q in proposed.items())
    if total_value <= max_value:
        return proposed  # nothing to do

    value_new = sum(den * q for (den, nt), q in proposed.items() if nt == 'NEW')
    remaining_for_fit = max_value - value_new

    # 2) collect FIT lines
    fit_keys = [(den, nt) for (den, nt) in proposed if nt == 'FIT']
    fit_value_total = sum(den * proposed[(den, nt)] for (den, nt) in fit_keys)

    # 3) no room for FIT
    if remaining_for_fit <= 0 or fit_value_total == 0:
        for k in fit_keys:
            proposed[k] = 0
        return proposed

    # 4) proportional downscale
    scale = min(1.0, remaining_for_fit / fit_value_total)
    scaled_fit = {k: int(np.floor(proposed[k] * scale)) for k in fit_keys}

    # 5) fix rounding overflow if any (remove from largest denoms first)
    curr_fit_val = sum(den * scaled_fit[(den, nt)] for (den, nt) in fit_keys)
    if curr_fit_val > remaining_for_fit:
        for den, nt in sorted(fit_keys, key=lambda k: k[0], reverse=True):
            while scaled_fit[(den, nt)] > 0 and curr_fit_val > remaining_for_fit:
                scaled_fit[(den, nt)] -= 1
                curr_fit_val -= den
            if curr_fit_val <= remaining_for_fit:
                break

    # 6) write back
    for k in fit_keys:
        proposed[k] = scaled_fit[k]
    return proposed

def _aggregate_rep_arrival_today(rdp, t):
    """Total notes and value arriving at day t across all inv managers."""
    total_notes = 0
    total_value = 0
    for (denom, note_type), inv in rdp.inventory_managers.items():
        if t >= inv.horizon:
            continue
        if inv.incoming_orders[t]:  # deque of quantities that arrive today
            qty_today = sum(inv.incoming_orders[t])
            total_notes += qty_today
            total_value += denom * qty_today
    return total_notes, total_value

def _callback_load_today(rdp, t):
    """Bags/value scheduled to leave today (already applied in update_day_unfit)."""
    if t >= rdp.unfit_inventory.horizon:
        return 0, 0
    bags  = float(rdp.unfit_inventory.scheduled_unfit_bag_removal[t])
    value = float(rdp.unfit_inventory.scheduled_unfit_value_removal[t])
    num_notes_surpluss=0
    for (denom, note_type), inv in rdp.inventory_managers.items():
        if note_type == 'FIT':
            if t >= inv.horizon:
                continue
            num_notes_surpluss += float(inv.scheduled_removal[t])
            value += float(denom * inv.scheduled_removal[t])
    bags= bags + int(num_notes_surpluss/num_notes_in_bag)
    print(f"callback_load_today - bags:{bags}, value:{value}- Surplus notes={num_notes_surpluss}", flush=True)
    return bags, value
def _start_idx_for(rdp, start_date_str=None):
    """Return the index in rdp.calendar_index where the sim should start."""
    if not start_date_str:
        return 0
    # normalize to midnight
    s = pd.to_datetime(start_date_str).normalize()
    # exact match first
    i = rdp.date2t_str.get(s.strftime('%Y-%m-%d'))
    if i is not None:
        return i
    # else: first calendar date >= s
    return int(rdp.calendar_index.searchsorted(s))

def handle_callback_on_day(rdp, t, callback_bags, callback_value, overcap_notes):

    # New
    # find strictly AFTER t

    leave_t = t+1
    if leave_t >= rdp.unfit_inventory.horizon:
        return

    # schedule removals on leave_t
    rdp.unfit_inventory.scheduled_unfit_bag_removal[leave_t]   += callback_bags
    rdp.unfit_inventory.scheduled_unfit_value_removal[leave_t] += callback_value

    for (denom, note_type), qty in overcap_notes.items():
        if note_type == 'FIT':
            inv = rdp.inventory_managers[(denom, note_type)]
            #if leave_t <= inv.horizon:
            inv.scheduled_removal[leave_t] += qty

