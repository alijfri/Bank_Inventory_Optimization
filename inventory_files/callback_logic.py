
def should_trigger_callback(rdp, day,t, CONFIG, last_callback_day_dict):
    cfg = CONFIG[rdp.id]
    callback_day = cfg['callback_plan_day']
    callback_frequency_daily = cfg['daily_unfit_removal']
     # Use the RDP's real calendar (handles weekends/holidays dropped from the data)
    # if day.weekday() != callback_day:
    #     return False
    # Force callback if no callback in last X business days
    last_callback_day = last_callback_day_dict.get(rdp.id, None)
    # 🔹 CASE 1: No callback yet in this RHC horizon
    if last_callback_day is None:
        # Only start periodic callbacks after callback_frequency_daily days,
        # and only on the correct weekday.
        if t >= callback_frequency_daily and day.weekday() == callback_day:
            last_callback_day_dict[rdp.id] = t
            print(
                f"[{rdp.id}] FIRST periodic callback on day {t} "
                f"({callback_frequency_daily} business days since start of RHC).",
                flush=True,
            )
            return True

    # 🔹 CASE 2: We had at least one callback already → normal periodic rule
    else:
        if (t - last_callback_day) >= callback_frequency_daily and day.weekday() == callback_day:
            last_callback_day_dict[rdp.id] = t
            print(
                f"[{rdp.id}] Callback forced on day {t} "
                f"(≥ {callback_frequency_daily} business days since last).",
                flush=True,
            )
            return True

    total_value = (
        rdp.unfit_inventory.get_unfit_inventory(t) +
        sum(denom * inv.get_on_hand(t) for (denom, _), inv in rdp.inventory_managers.items())
    )

    if total_value >= int(1 * rdp.get_capacity()):
        last_callback_day_dict[rdp.id] = t
        print(f"[{rdp.id}] Callback triggered on day {t} due to overcapacity", flush=True)
        print(f"    Total value: {total_value}, Capacity: {rdp.get_capacity()}", flush=True)
        return True

    return False
