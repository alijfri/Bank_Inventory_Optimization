import numpy as np
import pandas as pd
from inputs import start_date, finish_date,target_freq

# Master business-day calendar that aligns with rdc.unfit_value positions
_cal = pd.bdate_range(start_date, finish_date, freq="B")

def weekly_unfit_process(network):
    for rdp in network.rdps.values():
        # RDP aggregate map: {week_end_date (np.datetime64[D]): total_value}
        rdp_map = {}

        for rdc in rdp.get_rdcs():
            arr = getattr(rdc, "unfit_value", None)

            # Default empty result for this RDC
            rdc.weekly_unfit_demand = (np.array([], dtype="datetime64[D]"),
                                       np.array([], dtype=int))

            if arr is None or len(arr) == 0:
                continue

            # Clip to common length (calendar vs array)
            L = min(len(_cal), len(arr))
            if L == 0:
                continue

            # Per-RDC map
            rdc_map = {}

            # Positional loop over aligned B-days
            for i in range(L):
                bday = _cal[i]
                # Friday-ending week bucket
                wk_end = bday.to_period(target_freq).end_time.normalize()
                wk_end_d = np.datetime64(wk_end.date())  # dtype 'datetime64[D]'

                v = float(arr[i])
                if np.isnan(v):
                    v = 0.0

                rdc_map[wk_end_d] = rdc_map.get(wk_end_d, 0.0) + v

            # Store RDC weekly result (sorted by week-end)
            rdc_dates = np.array(sorted(rdc_map.keys()), dtype="datetime64[D]")
            rdc_vals  = np.array([int(round(rdc_map[d])) for d in rdc_dates], dtype=int)
            rdc.weekly_unfit_demand = (rdc_dates, rdc_vals)

            # Add into RDP aggregate
            for d, v in zip(rdc_dates, rdc_vals):
                rdp_map[d] = rdp_map.get(d, 0) + int(v)

        # Store RDP weekly result
        if rdp_map:
            rdp_dates = np.array(sorted(rdp_map.keys()), dtype="datetime64[D]")
            rdp_vals  = np.array([rdp_map[d] for d in rdp_dates], dtype=int)
        else:
            rdp_dates = np.array([], dtype="datetime64[D]")
            rdp_vals  = np.array([], dtype=int)

        rdp.weekly_unfit_demand = (rdp_dates, rdp_vals)
