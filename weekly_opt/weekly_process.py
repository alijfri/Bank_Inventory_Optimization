from inputs import start_date, finish_date
import pandas as pd
import numpy as np
from inputs import target_freq
np.set_printoptions(suppress=True)


def weekly_demand(network):
    """
    For each RDP:
      - For each child RDC, denom, note_type:
          * Align daily net demand to business-day calendar
          * Sum to Friday-ending weeks (W-FRI), keeping partial weeks (missing B-days = 0)
          * Store on the RDC: rdc.weekly_net_demand[(denom, note_type)] = (week_ends, weekly_vals)
      - Aggregate its RDC weekly series into the RDP:
          * rdp.weekly_net_demand[(denom, note_type)] = (week_ends, summed_vals)
    """
    denoms = [5, 10, 20, 50, 100]
    note_types = ['FIT', 'NEW']
    cal = pd.bdate_range(start_date, finish_date, freq="B")

    # Iterate RDPs first
    rdp_iter = network.rdps.items() if isinstance(getattr(network, "rdps", None), dict) else network.rdps
    for item in rdp_iter:
        rdp = item[1] if (isinstance(item, tuple) and len(item) == 2) else item

        if not hasattr(rdp, 'weekly_net_demand'):
            rdp.weekly_net_demand = {}

        # Prepare each RDC under this RDP
        for rdc in getattr(rdp, 'rdcs', []):
            if not hasattr(rdc, 'weekly_net_demand'):
                rdc.weekly_net_demand = {}

        # ---- per denom/note_type compute RDC weekly, then aggregate to RDP ----
        for denom in denoms:
            for note_type in note_types:
                series_list = []   # to aggregate to RDP

                # --- per-RDC weekly ---
                for rdc in getattr(rdp, 'rdcs', []):
                    daily = np.asarray(rdc.get_net_demand(denom, note_type))
                    if daily.size == 0:
                        # ensure RDC key exists but empty
                        rdc.weekly_net_demand[(denom, note_type)] = (
                            np.array([], dtype='datetime64[D]'), np.array([], dtype=int)
                        )
                        continue

                    L = min(len(cal), len(daily))
                    if L == 0:
                        rdc.weekly_net_demand[(denom, note_type)] = (
                            np.array([], dtype='datetime64[D]'), np.array([], dtype=int)
                        )
                        continue

                    idx = cal[:L]
                    vals = daily[:L].astype(float)
                    vals = np.nan_to_num(vals, nan=0.0)

                    # Build Series on B-day index and resample to Friday-ending weeks
                    s = pd.Series(vals, index=idx)
                    weekly = s.resample(target_freq).sum()  # partial weeks kept 

                    week_ends = weekly.index.normalize().to_numpy(dtype='datetime64[D]')
                    weekly_vals = weekly.to_numpy(dtype=int)
                    rdc.weekly_net_demand[(denom, note_type)] = (week_ends, weekly_vals)

                    if week_ends.size > 0:
                        series_list.append(pd.Series(weekly_vals.astype(float), index=pd.to_datetime(week_ends)))

                # --- aggregate RDCs -> RDP (align on union of Fridays) ---
                if series_list:
                    union_idx = series_list[0].index
                    for s in series_list[1:]:
                        union_idx = union_idx.union(s.index)
                    aligned = [s.reindex(union_idx, fill_value=0.0) for s in series_list]
                    summed = np.sum(aligned, axis=0)  # numpy array

                    dates_sorted = union_idx.normalize().to_numpy(dtype='datetime64[D]')
                    vals_sorted = summed.astype(int)
                    rdp.weekly_net_demand[(denom, note_type)] = (dates_sorted, vals_sorted)
                else:
                    rdp.weekly_net_demand[(denom, note_type)] = (
                        np.array([], dtype='datetime64[D]'), np.array([], dtype=int)
                    )
