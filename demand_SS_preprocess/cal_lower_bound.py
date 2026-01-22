import pandas as pd
import numpy as np
from inputs import  start_date, finish_date

# Master business-day calendar and horizon length
# _cal = pd.bdate_range(start_date, finish_date, freq="B")
# h_start = _cal.get_indexer([pd.Timestamp(opt_start_date)], method="bfill")[0]
# h_end   = _cal.get_indexer([pd.Timestamp(opt_finish_date)], method="ffill")[0]
# H = h_end - h_start + 1
_cal = pd.bdate_range(start_date, finish_date, freq="B")
def idx_of(dt):
    i = _cal.get_indexer([pd.Timestamp(dt)], method="bfill")[0]
    if i == -1:
        i = _cal.get_indexer([pd.Timestamp(dt)], method="ffill")[0]
    return i

def _p_mu_M(window_vals: np.ndarray):
    """Return (p_nonzero, mu_nonzero, max_spike) for a window."""
    if window_vals.size == 0:
        return 0.0, 0.0, 0.0
    nz = window_vals[window_vals > 0]
    p = float(len(nz)) / float(len(window_vals)) if len(window_vals) > 0 else 0.0
    mu = float(nz.mean()) if len(nz) > 0 else 0.0
    M = float(nz.max()) if len(nz) > 0 else 0.0
    return p, mu, M

def setting_lower_bound(
    network,
    opt_start_date,
    opt_finish_date,
    H_days=10,            # protection window (business days)
    w_last=0.7,           # weight for last year
    w_prev=0.3,           # weight for two years ago
    alpha=0.1,           # spike buffer fraction 0.1 was good
    granularity=1000      # round s up to this multiple
):
    """
    Zero-inflated small-s:
      small_s = ceil( p_nz * H_days * mu_nz + alpha * M )
    - p_nz, mu_nz are from last-year and two-years-ago windows (weighted).
    - If one window is all-zero/empty, ignore it (use the other with weight=1).
    - M is the max non-zero daily demand across the two historical windows.
    """
    
    h_start = _cal.get_indexer([pd.Timestamp(opt_start_date)], method="bfill")[0]
    h_end   = _cal.get_indexer([pd.Timestamp(opt_finish_date)], method="ffill")[0]
    H = h_end - h_start + 1
    for rdp in network.rdps.values():
        for denom in [5, 10, 20, 50, 100]:
            for note_type in ['FIT', 'NEW']:

                # Align historical windows of length H
                ly_start = idx_of(pd.Timestamp(opt_start_date) - pd.DateOffset(years=1))
                ly_end   = ly_start + (H - 1)
                ty_start = idx_of(pd.Timestamp(opt_start_date) - pd.DateOffset(years=2))
                ty_end   = ty_start + (H - 1)

                #series = rdp.get_net_demand(denom, note_type)
                series=rdp.get_demand(denom, note_type,'W')
                ly = np.array(series[ly_start:ly_end + 1], dtype=float)
                ty = np.array(series[ty_start:ty_end + 1], dtype=float)

                p_ly, mu_ly, M_ly = _p_mu_M(ly)
                p_ty, mu_ty, M_ty = _p_mu_M(ty)

                ly_valid = (mu_ly > 0.0) and (p_ly > 0.0)
                ty_valid = (mu_ty > 0.0) and (p_ty > 0.0)

                if not ly_valid and not ty_valid:
                    p_nz, mu_nz, M = 0.0, 0.0, 0.0
                elif ly_valid and not ty_valid:
                    p_nz, mu_nz, M = p_ly, mu_ly, M_ly
                elif ty_valid and not ly_valid:
                    p_nz, mu_nz, M = p_ty, mu_ty, M_ty
                else:
                    # both valid: weighted blend
                    p_nz = w_last * p_ly + w_prev * p_ty
                    mu_nz = w_last * mu_ly + w_prev * mu_ty
                    M = max(M_ly, M_ty)

                # small-s with spike guard
                expected_total = p_nz * H_days * mu_nz
                small_s = expected_total + alpha * M

                # round up to granularity and clamp at 0
                
                s_lower_bound = int(np.ceil(small_s / granularity) * granularity)

                s_lower_bound = max(s_lower_bound, 0)

                rdp.set_lower_bound(denom, note_type, s_lower_bound)

     
