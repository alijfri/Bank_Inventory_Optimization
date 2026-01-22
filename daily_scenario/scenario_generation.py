
from inputs import start_date, finish_date
import pandas as pd
import numpy as np





def prediction_weighted_average(network,
                                prediction_horizon_start,
                                prediction_horizon_end,
                                w_last=0.55, w_prev=0.45):
    """
    For each (rdp, denom, note_type), predict on each business day t in the horizon:
        pred(t) = w_last * demand(t-1y) + w_prev * demand(t-2y)
    Implementation is purely index-based:
      - Build a master business-day calendar `_cal` from global start_date..finish_date
      - Compute [h_start, h_end] = indices for prediction window
      - Compute aligned slices 1y back and 2y back:
           [h1_start, h1_end] and [h2_start, h2_end]
      - Slice the historical array and take a weighted average
    """

    # 0) Master calendar and horizon indices (exactly like your code)
    _cal = pd.bdate_range(start_date, finish_date, freq="B")

    h_start = _cal.get_indexer([pd.Timestamp(prediction_horizon_start)], method="bfill")[0]
    h_end   = _cal.get_indexer([pd.Timestamp(prediction_horizon_end)],   method="ffill")[0]
    H = h_end - h_start + 1
    if H <= 0:
        raise ValueError("Empty prediction horizon.")

    # 1) Indices for 1y-back and 2y-back windows (same length H)
    def idx_of(dt):  # helper: calendar index of a date using your rule
        i = _cal.get_indexer([pd.Timestamp(dt)], method="bfill")[0]
        if i == -1:
            i = _cal.get_indexer([pd.Timestamp(dt)], method="ffill")[0]
        return i

    one_year_back_start = idx_of(pd.Timestamp(prediction_horizon_start) - pd.DateOffset(years=1))
    one_year_back_end   = one_year_back_start + (H - 1)

    two_year_back_start = idx_of(pd.Timestamp(prediction_horizon_start) - pd.DateOffset(years=2))
    two_year_back_end   = two_year_back_start + (H - 1)

    # 2) Simple safe slicer for numpy/list histories
    def take_slice(arr, i0, i1, H):
        """
        Return length-H slice arr[i0:i1+1].
        If indices are invalid or arr too short, return zeros(H).
        """
        if arr is None:
            return np.zeros(H, dtype=float)
        if not hasattr(arr, "__len__"):
            return np.zeros(H, dtype=float)
        n = len(arr)
        if i0 is None or i1 is None or i0 < 0 or i1 < 0 or i1 >= n or i0 >= n:
            return np.zeros(H, dtype=float)
        # If the computed range overshoots, bail out to zeros (keeps it simple)
        if (i1 - i0 + 1) != H:
            return np.zeros(H, dtype=float)
        sl = np.asarray(arr[i0:i1+1], dtype=float)
        # Guard NaNs/inf
        sl = np.where(np.isfinite(sl), sl, 0.0)
        return sl

    denominations = [5, 10, 20, 50, 100]
    note_types    = ['NEW', 'FIT']

    rows = []
    pred_dates = _cal[h_start:h_end+1]
    for rdp in network.rdps.values():
        rdp_id = rdp.id
        for b in denominations:
            for n in note_types:
                hist = rdp.get_net_demand(b, n)  # assumed aligned to _cal length
                s1 = take_slice(hist, one_year_back_start, one_year_back_end, H)
                s2 = take_slice(hist, two_year_back_start, two_year_back_end, H)
                preds = w_last * s1 + w_prev * s2
                # if rdp_id == 'REG' and b == 100 and n == 'NEW':
                #     print(f"Debug REG 100 NEW preds: {preds}")
                #     print(f"  from s1: {s1}")
                #     print(f"  from s2: {s2}")
                    
                # If you want integers, do it explicitly outside (I’m leaving as float)
                for d, y in zip(pred_dates, preds):
                    rows.append((d, rdp_id, b, n, float(y)))

    out = pd.DataFrame(rows, columns=['date','rdp','denom','note_type','pred'])
    return out



def attach_predictions_to_network(network, preds_df):
    """
    Write predictions into each RDP object.

    After this runs, every rdp will have:
      - rdp.pred_index : pd.DatetimeIndex of the prediction horizon (same for all (b,n) in that rdp)
      - rdp._pred_store: dict keyed by (denom, note_type) -> np.array values aligned to pred_index
      - rdp.pred_at(denom, note_type, t=None, date=None): accessor

    Usage:
        preds = prediction_weighted_average(network, ph_start, ph_end)
        attach_predictions_to_network(network, preds)
        v = network.rdps['REG'].pred_at(10, 'FIT', t=3)
    """

    # Ensure consistent ordering of the horizon
    horizon = (preds_df[['date']]
               .drop_duplicates()
               .sort_values('date')
               .iloc[:, 0]
               .to_list())
    horizon_index = pd.DatetimeIndex(horizon)

    # Build a per-RDP pivot: (date, denom, note_type) → pred
    # Then reindex rows by the full horizon to guarantee alignment.
    for rdp in network.rdps.values():
        rdp_id = rdp.id
        sub = preds_df.loc[preds_df['rdp'] == rdp_id].copy()

        # Prepare the store and index on the RDP
        rdp.pred_index = horizon_index
        rdp._pred_store = {}

        if sub.empty:
            # No predictions for this rdp; create empty store and continue
            continue

        # Group by (denom, note_type) and align to the full horizon
        for (b, n), grp in sub.groupby(['denom', 'note_type']):
            s = (grp[['date', 'pred']]
                 .drop_duplicates('date')
                 .set_index('date')
                 .reindex(horizon_index)                # align to full horizon
                 .fillna(0.0)['pred']                   # fill missing with 0.0
                 .astype(float)
                 .to_numpy())

            rdp._pred_store[(int(b), str(n))] = s

        # Lightweight accessor methods (attached once)
        if not hasattr(rdp, 'pred_at'):
            def _pred_at(denom, note_type, t=None, date=None, _rdp=rdp):
                """Get prediction by t (0-based on rdp.pred_index) or by date."""
                key = (int(denom), str(note_type))
                arr = _rdp._pred_store.get(key, None)
                if arr is None:
                    return 0.0
                if t is not None:
                    return float(arr[t]) if 0 <= t < len(arr) else 0.0
                if date is not None:
                    pos = _rdp.pred_index.get_indexer([pd.Timestamp(date)], method='nearest')[0]
                    return float(arr[pos]) if pos != -1 else 0.0
                return 0.0

            def _pred_series(denom, note_type, _rdp=rdp):
                """Return a pandas.Series indexed by rdp.pred_index."""
                key = (int(denom), str(note_type))
                arr = _rdp._pred_store.get(key, None)
                if arr is None:
                    return pd.Series([], index=pd.DatetimeIndex([], name='date'), name='pred').to_numpy()
                return pd.Series(arr, index=_rdp.pred_index, name='pred').to_numpy()

            rdp.pred_at = _pred_at
            rdp.pred_series = _pred_series




def generate_lhs_scenarios(
    pred: np.ndarray,
    n_scen: int = 20,
    *,
    # EITHER give a symmetric percent half-width (e.g., 0.20 => ±20% → 40% band)…
    band_half_width_pct: float | None = None,

    seed: int | None = None
) -> np.ndarray:
    """
    LHS scenario generator inside a user-specified band around a point forecast.
    
    Parameters
    ----------
    pred : (T,) np.ndarray
        Point forecasts for the horizon.
    n_scen : int
        Number of scenarios to produce.
    band_half_width_pct : float, optional
        Symmetric half-width as a fraction of pred. Example: 0.20 => [0.8*pred, 1.2*pred]
        which is a central 40% band in the “engineering” sense.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    scenarios : (n_scen, T) np.ndarray
        LHS scenarios; each row is a trajectory across the horizon.
    """
    rng = np.random.default_rng(seed)
    pred = np.asarray(pred, dtype=float)
    T = pred.shape[0]


 
    L = pred * (1.0 - band_half_width_pct)
    U = pred * (1.0 + band_half_width_pct)

    # Guard against inverted bands
    L, U = np.minimum(L, U), np.maximum(L, U)

    # --- LHS uniforms in [0,1] then map to [L_t, U_t] per time t ---
    # Precompute stratum edges
    edges = np.linspace(0.0, 1.0, n_scen + 1)  # length n_scen+1
    scenarios = np.empty((n_scen, T), dtype=float)

    for t in range(T):
        # One u from each stratum, then shuffle (Latin permutation) at this t
        u = rng.uniform(edges[:-1], edges[1:])      # shape (n_scen,)
        rng.shuffle(u)
        # Map to numeric band at time t
        scenarios[:, t] = L[t] + u * (U[t] - L[t])
        # make it integer
        scenarios[:, t] = np.ceil(scenarios[:, t])

    return scenarios
# Example: get the prediction numpy array for (denom=10, note_type='FIT') at RDP='REG'


# 'scen' is (30, T). To fill your demand dict later, iterate rows as separate scenarios.
# Example: demand for scenario s at time t:
# demand_value = scen[s, t]
