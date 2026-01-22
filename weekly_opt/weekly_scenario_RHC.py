from inputs import start_date, finish_date ,target_freq
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)


# Build the same business-day calendar your 3-year series uses
weeks = pd.bdate_range(start_date, finish_date, freq="B").to_period(target_freq)
weeks_unique = weeks.drop_duplicates()                          # now unique


def _week_idx_of(ts: pd.Timestamp) -> int:
    """
    Map a timestamp to an index in weeks_unique using W-day periods,
    bfill/ffill just like your daily idx_of.
    """
    period = ts.to_period(target_freq)
    i = weeks_unique.get_indexer([period], method="bfill")[0]
    if i == -1:
        i = weeks_unique.get_indexer([period], method="ffill")[0]
    return i


def _take_slice_week(arr, i0, i1, H):
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

    if (i1 - i0 + 1) != H:
        # misaligned → keep it simple and return zeros
        return np.zeros(H, dtype=float)

    sl = np.asarray(arr[i0:i1+1], dtype=float)
    sl = np.where(np.isfinite(sl), sl, 0.0)
    return sl


def prediction_weighted_average_weekly(
    network,
    opt_start_date,
    opt_finish_date,
    w_last: float = 0.45,
    w_prev: float = 0.55,
):
    """
    Weekly version of your daily weighted average:

        pred(week t) = w_last * demand(t - 1y) + w_prev * demand(t - 2y)

    All indexing is based on weeks_unique (W-FRI). The horizon is exactly
    [opt_start_date, opt_finish_date] ⇒ indices [_start_idx, _end_idx].
    """



    target_week = pd.Timestamp(opt_start_date).to_period(target_freq)   # convert target to a weekly Period
    _start_idx = weeks_unique.get_indexer([target_week], method="bfill")[0]
    finish_week= pd.Timestamp(opt_finish_date).to_period(target_freq)
    _end_idx = weeks_unique.get_indexer([finish_week], method="ffill")[0]


    # Horizon weeks (Periods) and a timestamp representation if needed
    horizon_weeks = weeks_unique[_start_idx:_end_idx + 1]  # PeriodIndex
    H = len(horizon_weeks)

    # 1) Indices for 1y-back and 2y-back windows (same length H)
    one_year_back_start = _week_idx_of(
        pd.Timestamp(opt_start_date) - pd.DateOffset(years=1)
    )
    one_year_back_end = one_year_back_start + (H - 1)

    two_year_back_start = _week_idx_of(
        pd.Timestamp(opt_start_date) - pd.DateOffset(years=2)
    )
    two_year_back_end = two_year_back_start + (H - 1)

    denominations = [5, 10, 20, 50, 100]
    note_types = ["NEW", "FIT"]

    rows = []

    for rdp in network.rdps.values():
        rdp_id = rdp.id

        for b in denominations:
            for n in note_types:
                # --- Get weekly history aligned to weeks_unique ---
                hist = None

                # Option 1: method on the object
                if hasattr(rdp, "weekly_net_demand"):
                    hist = rdp.weekly_net_demand[(b,n)][1]  # assuming (week_array, values_array)


                s1 = _take_slice_week(hist, one_year_back_start, one_year_back_end, H)
                s2 = _take_slice_week(hist, two_year_back_start, two_year_back_end, H)
                
                
                # print(s1,"lsat year")
                # print(s2,"2nd last year")

                preds = w_last * s1 + w_prev * s2

                # store results; keep week as Period for clarity
                for wk, y in zip(horizon_weeks, preds):
                    rows.append((wk, rdp_id, b, n, float(y)))

    out = pd.DataFrame(rows, columns=["week", "rdp", "denom", "note_type", "pred"])
    return out


def attach_weekly_predictions_to_network(network, preds_df: pd.DataFrame):
    """
    Attach weekly predictions to each RDP, similar to your daily attach function.

    After this runs, every rdp will have:
      - rdp.pred_index : DatetimeIndex for the *end* of each week (Friday)
      - rdp._pred_store: dict keyed by (denom, note_type) -> np.array
      - rdp.pred_at(denom, note_type, t=None, date=None)
      - rdp.pred_series(denom, note_type)
    """

    # Horizon as weekly periods
    horizon_weeks = (
        preds_df[["week"]]
        .drop_duplicates()
        .sort_values("week")
        .iloc[:, 0]
        .to_list()
    )
    # Convert weekly Periods (W-FRI) to actual Friday timestamps
    horizon_index = pd.PeriodIndex(horizon_weeks, freq=target_freq).to_timestamp(how="end")

    for rdp in network.rdps.values():
        rdp_id = rdp.id
        sub = preds_df.loc[preds_df["rdp"] == rdp_id].copy()

        rdp.pred_index = horizon_index
        rdp._pred_store = {}

        if sub.empty:
            continue

        # Group by (denom, note_type) and align to the full weekly horizon
        for (b, n), grp in sub.groupby(["denom", "note_type"]):
            # grp["week"] is a PeriodIndex; convert to timestamps (Friday)
            s = (
                grp[["week", "pred"]]
                .drop_duplicates("week")
                .assign(
                    week_ts=lambda df: pd.PeriodIndex(
                        df["week"], freq=target_freq
                    ).to_timestamp(how="end")
                )
                .set_index("week_ts")
                .reindex(horizon_index)
                .fillna(0.0)["pred"]
                .astype(float)
                .to_numpy()
            )

            rdp._pred_store[(int(b), str(n))] = s

        # Attach accessors if not already present
        if not hasattr(rdp, "pred_at"):

            def _pred_at(denom, note_type, t=None, date=None, _rdp=rdp):
                """
                Get prediction by:
                  - t: integer index 0..H-1
                  - date: any date; we snap to the nearest Friday in rdp.pred_index
                """
                key = (int(denom), str(note_type))
                arr = _rdp._pred_store.get(key, None)
                if arr is None:
                    return 0.0

                if t is not None:
                    return float(arr[t]) if 0 <= t < len(arr) else 0.0

                if date is not None:
                    pos = _rdp.pred_index.get_indexer(
                        [pd.Timestamp(date)], method="nearest"
                    )[0]
                    return float(arr[pos]) if pos != -1 else 0.0

                return 0.0

            def _pred_series(denom, note_type, _rdp=rdp):
                """
                Return a pandas.Series of weekly predictions indexed by rdp.pred_index.
                """
                key = (int(denom), str(note_type))
                arr = _rdp._pred_store.get(key, None)
                if arr is None:
                    return pd.Series(
                        [], index=pd.DatetimeIndex([], name="week_end"), name="pred"
                    )
                return pd.Series(arr, index=_rdp.pred_index, name="pred")

            rdp.pred_at = _pred_at
            rdp.pred_series = _pred_series


def generate_lhs_scenarios(
    pred: np.ndarray,
    n_scen: int = 20,
    *,
    band_half_width_pct: float | None = None,
    seed: int | None = None
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pred = np.asarray(pred, dtype=float)
    T = pred.shape[0]

    L = pred * (1.0 - band_half_width_pct)
    U = pred * (1.0 + band_half_width_pct)

    L, U = np.minimum(L, U), np.maximum(L, U)

    edges = np.linspace(0.0, 1.0, n_scen + 1)
    scenarios = np.empty((n_scen, T), dtype=float)

    for t in range(T):
        u = rng.uniform(edges[:-1], edges[1:])
        rng.shuffle(u)
        scenarios[:, t] = L[t] + u * (U[t] - L[t])
        scenarios[:, t] = np.ceil(scenarios[:, t])

    return scenarios


def prediction_unfit_weighted_average_weekly(
    network,
    opt_start_date,
    opt_finish_date,
    w_last: float = 0.55,
    w_prev: float = 0.45,
):
    """
    Weekly weighted-average prediction for UNFIT notes, analogous to
    `prediction_weighted_average_weekly` for demand.

    For each RDP:
        unfit_pred(t) = w_last * unfit(t - 1y) + w_prev * unfit(t - 2y)

    All indexing is based on weeks_unique (W-FRI). The horizon is exactly
    [opt_start_date, opt_finish_date] ⇒ indices [_start_idx, _end_idx].
    """
    target_week = pd.Timestamp(opt_start_date).to_period(target_freq)   # convert target to a weekly Period
    _start_idx = weeks_unique.get_indexer([target_week], method="bfill")[0]
    finish_week= pd.Timestamp(opt_finish_date).to_period(target_freq)
    _end_idx = weeks_unique.get_indexer([finish_week], method="ffill")[0]



    # Horizon weeks (Periods) and weekly horizon length H
    horizon_weeks = weeks_unique[_start_idx:_end_idx + 1]  # PeriodIndex
    H = len(horizon_weeks)

    # 1) Indices for 1y-back and 2y-back windows (same length H)
    one_year_back_start = _week_idx_of(
        pd.Timestamp(opt_start_date) - pd.DateOffset(years=1)
    )
    one_year_back_end = one_year_back_start + (H - 1)

    two_year_back_start = _week_idx_of(
        pd.Timestamp(opt_start_date) - pd.DateOffset(years=2)
    )
    two_year_back_end = two_year_back_start + (H - 1)

    rows = []

    for rdp in network.rdps.values():
        rdp_id = rdp.id

        # --- Get weekly unfit history aligned to weeks_unique ---
        # From your note: rdp.weekly_unfit_demand[1][t]
        hist = None
        if hasattr(rdp, "weekly_unfit_demand"):
            # assuming weekly_unfit_demand = (week_array, values_array)
            hist = rdp.weekly_unfit_demand[1]

        # Take the 1-year-back and 2-year-back windows (length H)
        s1 = _take_slice_week(hist, one_year_back_start, one_year_back_end, H)
        s2 = _take_slice_week(hist, two_year_back_start, two_year_back_end, H)

        # Weighted average baseline
        preds = w_last * s1 + w_prev * s2

        # Store results; keep week as Period for clarity
        for wk, y in zip(horizon_weeks, preds):
            rows.append((wk, rdp_id, float(y)))

    out = pd.DataFrame(rows, columns=["week", "rdp", "unfit_pred"])
    return out

def attach_weekly_unfit_predictions_to_network(network, preds_df: pd.DataFrame):
    """
    Attach weekly unfit predictions to each RDP, similar to your daily attach function.

    After this runs, every rdp will have:
      - rdp.unfit_pred_index : DatetimeIndex for the *end* of each week (Friday)
      - rdp.unfit_pred_series()
    """

    # Horizon as weekly periods
    horizon_weeks = (
        preds_df[["week"]]
        .drop_duplicates()
        .sort_values("week")
        .iloc[:, 0]
        .to_list()
    )
    # Convert weekly Periods (W-FRI) to actual Friday timestamps
    horizon_index = pd.PeriodIndex(horizon_weeks, freq=target_freq).to_timestamp(how="end")

    for rdp in network.rdps.values():
        rdp_id = rdp.id
        sub = preds_df.loc[preds_df["rdp"] == rdp_id].copy()

        rdp.unfit_pred_index = horizon_index

        if sub.empty:
            continue

        # Build the series aligned to the full weekly horizon
        s = (
            sub[["week", "unfit_pred"]]
            .drop_duplicates("week")
            .assign(
                week_ts=lambda df: pd.PeriodIndex(
                    df["week"], freq=target_freq
                ).to_timestamp(how="end")
            )
            .set_index("week_ts")
            .reindex(horizon_index)
            .fillna(0.0)["unfit_pred"]
            .astype(float)
        )

        def _unfit_pred_series(_rdp=rdp):
            """
            Return a pandas.Series of weekly unfit predictions indexed by rdp.unfit_pred_index.
            """
            return s

        rdp.unfit_pred_series = _unfit_pred_series