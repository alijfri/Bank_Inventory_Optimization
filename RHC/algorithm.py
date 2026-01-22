from inputs import opt_start_date, opt_finish_date, start_date, finish_date, target_freq,rdp_to_be_solved
import pandas as pd
import os
from demand_SS_preprocess import setting_lower_bound
from inventory_files import set_initial_inventory


def build_initial_inventory_dict_from_network(network,rdp_to_be_solved, note_types=("NEW", "FIT")):
    """
    Build the initial_inventory dict for BanknoteInventoryModel from RDP.get_initial_inventory.

    Returns a dict:
        {(denom, rdp_id, note_type): count}
    """
    init = {}

    # I'm assuming network has something like network.denoms or similar.
    # If your denominations live somewhere else, adjust this line.


    for rdp_id, rdp in network.rdps.items():
        if rdp_id != rdp_to_be_solved:
            continue
        for denom in [5,10,20,50,100]:
            for note_type in note_types:
                qty = rdp.get_initial_inventory(denom, note_type)
                print(f"Initial inventory for RDP {rdp_id}, denom {denom}, type {note_type}: {qty}")
                if qty:
                    init[(denom, rdp_id, note_type)] = qty
                else:
                    init[(denom, rdp_id, note_type)] = 0
    return init




def RHC_algo2(
    optimization_model: str,
    network,
    rdp_to_be_solved: str,
    optimization_policy: str,
    opt_weeks: int,
    sim_weeks: int,
):
    """
    Rolling Horizon Control algorithm.

    - Optimization horizon is in weeks (opt_weeks).
    - Simulation horizon moves in blocks of sim_weeks * business days,
      restricted to [rhc_start_date, rhc_finish_date].
    """

    # --- 1. Pick the optimization model class ---
    print("Starting Optimization...")
    from weekly_opt.stochastic_OrTools import BanknoteInventoryModel

    # --- 2. Build calendars ---

    # Weekly calendar over the full historical horizon (must match rest of code)
    weeks_historical = pd.bdate_range(start_date, finish_date, freq="B").to_period(target_freq)
    weeks_unique = weeks_historical.drop_duplicates()  # PeriodIndex

    # Convert RHC start/end dates to weekly periods
    opt_start_period = pd.Timestamp(opt_start_date).to_period(target_freq)
    opt_end_period = pd.Timestamp(opt_finish_date).to_period(target_freq)

    # Map those periods to indices in weeks_unique
    opt_start_idx = weeks_unique.get_indexer([opt_start_period], method="bfill")[0]
    opt_end_idx = weeks_unique.get_indexer([opt_end_period], method="ffill")[0]

    # Global daily calendar
    cal_simulation = pd.bdate_range(start_date, finish_date, freq="B")

    # Daily RHC calendar, restricted to [rhc_start_date, rhc_finish_date]
    cal_rhc = pd.bdate_range(opt_start_date, opt_finish_date, freq="B")
    num_rhc_days = len(cal_rhc)

    # Map RHC daily index 0 -> global daily index in cal_simulation
    opt_start_global_idx = cal_simulation.get_indexer([pd.Timestamp(opt_start_date)], method="bfill")[0]

    # Simulation step in daily RHC indices
    days_per_week = 5  # business days; adjust if your target_freq implies something else
    sim_days = sim_weeks * days_per_week

    # --- 3. RHC Loop ---
    current_week_idx = opt_start_idx   # weekly index for optimization
    rhc_day_start_idx = 0              # daily index in cal_rhc for simulation



    opt_start_idx = current_week_idx
    opt_end_idx = current_week_idx + opt_weeks - 1  # fixed opt_weeks; assume horizon is large enough

    opt_start_period = weeks_unique[opt_start_idx]
    opt_end_period = weeks_unique[opt_end_idx]
    num_opt_weeks = opt_end_idx - opt_start_idx + 1

    opt_start_date_block = opt_start_period.start_time.normalize()
    opt_end_date_block = opt_end_period.end_time.normalize()

    print(
        f"  Optimization window: {opt_start_date_block.date()} "
        f"→ {opt_end_date_block.date()}"
    )


    set_initial_inventory(network, date_initial=opt_start_date_block)
    initial_inventory = build_initial_inventory_dict_from_network(network, rdp_to_be_solved)
    initial_unfit = {}  # or build a similar dict if you track unfit per RDP
    

    # --- 3.1b Build demand lower bounds / predictions for this optimization block ---
    setting_lower_bound(
        network,
        opt_start_date_block,
        opt_end_date_block,
        H_days=10, # 10 was the best result
        w_last=0.7,
        w_prev=0.3,
    )
    print(f"Solving the optimization model for RDP {rdp_to_be_solved} from {opt_start_date_block.date()} to {opt_end_date_block.date()}")
    # --- 3.1c Solve optimization model ---
    model = BanknoteInventoryModel(
        network,
        optimization_policy,
        initial_inventory,
        opt_start_date_block,
        opt_end_date_block,
        num_opt_weeks,
        initial_unfit=initial_unfit,
    )

    results = model.solve()
    print("  Finished optimization")
    model.report_results()
    # saving the s, S policy resulting from the optimization, creating the dataframe: 
    rows=[]
    rdp=network.rdps[rdp_to_be_solved]
    for b in [5,10,20,50,100]:
        for n in ['NEW','FIT']:
            big_S=round(rdp.opt_S[(b, n)]) 
            small_s=round(rdp.opt_s[(b, n)])
            rows.append({'Start_date': opt_start_date_block.date(),'End_date': opt_end_date_block.date(),'RDP': rdp_to_be_solved,'Denomination':b,'Note_Type':n,'small_s':small_s,'big_S':big_S})
    df_policy=pd.DataFrame(rows)

    policy_path=os.path.join("results",f"Optimization_{rdp_to_be_solved}.csv")
    if not os.path.exists("results"):
        os.makedirs("results")
    if not os.path.isfile(policy_path):
        df_policy.to_csv(policy_path, index=False)
    else:
        df_policy.to_csv(policy_path, mode='a', header=False, index=False)

    return