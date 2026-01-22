import argparse
import pandas as pd
from RHC.algorithm import RHC_algo2
import numpy as np

# --- Custom Imports ---
from geraph import Network
from network_relationship import RDC_RDP_total_capacity, rdp_adj, AOC_to_rdp
from demand_SS_preprocess import (reader, fill_all_demands, fill_sS_policy)
from inputs import (
    RDC_path,
    demand_path,
    unfit_path,
    aoc_to_rdps,
    ss_path,
    optimization_policy,
    sim_weeks,
    opt_weeks,
    rdp_to_be_solved,
)
from unfit_demand import unfit_deposit
from inventory_files import  initialize_inventory_managers
from weekly_opt import weekly_demand, weekly_unfit_process


# --- New Imports for Clean Code ---
from utils_logging import setup_logging
from reporting import save_simulation_logs


def main():




    # --- Configuration ---
    pd.set_option('future.no_silent_downcasting', True)
    setup_logging()

    # --- 1. Network Initialization ---
    print("Initializing Network...")
    network = Network()
    RDC_RDP_total_capacity(RDC_path, network)

    # --- 2. Demand & Policy Loading ---
    print("Loading Demand and Policies...")
    df = reader(demand_path)
    fill_all_demands(df, network)
    #fill_sS_policy(ss_path, network)

    # --- 3. Calendar & Connectivity ---
    rdp_adj(network)
    AOC_to_rdp(network, aoc_to_rdps)

    # --- 4. Unfits & Routes ---
    unfit_deposit(unfit_path, network)

    # --- 5. Weekly Processing ---
    weekly_demand(network)
    weekly_unfit_process(network)

    # --- 6. Inventory Initialization ---
    initialize_inventory_managers(network)

    # --- 7. Optimization ---
    print("Starting Optimization...")

    # If RHC_algo2 accepts extra parameters, pass them here:
    #   rhc_start_date=rhc_start_date, rhc_finish_date=rhc_finish_date
    RHC_algo2(
        optimization_model='weekly_opt',  # deterministic_RHC, weekly_opt, ...
        network=network,
        rdp_to_be_solved=rdp_to_be_solved,
        optimization_policy=optimization_policy,
        opt_weeks=opt_weeks,
        sim_weeks=sim_weeks,
        # rhc_start_date=rhc_start_date,
        # rhc_finish_date=rhc_finish_date,
    )


if __name__ == "__main__":
    main()


# .\run.bat 2024-02-01 2024-03-01 WPG