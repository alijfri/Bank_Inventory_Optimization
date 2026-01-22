# inputs.py — Fabric paths
from pathlib import Path
import os, json
import argparse
from dataclasses import dataclass
import pandas as pd 
from datetime import datetime
# ---- Folders (adjust only if your layout changes) ----
FILES_DIR = Path("/lakehouse/default/Files")
DATA_DIR  = FILES_DIR / "Data"   # where your CSV/XLSX/JSON live
OPT_DIR   = FILES_DIR / "Codes"    # your code folder (not used below, kept for reference)

# ---- Data file paths ----
path_json               = DATA_DIR / "config_trip.json"
RDC_path                = DATA_DIR / "RDC_capacity.csv"
unfit_path              = DATA_DIR / "unfit.csv"
ss_path                 = DATA_DIR / "ss_policy.xlsx"
demand_path_newnotes    = DATA_DIR / "newnotes_updated.xlsx"
demand_path             = DATA_DIR / "updated_demand.csv"
initial_inventory_path  = DATA_DIR / "RDP_daily.csv"

# ---- Config loaders ----
def load_trip_config(path: Path):
    with open(path, "r") as f:
        return json.load(f)

CONFIG = load_trip_config(path_json)

def get_trip_config(city: str):
    city_info = CONFIG[city]
    selected_company = city_info["selected_company"]
    return city_info["companies"][selected_company]

# ---- Project settings ----
optimization_policy='joint_ss'  # 'separate_ss' or 'joint_ss'
today_date = datetime.today()
year = today_date.year
start_date = "2022-01-01"
finish_date='2025-03-10' # later it should be todays date
# Length of the optimization horizon in business days
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--opt_start_date",
        type=str,
        help="Optimization start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--opt_finish_date",
        type=str,
        help="Optimization finish date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--rdp_to_be_solved",
        type=str,
        help="RDP under consideration (e.g. A, B, C, ...)",
    )

    return parser.parse_args()

args = parse_args()
opt_start_date = args.opt_start_date
opt_finish_date = args.opt_finish_date
rdp_to_be_solved = args.rdp_to_be_solved# ---- Static network info ----
AOC_names = ["AOC1", "AOC2"]
aoc_to_rdps = {
    "AOC1": ["A", "B", "C", "D"],
    "AOC2": ["E", "F"],
}
adjacency_list_rdps = [
    ("A", "B"),

]
review_day_int = int(CONFIG[rdp_to_be_solved]['rep_plan_day'])
day_map = {
    0: "MON",
    1: "TUE",
    2: "WED",
    3: "THU",
    4: "FRI"
}
target_freq = f"W-{day_map[review_day_int]}"
sim_weeks=4
opt_weeks=7
# ---- Costs / constants ----
night_carrier_cost_per_bag = 310  # 
num_notes_in_bag           = 44000
bag_weight                 = 45  # kg

# ---- Helper for rebalancing params from JSON config ----
def get_rebalancing_params(config, origin_rdp, target_rdp):
    route_key   = f"{origin_rdp}->{target_rdp}"
    reverse_key = f"{target_rdp}->{origin_rdp}"
    routes = config["routes"]

    if route_key in routes:
        route = routes[route_key]
    elif reverse_key in routes:
        route = routes[reverse_key]
    else:
        return {"error": f"No route found between {origin_rdp} and {target_rdp}"}

    selected_company = route["selected_company"]
    return route["companies"][selected_company]

# ---- Cost struct ----
from dataclasses import dataclass
@dataclass
class CostParams:
    c_er: float = 1.0   # per emergency note
    c_h: float = 1.0    # per trigger (fixed replenishment)
    c_q: float = 1.0    # per replenished note (variable)
    c_O: float = 1.0    # per overcap event
    c_w: float = 1.0    # per callback note
    c_x: float = 1.0    # per NEW→FIT converted note

costs = CostParams(
    c_er=3000.0,
    c_h=50.0,
    c_q=0.01,      # e.g., $0.02 per note moved
    c_O=50.0,
    c_w=0.01,
    c_x=1
)

