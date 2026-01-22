# Bank Inventory Optimization

A simulation–optimization framework for banknote inventory management using stochastic programming and receding horizon control. The project models multi-echelon cash logistics networks, integrates (s, S) policies, replenishment, callbacks, and rebalancing decisions, and evaluates operational cost and GHG emissions under demand uncertainty.

## Project structure

- `main.py`: Entry point that wires together data loading, network initialization, weekly preprocessing, and the receding-horizon optimization run.
- `inputs.py`: Centralized configuration for data paths, dates, optimization settings, and cost parameters.
- `RHC/`: Receding-horizon controller logic and optimization orchestration.
- `weekly_opt/`: Weekly demand/unfit aggregation and stochastic OR-Tools optimization model.
- `inventory_files/`: Inventory state, initialization, replenishment, callback, and simulation utilities.
- `transportation/`: Transportation cost models for replenishment and rebalancing.
- `demand_SS_preprocess/`: Demand ingestion, (s, S) policy processing, calendars, and lower bounds.
- `network_relationship/`: Network connectivity, capacities, and RDC/RDP topology utilities.
- `geraph/`: Core network entities (RDC/RDP/AOC) and rebalancing route loading.
- `daily_scenario/`: Daily scenario generation helpers.
- `unfit_demand/`: Unfit (soiled) banknote processing.
- `reporting.py`, `utils_logging.py`: Logging and simulation output helpers.
- `run.bat`: Windows helper script for running `main.py` with CLI arguments.

## Setup

1. Create and activate a Python environment (3.9+ recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Configure data inputs

Data locations, dates, and optimization settings live in `inputs.py`. Update the file paths to match your storage layout (for example, `FILES_DIR`/`DATA_DIR` paths, and `config_trip.json`, demand, and capacity inputs).

Key inputs expected by the pipeline include:
- `config_trip.json` (transport and rebalancing routes)
- `RDC_capacity.csv` (RDC/RDP capacity)
- `unfit.csv` (unfit deposit data)
- `ss_policy.xlsx` (s, S policy input)
- `updated_demand.csv` / `newnotes_updated.xlsx` (demand data)
- `RDP_daily.csv` (initial inventories)

## Run the simulation/optimization

Use `main.py` with the required CLI arguments:

```bash
python main.py --opt_start_date YYYY-MM-DD --opt_finish_date YYYY-MM-DD --rdp_to_be_solved <RDP_ID>
```

Windows users can also call:

```bat
run.bat 2024-02-01 2024-03-01 RDP_name
```

## Outputs

Logs and any saved simulation outputs are managed via `utils_logging.py` and `reporting.py`. Adjust those modules if you need different logging directories or output formats.