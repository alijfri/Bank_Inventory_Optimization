
from inputs import CONFIG # Ensure you bring necessary imports
from inventory_files import InventoryManager
from .initial_inv import set_initial_inventory
import pandas as pd
from inputs import start_date, finish_date
from inputs import opt_start_date, opt_finish_date,target_freq

# Weekly calendar over the full historical horizon (this must match the rest of your code)
weeks_historical = (
    pd.bdate_range(start_date, finish_date, freq="B").to_period(target_freq)
)
weeks_unique = weeks_historical.drop_duplicates()  # PeriodIndex
opt_start_period = pd.Timestamp(opt_start_date).to_period(target_freq)
opt_start_idx = weeks_unique.get_indexer([opt_start_period], method="bfill")[0]

# --- 3. RHC Loop over weeks, stepping by sim_weeks ---
current_idx = opt_start_idx
opt_start_idx = current_idx
opt_start_period = weeks_unique[opt_start_idx]
opt_start_date_block = opt_start_period.start_time.normalize()




_cal = pd.bdate_range(opt_start_date, opt_finish_date, freq="B")
def initialize_inventory_managers(network):
    set_initial_inventory(network, date_initial=opt_start_date_block)
    for rdp in  network.get_rdp():
        #horizon = len(rdp.calendar_index) 
        horizon=len(_cal)
        #dow = rdp.dow
        rep_arrival_day = CONFIG[rdp.id]['rep_arrival_day']
        callback_leaves_day = CONFIG[rdp.id]['callback_leaves_day']
        AOC_received_day = CONFIG[rdp.id]['AOC_received_day']
        lead_time = CONFIG[rdp.id]['lead_time']
        
        rdp.unfit_inventory = InventoryManager(
        location_id=rdp.id,
        capacity=rdp.get_capacity(),      # or separate capacity for unfit, if needed
        s=0,                              # s/S aren't used for unfit, set to dummy
        S=0,
        rep_arrival_day=rep_arrival_day,
        lead_time=lead_time,
        callback_leaves_day=callback_leaves_day,
        AOC_received_day=AOC_received_day,
        horizon=horizon,
        denomination=None,
        note_type='UNFIT',
       # dow=dow,
        initial_inventory=None
    )

        rdp.inventory_managers = {}
        for denom in [5,10,20,50,100]:
                for note_type in ['NEW','FIT']:
                    note_type = note_type.upper()
                    s = rdp.get_s(denom, note_type)
                    S = rdp.get_bigS(denom, note_type)
                    cap = rdp.get_capacity()
                    key = (denom, note_type)
                    initial_inv=rdp.get_initial_inventory(denom, note_type)
                    rdp.inventory_managers[key] = InventoryManager(
                        location_id=rdp.id,
                        capacity=cap,
                        s=s,
                        S=S,
                        rep_arrival_day=rep_arrival_day,
                        lead_time=lead_time,
                        callback_leaves_day=callback_leaves_day,
                        AOC_received_day=AOC_received_day,
                        horizon=horizon,
                        denomination=denom,             # ✅ pass this
                        note_type=note_type,             # ✅ and this
                        #dow=dow,
                        initial_inventory=initial_inv
                )