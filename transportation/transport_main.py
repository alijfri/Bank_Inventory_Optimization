import pandas as pd
import numpy as np
from inputs import night_carrier_cost_per_bag, num_notes_in_bag,get_trip_config,get_rebalancing_params
#from transport_cost import get_cost_by_value
from .B_air_replenishment import trip_cost_air_B_replenishment
from .B_road_replenishment import trip_cost_road_B_replenishment
from .G_road_replenishment import trip_cost_road_G_replenishment
from .rebalancing import rebalance_cost


class Transportation:
    def __init__(self, file_path):
        #self.name = name
        self.cost_data = {}

        self.load_cost_data(file_path)

    def load_cost_data(self, file_path):
        sheets = ['B_road_replenishment','B_air_replenishment','A_road_replenishment','rdp_rebalancing']  # Add more if needed
        for sheet in sheets:
            try:
                df = pd.read_excel(file_path, sheet_name=sheet)
                df.columns = df.columns.str.strip()
                #df.set_index('condition', inplace=True)
                self.cost_data[sheet] = df
            except Exception as e:
                print(f"Failed to load {sheet}: {e}")

    def repcall_cost(self,origin,destination , num_notes, value_notes,
                  callback_value , callback_bags, activity,
                  round_trip=False, night_courier=False):
        if night_courier:
            num_bags = np.ceil(num_notes / num_notes_in_bag)
            return night_carrier_cost_per_bag * num_bags
        if origin in ['AOC1', 'AOC2']:
            name_center=destination
        else:
            name_center=origin
        trip_info=get_trip_config(name_center)
        mode=trip_info['mode']  # 'road' or 'air'
        ACC_name=trip_info['name']
        sheet_key = f"{ACC_name}_{mode}_replenishment"
        df = self.cost_data.get(sheet_key)
        #print(origin,destination)
        row = df[(df['From'] == origin) & (df['To'] == destination)]
        if df is None:
            raise ValueError(f"Cost data not available for key: {sheet_key}")
        if ACC_name == 'B'  and  activity in ['replenishment', 'callback']   and mode=='road':
            return trip_cost_road_B_replenishment(row,name_center, num_notes, value_notes,callback_value , callback_bags,round_trip)
        if ACC_name == 'B' and  activity in ['replenishment', 'callback']  and mode == 'air':
            #print('so far so good')
            return trip_cost_air_B_replenishment(destination,num_notes, value_notes, callback_value , callback_bags ,round_trip)
        if ACC_name == 'A' and activity in ['replenishment', 'callback'] and mode == 'road':
            return trip_cost_road_G_replenishment(name_center,row, num_notes, value_notes, callback_value , callback_bags ,round_trip)
        

    def compute_cost_rebalancing(self, origin, destination, num_notes, value_notes,round_trip=False):
        """
        Computes the cost of rebalancing between two RDPs.
        
        Parameters:
        - origin: Origin RDP ID
        - destination: Destination RDP ID
        - num_notes: Number of notes to transport
        - value_notes: Value of the notes
        
        Returns:
        - Cost of rebalancing
        """
        rebalance_info=get_rebalancing_params(config_rebalancing,origin,destination)
        mode=rebalance_info['mode']  # 'road' or 'air'
        ACC_name=rebalance_info['provider']
        sheet_key ="rdp_rebalancing"
        df = self.cost_data.get(sheet_key)
        row = df[(df['From'] == origin) & (df['To'] == destination)& (df['ACC'] == ACC_name)]
        if row.empty:
            row = df[(df['From'] == destination) & (df['To'] == origin)& (df['ACC'] == ACC_name)]
        # Placeholder for actual logic to compute rebalancing cost
        # This should be replaced with the actual cost calculation logic

        return rebalance_cost(row, origin, destination, num_notes, value_notes, round_trip=round_trip)

