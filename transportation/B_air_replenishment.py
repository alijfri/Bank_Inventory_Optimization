
import pandas as pd
import numpy as np
from inputs import bag_weight,num_notes_in_bag,get_trip_config


def trip_cost_air_B_replenishment(destination,num_notes, value_notes, callback_value , callback_bags ,round_trip=False):
        #print('so far so good')
        params = get_trip_config(destination)
        max_value = params["max_value"]
        liability = params["liability_per_1000"]
        per_kg_cost = params["per_kilo_cost"]
        fixed_one_way = params["cost_one_way"]
            ### Fixed parameters 
           # first_row = df.iloc[0]
        # if value_notes > max_value or callback_value > max_value:
        #     raise ValueError(f"Value of notes {value_notes} or callback value {callback_value} exceeds max value {max_value} for destination {destination}")

        #####
      
        if round_trip:
            fixed_two_way = params['cost_round_trip']
            first_way_liability_cost= (liability/1000)*value_notes
            first_way_weight_cost= per_kg_cost*bag_weight*(num_notes/num_notes_in_bag)
            second_way_liability_cost= (liability/1000)*callback_value
            second_way_weight_cost= per_kg_cost*(callback_bags*bag_weight)
            violation_cost=(np.max([0,callback_value-max_value])+np.max([0,value_notes-max_value]))*0
            total_cost= fixed_two_way + first_way_liability_cost + first_way_weight_cost + second_way_liability_cost + second_way_weight_cost+violation_cost
            return total_cost
        else:
            first_way_liability_cost= (liability/1000)*(callback_value+value_notes)
            first_way_weight_cost= per_kg_cost*bag_weight*(callback_bags+num_notes/num_notes_in_bag)
            violation_cost=(np.max([0,callback_value-max_value]))*0
            total_cost= fixed_one_way + first_way_liability_cost + first_way_weight_cost +violation_cost
            print(f"liability cost: {first_way_liability_cost}, weight cost: {first_way_weight_cost}, violation cost: {violation_cost}", flush=True)
            print(f"Total cost for one-way trip to {destination} with {num_notes} notes and {callback_bags} callback bags: {total_cost}", flush=True)
            return total_cost    


