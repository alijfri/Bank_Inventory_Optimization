
from .trasnport_cost import get_cost_by_value
from inputs import num_notes_in_bag,get_trip_config
import numpy as np


def trip_cost_road_B_replenishment(row,ACC_name ,num_notes, value_notes, callback_value , callback_bags,round_trip=False):
        params = get_trip_config(ACC_name)
        max_value_B_road_rep = params["max_value"]
        if value_notes > max_value_B_road_rep:
            raise ValueError(f"Value of notes ({value_notes}) exceeds maximum allowed for B road replenishment ({max_value_B_road_rep}).")
        #penalty = params["penalty"]
        max_bags_per_trip = params["max_bags_per_trip"]
        max_containers_per_trip = params["max_containers_per_trip"]
        bags_per_container = params["bags_per_container"]
        

     #### For B repleishment by road only
        # --- Outbound leg ---
        num_bags_outbound = np.ceil(num_notes / num_notes_in_bag)
        containers_outbound = np.ceil(num_bags_outbound / bags_per_container)
        trips_bags_outbound = np.ceil(num_bags_outbound / max_bags_per_trip)
        trips_containers_outbound = np.ceil(containers_outbound / max_containers_per_trip)
        trips_outbound = int(max(trips_bags_outbound, trips_containers_outbound))

        # --- Return leg (unfit,surpluss) ---
        num_bags_return = callback_bags
        containers_return = np.ceil(num_bags_return / bags_per_container)
        trips_bags_return = np.ceil(num_bags_return / max_bags_per_trip)
 
 

        if round_trip:
            num_trips = 0
            #row = df.loc['two_way']
            row=row[row['condition'] == 'two_way']
            
            bags_out = num_bags_outbound
            bags_return = num_bags_return

            notes_remaining = num_notes
            value_remaining = value_notes
            unfit_value_remaining = callback_value

            total_fixed_cost = 0

            while bags_out > 0 or bags_return > 0:
                
                # --- Determine how many bags go in this truck ---
                bags_this_trip_out = min(max_bags_per_trip, bags_out)
                bags_this_trip_return = min(max_bags_per_trip, bags_return)

                # --- Outbound value ---
                notes_this_trip_out = bags_this_trip_out * num_notes_in_bag

                value_this_trip_out = min(value_remaining, max_value_B_road_rep)

                # --- Return value ---
                value_this_trip_return = (bags_this_trip_return / num_bags_return) * callback_value if num_bags_return > 0 else 0
                value_this_trip_return = min(callback_value, max_value_B_road_rep)

            
                # --- Get cost ---
                if num_trips >=2:
                    trip_cost = get_cost_by_value(value_this_trip_out, row, round_trip=False)+ get_cost_by_value(value_this_trip_return, row, round_trip=False)
                    total_fixed_cost += trip_cost
                else:
                    trip_cost = get_cost_by_value(value_this_trip_out, row, round_trip=True)+ get_cost_by_value(value_this_trip_return, row, round_trip=True)
                    total_fixed_cost += trip_cost
                    num_trips += 1

                # --- Update remaining load ---
                bags_out -= bags_this_trip_out
                notes_remaining -= notes_this_trip_out
                value_remaining -= value_this_trip_out

                bags_return -= bags_this_trip_return
                unfit_value_remaining -= value_this_trip_return

        else:
           # row = df.loc['one_way']
            row=row[row['condition'] == 'one_way']
            bags_out = num_bags_outbound
            
            value_remaining = value_notes

            total_fixed_cost = 0
            num_trips = 0

            while bags_out > 0:
                # Fill one truck with up to max_bags_per_trip
                bags_this_trip_out = min(max_bags_per_trip, bags_out)

                # Compute the value of notes in this truck
                
                value_this_trip_out = min(value_remaining, max_value_B_road_rep)
                    
                # Get cost for this one-way shipment
                trip_cost = get_cost_by_value(value_this_trip_out, row, round_trip=False)
                total_fixed_cost += trip_cost
                num_trips += 1

                # Update remaining load
                bags_out -= bags_this_trip_out
                
                value_remaining -= value_this_trip_out

        return total_fixed_cost
