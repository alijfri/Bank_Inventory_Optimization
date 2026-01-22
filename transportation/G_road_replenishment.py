from .trasnport_cost import get_cost_by_value
from inputs import num_notes_in_bag,get_trip_config,CONFIG
import numpy as np




    


def trip_cost_road_G_replenishment(destination,row, num_notes, value_notes,callback_value , callback_bags,round_trip=False):
    if destination not in CONFIG:
        raise ValueError(f"Unknown destination: {destination}")
    params = get_trip_config(destination)
    max_value_G_road_rep = params["max_value"]
    penalty = params["penalty"]
    max_bags_per_trip = params["max_bags_per_trip"]
    max_containers_per_trip = params["max_containers_per_trip"]
    bags_per_container = params["bags_per_container"]

    # --- Outbound leg ---
    num_bags_outbound = np.ceil(num_notes / num_notes_in_bag)
    containers_outbound = np.ceil(num_bags_outbound / bags_per_container)
    trips_containers_outbound = np.ceil(containers_outbound / max_containers_per_trip)

    # --- Return leg (unfit) ---

    num_bags_return = callback_bags
    containers_return = np.ceil(num_bags_return / bags_per_container)
    trips_containers_return = np.ceil(containers_return / max_containers_per_trip)

    value_this_trip_out= np.max([value_notes, callback_value])
    if value_this_trip_out>max_value_G_road_rep:
        return print(f"Value {value_this_trip_out} exceeds maximum allowed value {max_value_G_road_rep} for road replenishment to {destination}.")
    if round_trip:

        if np.max([containers_outbound,containers_return])>=10:
            
            row=row[row['condition'] == 'two_way_10_18']
            total_cost=get_cost_by_value(value_this_trip_out, row, round_trip=True)
        else:
            
            row=row[row['condition'] == 'two_way']
            total_cost=get_cost_by_value(value_this_trip_out, row, round_trip=True)

    else:
        if np.max([containers_outbound,containers_return])>=10:
           
            row=row[row['condition'] == 'one_way_10_18']
            total_cost=get_cost_by_value(value_this_trip_out, row, round_trip=False)
        else:
      
            row=row[row['condition'] == 'one_way']
            total_cost=get_cost_by_value(value_this_trip_out, row, round_trip=False)
    # --- Penalty for exceeding max containers per trip ---
    exceed_continers = np.max([trips_containers_outbound, trips_containers_return]) - max_containers_per_trip
    if exceed_continers > 0:
        total_cost += penalty * (np.ceil(exceed_continers/4))
    # if trips_containers_outbound>=max_containers_per_trip or trips_containers_return>=max_containers_per_trip:
    #     total_cost+=penalty*()
    return total_cost





