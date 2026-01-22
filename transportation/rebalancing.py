from .trasnport_cost import get_cost_by_value
import numpy as np
from inputs import num_notes_in_bag,get_rebalancing_params



def rebalance_cost(row,origin, destination, num_notes, value_notes, return_notes, return_value, round_trip=False):
    """
    Computes the cost of rebalancing between two RDPs.
    
    Parameters:
    - origin: The origin RDP.
    - destination: The destination RDP.
    - num_notes: Number of notes to be transported.
    - value_notes: Value of the notes.
    - callback_value: Value for callback.
    - callback_bags: Number of bags for callback.
    - round_trip: Boolean indicating if it's a round trip.
    
    Returns:
    - Total cost of rebalancing.
    """
    rebalance_info=get_rebalancing_params(config_rebalancing,origin,destination)
    ACC_name=rebalance_info['provider']
    max_value = rebalance_info["max_value"]
    penalty_1_2 = rebalance_info["penalty"]
    penalty_2_4= rebalance_info["penalty_2_4"]
    max_bags_per_trip = rebalance_info["max_bags_per_trip"]
    max_containers_per_trip = rebalance_info["max_containers_per_trip"]
    bags_per_container = rebalance_info["bags_per_container"]
    max_weight= rebalance_info["max_weight"]
    # Placeholder for the actual logic to compute the cost
    # --- Outbound leg ---
    num_bags_outbound = np.ceil(num_notes / num_notes_in_bag)
    containers_outbound = np.ceil(num_bags_outbound / bags_per_container)
    trips_containers_outbound = np.ceil(containers_outbound / max_containers_per_trip)
    # --- Return leg (unfit) ---
    return_bags=np.ceil(return_notes / num_notes_in_bag)
    containers_return = np.ceil(return_bags / bags_per_container)
    if np.max(containers_outbound, containers_return) >= max_containers_per_trip:
        return print(f"Number of allowed {np.max(containers_outbound, containers_return)} exceeds maximum allowed containers {max_containers_per_trip} for rebalancing from {origin} to {destination}.")
        
    if np.max(value_notes, return_value) > max_value:
        return print(f"Value {np.max(value_notes, return_value)} exceeds maximum allowed value {max_value} for rebalancing from {origin} to {destination}.")

    #######
    value_this_trip=np.max(value_notes,return_value)
    if round_trip:
        if ACC_name=='F':
            if np.max(containers_outbound,containers_return)>=10:
                
                row=row[row['condition'] == 'two_way_10_18']
                total_cost=get_cost_by_value(value_this_trip, row, round_trip=True)
            else:
                
                row=row[row['condition'] == 'two_way']
                total_cost=get_cost_by_value(value_this_trip, row, round_trip=True)
        else:
                row=row[row['condition'] == 'two_way']
                total_cost=get_cost_by_value(value_this_trip, row, round_trip=True)

    else:
        if ACC_name=='F':
            if np.max(containers_outbound,containers_return)>=10:
                row=row[row['condition'] == 'one_way_10_18']
                total_cost=get_cost_by_value(value_this_trip, row, round_trip=False)
            else:
        
                row=row[row['condition'] == 'one_way']
                total_cost=get_cost_by_value(value_this_trip, row, round_trip=False)
        else:
            row=row[row['condition'] == 'one_way']
            total_cost=get_cost_by_value(value_this_trip, row, round_trip=False)
    # --- Penalty for exceeding max containers per trip ---
    exceed_continers = np.max([trips_containers_outbound, containers_return]) - max_containers_per_trip
    if exceed_continers > 0:
        total_cost += penalty_1_2 * (np.ceil(exceed_continers/4))


    return total_cost