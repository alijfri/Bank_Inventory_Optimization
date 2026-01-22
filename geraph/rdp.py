import numpy as np
from .rebalancing_RDP import RebalancingRoute

class RDP:
    def __init__(self, id):
        self.id = id
        self.D_demand = {}  # key: (denomination, note_type), value: demand value for deposits
        self.W_demand = {}  # key: (denomination, note_type), value: demand value for Withdrawals
        self.rdcs = []
        self.adjacent_rdps = []  # connected RDPs (for rebalancing)
        self.total_target=None
        self.rebalancing_routes = {}
        self.bigS={}  # key: (denomination, note_type), value: small bigs value for the denomination and note type and RDP
        self.s={}  # key: (denomination, note_type), value: small s value for the denomination and note type and RDP
        self.lower_bound={}  # key: (denomination, note_type), value: lower bound for the denomination and note type and RDP
        self.bigS_fit_and_new = {}  # key: (denomination), value: fit and new values for the denomination 
        self.s_fit_and_new= {}  # key: (denomination), value: small s value for the denomination of fit and new notes
        self.unfit_value_inventory = np.zeros(366)
        self.unfit_bag_inventory = np.zeros(366)
        self.calendar_dates=[]
        self.initial_inventory = {}  # key: (denomination, note_type), value: initial inventory count

        
    def set_initial_inventory(self, denom, note_type, count):
        """
        Store the initial inventory for a denomination and note type.
        Overwrites if value already exists (intended behavior).
        """
        self.initial_inventory[(denom, note_type)] = count

    def get_initial_inventory(self, denom, note_type):
        """
        Retrieve inventory value (0 if missing).
        """
        return self.initial_inventory.get((denom, note_type), 0)
        
    def get_total_unfit_value(self, t):
        return self.unfit_value_inventory[t]

    def get_total_unfit_bags(self, t):
        return self.unfit_bag_inventory[t]

    def add_rebalancing_route(self, destination_rdp_id, company, mode, ship_day, arrival_day):
        route = RebalancingRoute(destination_rdp_id, company, mode, ship_day, arrival_day)
        self.rebalancing_routes.setdefault(destination_rdp_id, []).append(route)

    def get_rebalancing_routes(self, destination_rdp_id=None):
        if destination_rdp_id:
            return self.rebalancing_routes.get(destination_rdp_id, [])
        return self.rebalancing_routes


    def unfit_deposit_value(self, expected_length):
        bags_containers = np.zeros(expected_length, dtype=float)
        unfit_value = np.zeros(expected_length, dtype=float)

        for rdc in self.rdcs:
            if rdc.unfit_bag is not None and rdc.unfit_value is not None:
                bags_containers += rdc.unfit_bag
                unfit_value += rdc.unfit_value

        return (
    np.nan_to_num(bags_containers, nan=0),
    np.nan_to_num(unfit_value, nan=0)
)
    
    def add_adjacent_rdp(self, rdp):
        if rdp not in self.adjacent_rdps:
            self.adjacent_rdps.append(rdp)

    def get_adjacent_rdps(self):
        return self.adjacent_rdps
    
    def add_rdc(self, rdc):
        if rdc not in self.rdcs:
            self.rdcs.append(rdc)

    def get_rdcs(self):
        return self.rdcs
    def compute_all_demands(self):
        all_D_keys = set()
        all_W_keys = set()

        # Step 1: collect all keys
        for rdc in self.rdcs:
            all_D_keys.update(rdc.D_demand.keys())
            all_W_keys.update(rdc.W_demand.keys())

        # Step 2: dynamically determine array length
        def get_shape_for_key(key, demand_type):
            for rdc in self.rdcs:
                if demand_type == 'D' and key in rdc.D_demand:
                    return rdc.D_demand[key].shape
                if demand_type == 'W' and key in rdc.W_demand:
                    return rdc.W_demand[key].shape
            return (0,)  # fallback if nothing is found

        # Step 3: sum arrays for each key
        for key in all_D_keys:
            shape = get_shape_for_key(key, 'D')
            self.D_demand[key] = sum(
                (rdc.D_demand.get(key, np.zeros(shape)) for rdc in self.rdcs)
            )

        for key in all_W_keys:
            shape = get_shape_for_key(key, 'W')
            self.W_demand[key] = sum(
                (rdc.W_demand.get(key, np.zeros(shape)) for rdc in self.rdcs)
            )
    # def get_net_demand(self, denom, note_type, expected_length):

    #     net = np.zeros(expected_length)

    #     for rdc in self.rdcs:
    #         net += rdc.get_net_demand(denom, note_type.upper(), expected_length)
    #     return net.astype(int)
    
    def get_net_demand(self, denom, note_type):
        note_type = note_type.upper()

        # --- Determine max length across all RDCs ---
        lengths = [
            len(rdc.get_net_demand(denom, note_type))
            for rdc in self.rdcs
            if len(rdc.get_net_demand(denom, note_type)) > 0
        ]

        if not lengths:  # No data from any RDC
            return np.array([], dtype=int)

        max_len = max(lengths)

        # --- Initialize net demand vector ---
        net = np.zeros(max_len, dtype=int)

        # --- Sum aligned RDC net demands ---
        for rdc in self.rdcs:
            nd = rdc.get_net_demand(denom, note_type)
            if len(nd) == 0:
                continue
            net[:len(nd)] += nd  # pad shorter series with zeros automatically

        return net
    def get_demand(self, denom, note_type, demand_type):
        key = (denom, note_type.upper())
         # --- Determine max length across all RDCs ---
        lengths = [
            len(rdc.get_net_demand(denom, note_type))
            for rdc in self.rdcs
            if len(rdc.get_net_demand(denom, note_type)) > 0
        ]

        if not lengths:  # No data from any RDC
            return np.array([], dtype=int)

        max_len = max(lengths)
        demand_D=np.zeros(max_len, dtype=int)
        if demand_type == 'D':
            for rdc in self.rdcs:
                nd = rdc.get_demand(denom, note_type, demand_type)
                if len(nd) == 0:
                    continue
                demand_D[:len(nd)] += nd  # pad shorter series with zeros automatically
        elif demand_type == 'W':
            for rdc in self.rdcs:
                nd = rdc.get_demand(denom, note_type, demand_type)
                if len(nd) == 0:
                    continue
                demand_D[:len(nd)] += nd  # pad shorter series with zeros automatically
        return demand_D

        
    def get_capacity(self):
        total = 0
        for rdc in self.rdcs:
            cap = rdc.capacity_value
            if cap is not None:
                total += cap
        return total

    def set_bigS(self, denomination, note_type, value):
        key = (denomination, note_type.upper())
        self.bigS[key] = value

        #self.bigS[(denomination, note_type)] = value

    def set_s(self, denomination, note_type, value):
        key = (denomination, note_type.upper())
        self.s[key] = value
        #self.s[(denomination, note_type)] = value
    def set_lower_bound(self, denomination, note_type, value):
        key = (denomination, note_type.upper())
        self.lower_bound[key] = value
    def get_lower_bound(self, denomination, note_type):
        key = (denomination, note_type.upper())
        return self.lower_bound.get(key, 0)
    def get_bigS(self, denomination, note_type):
        key = (denomination, note_type.upper())

        return self.bigS.get(key, 0)
    def get_s(self, denomination, note_type):
        key = (denomination, note_type.upper())
        return self.s.get(key, 0)
    def get_bigS_fit_and_new(self, denomination):
        return self.bigS_fit_and_new.get(denomination, 0)

    def get_s_fit_and_new(self, denomination):
        return self.s_fit_and_new.get(denomination, 0)


    def get_rdp_total_target(self):
        return sum(rdc.total_target for rdc in self.rdcs)
    
