import numpy as np

class RDC:
    def __init__(self,id,capacity_value):
        self.name = None
        self.id=id
        self.D_demand = {}  # key: (denomination, note_type), value: demand value for deposits
        self.W_demand = {}  # key: (denomination, note_type), value: demand value for Withdrawals
        self.unfit_value = None  # numpy array :total value of unfit notes per day
        self.unfit_bag = None    #  numpy array: number of unfit bags per day
        #self.profile = {}  # key: (denomination, note_type), value: 'Withdrawer' or 'Depositor'
        self.capacity_value = capacity_value  # single capacity value for regular days
        #self.holiday_capacity = None  # single capacity value for holidays
        self.adjacent_rdcs = []  # connected RDCs (for rebalancing)
        self.total_target = None
        self.bigS={}  # key: (denomination, note_type), value: small bigs value for the denomination and note type and RDC
        self.s={}  # key: (denomination, note_type), value: small s value for the denomination and note type and RDC
        self.bigS_fit_and_new = {}  # key: (denomination, note_type), value: fit and new values for the denomination and note type
        self.s_fit_and_new= {}  # key: (denomination, note_type), value: small s value for the denomination and note type

    ### Set Unfits
    def set_unfit(self, unfit_value, unfit_bag):
        """
        Set the daily unfit note values and number of bags for the RDC.

        """
        self.unfit_value = np.array(unfit_value, dtype=float)
        self.unfit_bag = np.array(unfit_bag, dtype=float)
    # --- D_demand ---
    def set_demand(self, denom, note_type, demand_array, demand_type):
        key = (denom, note_type.upper())
        if demand_type == 'D':
            self.D_demand[key] = demand_array
        elif demand_type == 'W':
            self.W_demand[key] = demand_array
    def get_demand(self, denom, note_type, demand_type):
        key = (denom, note_type.upper())
        if demand_type == 'D':
            return self.D_demand.get(key, np.array([], dtype=int))
        elif demand_type == 'W':
            return self.W_demand.get(key, np.array([], dtype=int))



    def get_net_demand(self, denom, note_type):
        key = (denom, note_type.upper())

        # Retrieve data or empty arrays if not present
        actual_W = np.array(self.W_demand.get(key, []), dtype=int)
        actual_D = np.array(self.D_demand.get(key, []), dtype=int)

        # Determine the proper length automatically
        max_len = max(len(actual_W), len(actual_D))

        # Handle case where both arrays are empty
        if max_len == 0:
            return np.array([], dtype=int)

        # Pad shorter array with zeros
        W = np.zeros(max_len, dtype=int)
        D = np.zeros(max_len, dtype=int)

        W[:len(actual_W)] = actual_W
        D[:len(actual_D)] = actual_D

        return (W - D).astype(int)


    def set_total_target(self, value):
        self.total_target = value
    
    def get_total_target(self):
        return self.total_target
    
    def set_bigS(self, denomination, note_type, bigS):
        key = (denomination, note_type.upper())
        self.bigS[key] = bigS
    def set_s(self, denomination, note_type, s):
        key = (denomination, note_type.upper())
        self.s[key] = s

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