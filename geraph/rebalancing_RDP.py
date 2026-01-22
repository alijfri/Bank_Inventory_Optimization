class RebalancingRoute:
    def __init__(self, destination_rdp_id, company, mode, ship_day, arrival_day):
        self.destination_rdp_id = destination_rdp_id
        self.company = company
        self.mode = mode
        self.ship_day = ship_day
        self.arrival_day = arrival_day
        

    # def compute_cost_rebalancing(self, origin, destination, num_notes, value_notes):
    #     return self.transport.trip_cost(
    #         origin=origin,
    #         destination=destination,
    #         mode=self.mode,
    #         ACC_name=self.company,
    #         num_notes=num_notes,
    #         value_notes=value_notes,
    #     )
