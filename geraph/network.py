class Network:
    def __init__(self):
        self.aocs = {}
        self.rdps = {}  # key: RDP.id, value: RDP object
        self.rdcs = {}  # key: RDC.id, value: RDC object

    def add_aoc(self, aoc):
        self.aocs[aoc.name] = aoc

    def add_rdp(self, rdp):
        self.rdps[rdp.id] = rdp

    def add_rdc(self, key, rdc):
        self.rdcs[key] = rdc

    def connect_rdp_to_aoc(self, rdp_name, aoc_name):
        rdp = self.rdps.get(rdp_name)
        aoc = self.aocs.get(aoc_name)
        if rdp and aoc:
            aoc.add_rdp(rdp)
    def get_aoc_for_rdp(self, rdp_id):
        for aoc in self.aocs.values():
            for rdp in aoc.rdps:
                if rdp.id == rdp_id:
                    return aoc
        return None

    def connect_rdc_to_rdp(self, rdc_key, rdp_id):
        rdc = self.rdcs.get(rdc_key)
        rdp = self.rdps.get(rdp_id)
        if rdc and rdp:
            rdp.add_rdc(rdc)

    def connect_adjacent_rdcs(self, rdc1_name, rdc2_name):
        rdc1 = self.rdcs.get(rdc1_name)
        rdc2 = self.rdcs.get(rdc2_name)
        if rdc1 and rdc2:
            rdc1.add_adjacent_rdc(rdc2)
            rdc2.add_adjacent_rdc(rdc1)  # assume undirected connection

    # def get_rdc(self, name):
    #     return self.rdcs.get(name)

    def get_rdp(self):
        return self.rdps.values()

    def get_aoc(self):
        return self.aocs.values()

