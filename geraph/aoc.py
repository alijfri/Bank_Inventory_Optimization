
class AOC:
    def __init__(self, name):
        self.name = name
        self.rdps = []

    def add_rdp(self, rdp):
        if rdp not in self.rdps:
            self.rdps.append(rdp)

    def get_rdps(self):
        return self.rdps