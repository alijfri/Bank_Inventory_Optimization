from inputs import AOC_names
from geraph import AOC




def AOC_to_rdp(network, aoc_to_rdps):
    """
    Creates AOC objects, adds them to the network, and associates RDPs with their respective AOCs.
    
    Parameters:
    - network: the Network object
    - aoc_to_rdps: dict mapping AOC names to lists of RDP IDs
    """
    for aoc_name, rdp_ids in aoc_to_rdps.items():
        aoc = AOC(aoc_name)
        network.add_aoc(aoc)

        for rdp in network.get_rdp():
            if rdp.id in rdp_ids:
                aoc.add_rdp(rdp)
