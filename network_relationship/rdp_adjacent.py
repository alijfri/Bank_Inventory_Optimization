from inputs import adjacency_list_rdps


def rdp_adj(network):
    for rdp1_id, rdp2_id in adjacency_list_rdps:
        rdp1 = network.rdps[rdp1_id]
        rdp2 = network.rdps[rdp2_id]
        rdp1.add_adjacent_rdp(rdp2)
        rdp2.add_adjacent_rdp(rdp1) 
        
