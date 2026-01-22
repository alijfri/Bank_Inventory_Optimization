import pandas as pd
from geraph import RDC, RDP


def RDC_RDP_total_capacity(file_path, network):
    df = pd.read_csv(file_path)
    df['RDC Capacity'] = df['RDC Capacity'].fillna(0)
    df['RDC Capacity'] = df['RDC Capacity'].astype(int)
    for _, row in df.iterrows():
        rdp_id = row['REGION_ID']
        rdc_id = row['FI_ORG_ID']
        capacity = row['RDC Capacity']

        # Ensure RDP exists
        if rdp_id not in network.rdps:
            network.add_rdp(RDP(rdp_id))

        # Use (rdp_id, rdc_id) as composite key
        rdc_key = (rdp_id, rdc_id)
        if rdc_key not in network.rdcs:
            network.add_rdc(rdc_key, RDC(rdc_id, capacity))   # Still pass original ID to RDC

        # Connect RDC to RDP
        network.connect_rdc_to_rdp(rdc_key, rdp_id)

    # (Optional) Post-processing: call rdp.get_capacity()
    for rdp in network.rdps.values():
        rdp.get_capacity()
