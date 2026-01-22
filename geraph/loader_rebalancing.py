
import pandas as pd


def load_rebalancing_routes_from_excel(file_path, RebalancingRoute,network):
    """
    Loads RebalancingRoute objects from Excel without assigning to RDPs or attaching transport logic.

    Parameters:
    - file_path: path to Excel file
    - RebalancingRoute: reference to the RebalancingRoute class
    - sheet_name: name of the sheet in Excel

    Returns:
    - List of route_dicts with attributes from Excel
    """
    df = pd.read_excel(file_path, sheet_name='OD_rebalancing')
    df.columns = df.columns.str.strip()

    required_cols = ['From', 'To', 'activity', 'Plan_leave', 'Arrival', 'mode', 'ACC']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    route_list = []

    for _, row in df.iterrows():
        route = RebalancingRoute(
            destination_rdp_id=row['To'],
            company=row['ACC'],
            mode=row['mode'],
            ship_day=row['Plan_leave'],
            arrival_day=row['Arrival']
            # transport_obj=None  # to be set later
        )
        # Store origin for later assignment to RDP
        route.origin_rdp_id = row['From']
        route.activity = row['activity']
        route_list.append(route)
    for route in route_list:
        origin_rdp = network.rdps.get(route.origin_rdp_id)
        if origin_rdp:
            origin_rdp.add_rebalancing_route(
                destination_rdp_id=route.destination_rdp_id,
                company=route.company,
                mode=route.mode,
                ship_day=route.ship_day,
                arrival_day=route.arrival_day
                # transport_obj=route.transport  # likely still None here, unless set earlier
            )

    return route_list 