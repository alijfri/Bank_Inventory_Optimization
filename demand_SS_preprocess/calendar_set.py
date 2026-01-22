import numpy as np
import pandas as pd

#from inputs import years_to_include

# def calndar_dates(path,network):
#     df= pd.read_csv(path)
#     df['INVENT_SNAPSHOT_DT']=pd.to_datetime(df['INVENT_SNAPSHOT_DT'], errors='coerce')
#     df = df[df['INVENT_SNAPSHOT_DT'].dt.year.isin(years_to_include)]
#     df = df.sort_values('INVENT_SNAPSHOT_DT', ascending=True)
#     base_df=df.copy()
#     for rdp_id, rdp in network.rdps.items():
#         df_rdp = base_df[base_df['REGION_ID'] == rdp_id]
        
#         unique_dates = (
#         df_rdp['INVENT_SNAPSHOT_DT']
#             .dt.normalize()
#             .drop_duplicates()
#         .sort_values()
#         .dt.strftime('%Y-%m-%d')
#         .tolist())
        
#         rdp.calendar_dates=unique_dates
def calndar_dates(network):
    """
    For each RDP in the network, set rdp.calendar_dates to the list of
    business days (Mon–Fri) for the given year.
    """
    # Generate business days (Mon–Fri) for that year
    start = f"{years_to_include[0]}-01-01"
    end   = f"{years_to_include[0]}-12-31"
    bdays = pd.bdate_range(start=start, end=end, freq="B")
    bdays_str = bdays.strftime("%Y-%m-%d").tolist()

    # Assign to each RDP
    for rdp_id, rdp in network.rdps.items():
        rdp.calendar_dates = bdays_str

def build_rdp_calendar(date_str_list):
    # to DatetimeIndex (sorted, unique)
    idx = pd.DatetimeIndex(pd.to_datetime(sorted(set(date_str_list)), format='%Y-%m-%d'))
    horizon = len(idx)

    # Mappings for O(1) lookups
    date2t_str = {d.strftime('%Y-%m-%d'): i for i, d in enumerate(idx)}
    date2t_date = {d.date(): i for i, d in enumerate(idx)}  # if you use date objects
    t2date = np.array(idx.date)                              # fast array access
    dow = idx.weekday.values                                 # 0=Mon ... 6=Sun

    return {
        "index": idx,            # DatetimeIndex
        "horizon": horizon,      # int
        "date2t_str": date2t_str,
        "date2t_date": date2t_date,
        "t2date": t2date,
        "dow": dow
    }

# def next_t_on_weekday(t, target_weekday, idx):
#     """Find the next t' >= t where calendar weekday == target_weekday.
#        Returns None if not found."""
#     for i in range(t, len(idx)):
#         if idx[i].weekday() == target_weekday:
#             return i
#     return None