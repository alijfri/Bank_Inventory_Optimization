import pandas as pd
import numpy as np
from inputs import start_date, finish_date
def unfit_deposit(file, network):
    """
    Read the data from the file, process unfit deposits per RDC,
    and return numpy arrays of daily unfit bag counts and values.
    """
    df = pd.read_csv(file)


    # Drop unused
    df.drop(columns=['denom'], inplace=True)
    df.dropna(inplace=True)
    # Convert date
    #df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce').dt.normalize()
    #df = df[df['date'].dt.year.isin(years_to_include)]
    df = df[(df["date"] >= start_date) & (df["date"] <= finish_date)]

    # Extract FI_id and REGION_ID
    df[['FI_id', 'REGION_ID']] = df['RDC'].str.split(' ', n=1, expand=True)
    df['FI_id'] = df['FI_id'].astype(int)
    df.drop(columns=['RDC'], inplace=True)

    # Set index
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    df['RefNom'] = df['RefNom'].astype(int)
    df['note_value'] = df['note_value'].astype(int)
    # Full date range
    full_index = pd.date_range(start=start_date, end=finish_date, freq='B')



    for rdp_id, rdp in network.rdps.items():
        for rdc in rdp.rdcs:
            df_sub = df[(df['REGION_ID'] == rdp_id) & (df['FI_id'] == rdc.id)]


            # if df_sub.empty:
            #     continue

            # Daily number of unfit bags (assuming each RefNom is 1 bag)
            bag_series = (
                df_sub.groupby(df_sub.index)['RefNom']
                .nunique()
                .reindex(full_index, fill_value=0)
            )

            # Daily unfit note value
            value_series = (
                df_sub.groupby(df_sub.index)['note_value']
                .sum()
                .reindex(full_index, fill_value=0)
            )

            # Store in RDC instance
            rdc.set_unfit(unfit_value=value_series.to_numpy(), unfit_bag=bag_series.to_numpy())
