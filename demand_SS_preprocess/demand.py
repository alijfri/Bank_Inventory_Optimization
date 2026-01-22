import pandas as pd
from inputs import demand_path_newnotes,start_date,finish_date
import numpy as np

def clean_data_new_notes(file_path):
    df = pd.read_excel(file_path)
    df = df[~(df == 'na').all(axis=1)]
    df['Denomination / Coupure'] = df['Denomination / Coupure'].str.replace('CAD', '', regex=False).str.strip()
    df['Denomination / Coupure'] = pd.to_numeric(df['Denomination / Coupure'])
    df.rename(columns={
        'Dispatch / Réparti': 'date',
        'Delivery / Livré': 'delivery_date',
        'Shipping RDC / CRD expéditeur': 'RDC',
        'Ref No / Numéro de référence': 'RefNom',
        'RDC / CRD': 'RDC',
        'Denomination / Coupure': 'Denom',
        'Quantity / Quantité': 'Note_count',
        'Value / Valeur': 'Note_value'
    }, inplace=True)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['delivery_date'] = pd.to_datetime(df['delivery_date'], errors='coerce')
    df[['FI_id', 'REGION_ID']] = df['RDC'].str.split(' ', n=1, expand=True)
    df['FI_id'] = df['FI_id'].astype(int)
    df.drop(columns=['RDC'], inplace=True)
    #df = df[df['date'].dt.year.isin(years_to_include)]
    df = df[(df["date"] >= start_date) & (df["date"] <= finish_date)]
    return df

# def replace_new_with_real_and_redirect_fit(df_all):
#     """
#     Replaces Note_type='New' rows in df_all with actual realized deliveries from df_new,
#     and redirects any missing fit quantity to corresponding 'FIT' rows in df_all.
    
#     Parameters:
#     - df_all: DataFrame containing planned withdrawals with 'Note_type' column ('New', 'FIT', etc.)
#     - df_new: DataFrame containing real delivery data with columns ['date', 'FI_id', 'Denom', 'Note_count', 'Note_value']
    
#     Returns:
#     - df_all_updated: Updated DataFrame with corrected New rows and augmented FIT rows
#     """
#     # reading the new notes  data and clean them: 
#     df_new=clean_data_new_notes(demand_path_newnotes)
#     # Step 0: Make copy of df_all and keep only New rows for merging

   
#     df_new['Demand_type'] = 'W'  # Assuming new notes are always deposits
#     # Step 1: Aggregate real new deliveries
#     df_new_grouped = df_new.groupby(['date', 'FI_id', 'REGION_ID', 'Denom','Demand_type'])[['Note_count', 'Note_value']].sum().reset_index()
#     df_new_grouped = df_new_grouped.rename(columns={
#         'Note_count': 'Real_New_Note_count',
#         'Note_value': 'Real_New_Note_value'
#     })


#     # Step 3: Merge actual deliveries into NEW rows
#     merged_new = pd.merge(
#         df_all,
#         df_new_grouped,
#         on=['date', 'FI_id', 'REGION_ID', 'Denom','Demand_type'],
#         how='left'
#     )
#     mask_unfit= merged_new['Note_type'].str.upper() == 'UNFIT'
#     merged_new['real_value_unfit'] = 0
#     merged_new.loc[mask_unfit, 'real_value_unfit'] =merged_new.loc[mask_unfit,'Note_value'] 

#     merged_new['Real_New_Note_count'] = merged_new['Real_New_Note_count'].fillna(0)
#     merged_new['Real_New_Note_value'] = merged_new['Real_New_Note_value'].fillna(0)
#     mask_new= merged_new['Note_type'].str.upper() == 'NEW'
#     ### adding two column for added fit notes for new notes:

#     merged_new.loc[mask_new, 'added_count_fit_needed'] =(merged_new.loc[mask_new,'Note_count'] - merged_new.loc[mask_new,'Real_New_Note_count']).clip(lower=0)

#     merged_new.loc[mask_new, 'added_value_fit_needed'] =(merged_new.loc[mask_new,'Note_value'] - merged_new.loc[mask_new,'Real_New_Note_value']).clip(lower=0)
#     merged_new.loc[mask_new, 'Note_value'] = np.where(
#         merged_new.loc[mask_new, 'Real_New_Note_value'] == 0,
#         0,
#         merged_new.loc[mask_new, 'Note_value']
#     )
#     merged_new.loc[mask_new, 'Note_count'] = np.where(
#         merged_new.loc[mask_new, 'Real_New_Note_count'] == 0,
#         0,
#         merged_new.loc[mask_new, 'Note_count']
#     )

#     mask_fit= merged_new['Note_type'].str.upper() == 'FIT'

#     #filling NA with 0 in added_count_fit_needed	added_value_fit_needed	real_count_fit_needed	real_value_fit_needed
#     merged_new['added_count_fit_needed'] = merged_new['added_count_fit_needed'].fillna(0).astype(int)
#     merged_new['added_value_fit_needed'] = merged_new['added_value_fit_needed'].fillna(0).astype(int)
#     df_added_fit_needed=merged_new[merged_new['added_count_fit_needed']!=0].copy()
#     df_added_fit_needed['Note_type'] = 'FIT'  
#     df_added_fit_needed = df_added_fit_needed[['date', 'FI_id', 'REGION_ID', 'Denom', 'Demand_type','Note_type', 'added_count_fit_needed', 'added_value_fit_needed']]
#     merged_with_added = pd.merge(
#         merged_new,
#         df_added_fit_needed,
#         on=['date', 'FI_id', 'REGION_ID', 'Denom', 'Demand_type','Note_type'],
#         how='left'
#     )
#     merged_with_added['added_count_fit_needed_y'] = merged_with_added['added_count_fit_needed_y'].fillna(0).astype(int)
#     merged_with_added['added_value_fit_needed_y'] = merged_with_added['added_value_fit_needed_y'].fillna(0).astype(int)
#     mask_fit= merged_with_added['Note_type'].str.upper() == 'FIT'
#     merged_with_added.loc[mask_fit, 'Note_count'] += merged_with_added.loc[mask_fit, 'added_count_fit_needed_y']
#     merged_with_added.loc[mask_fit, 'Note_value'] += merged_with_added.loc[mask_fit, 'added_value_fit_needed_y']
#     del merged_with_added['added_count_fit_needed_x']
#     del merged_with_added['added_value_fit_needed_x']
#     del merged_with_added['added_count_fit_needed_y']
#     del merged_with_added['added_value_fit_needed_y']
#     mask_new= merged_with_added['Note_type'].str.upper() == 'NEW'
#     mask_unfit= merged_with_added['Note_type'].str.upper() == 'UNFIT'
#     merged_with_added.loc[mask_new,'Note_count'] = merged_with_added.loc[mask_new,'Real_New_Note_count']
#     merged_with_added.loc[mask_new,'Note_value'] = merged_with_added.loc[mask_new,'Real_New_Note_value']
#     merged_with_added.loc[mask_unfit,'Note_value'] = merged_with_added.loc[mask_unfit,'real_value_unfit'].astype(np.int32)

#     del merged_with_added['Real_New_Note_count']
#     del merged_with_added['Real_New_Note_value']
#     del merged_with_added['real_value_unfit']
#     merged_with_added=merged_with_added[merged_with_added['Note_value']>0]
#     return merged_with_added


def replace_new_with_real_and_redirect_fit(df_all):
    """
    Replace Note_type='NEW' in df_all with actual realized deliveries from the Excel 'new notes' file.
    Any shortfall (planned - real, floored at zero) is redirected to FIT demand on the same key
    (date, FI_id, REGION_ID, Denom, Demand_type).

    Returns a DataFrame with:
      - NEW rows set to actual delivered amounts
      - Added FIT rows for the shortfall
      - UNFIT values preserved
      - Zero-value rows removed
    """
    # --- Load & clean the real NEW deliveries ---
    df_new = clean_data_new_notes(demand_path_newnotes)
    df_new['Demand_type'] = 'W'  # real new deliveries correspond to withdrawals
    real = (
        df_new.groupby(['date', 'FI_id', 'REGION_ID', 'Denom', 'Demand_type'], as_index=False)
              [['Note_count', 'Note_value']].sum()
              .rename(columns={
                  'Note_count': 'Real_New_Note_count',
                  'Note_value': 'Real_New_Note_value'
              })
    )

    # --- Normalize df_all case and types we rely on ---
    df_all = df_all.copy()
    df_all['Note_type'] = df_all['Note_type'].astype(str).str.upper()
    df_all['Demand_type'] = df_all['Demand_type'].astype(str).str.upper()

    # --- Preserve UNFIT (and any other note types) as-is for now ---
    others = df_all[df_all['Note_type'] != 'NEW'].copy()
    # Keep UNFIT value as-is (you already do this later; we’ll leave it untouched here)

    # --- Work only with NEW rows; group them to avoid duplication inflation ---
    new_planned = (
        df_all[df_all['Note_type'] == 'NEW']
        .groupby(['date', 'FI_id', 'REGION_ID', 'Denom', 'Demand_type'], as_index=False)
        [['Note_count', 'Note_value']].sum()
        .rename(columns={'Note_count': 'Planned_New_Note_count',
                         'Note_value': 'Planned_New_Note_value'})
    )

    # --- Join planned NEW with real NEW (left join: keep planned keys) ---
    new_join = new_planned.merge(
        real,
        on=['date', 'FI_id', 'REGION_ID', 'Denom', 'Demand_type'],
        how='left'
    )

    # Fill missing reals with 0 (no delivery that day/key)
    new_join['Real_New_Note_count'] = new_join['Real_New_Note_count'].fillna(0).astype(int)
    new_join['Real_New_Note_value'] = new_join['Real_New_Note_value'].fillna(0).astype(int)

    # --- Compute shortfall: planned - real (clipped at 0) ---
    new_join['Shortfall_count'] = (
        new_join['Planned_New_Note_count'] - new_join['Real_New_Note_count']
    ).clip(lower=0).astype(int)

    new_join['Shortfall_value'] = (
        new_join['Planned_New_Note_value'] - new_join['Real_New_Note_value']
    ).clip(lower=0).astype(int)

    # --- Build the adjusted NEW rows = real delivered amounts ---
    new_adjusted = new_join.copy()
    new_adjusted['Note_type'] = 'NEW'
    new_adjusted['Note_count'] = new_adjusted['Real_New_Note_count']
    new_adjusted['Note_value'] = new_adjusted['Real_New_Note_value']
    new_adjusted = new_adjusted[
        ['date', 'FI_id', 'REGION_ID', 'Denom', 'Demand_type', 'Note_type', 'Note_count', 'Note_value']
    ]

    # --- Build FIT top-up rows from the shortfall ---
    fit_topup = new_join.copy()
    fit_topup = fit_topup[fit_topup['Shortfall_value'] > 0]  # keep only positive shortfalls
    if not fit_topup.empty:
        fit_topup['Note_type'] = 'FIT'
        fit_topup['Note_count'] = fit_topup['Shortfall_count']
        fit_topup['Note_value'] = fit_topup['Shortfall_value']
        fit_topup = fit_topup[
            ['date', 'FI_id', 'REGION_ID', 'Denom', 'Demand_type', 'Note_type', 'Note_count', 'Note_value']
        ]
    else:
        # empty frame with same columns
        fit_topup = new_adjusted.iloc[0:0].copy()

    # --- Reassemble: others + adjusted NEW + FIT top-ups ---
    out = pd.concat([others, new_adjusted, fit_topup], ignore_index=True)

    # If df_all had multiple FIT rows on the same key elsewhere, coalesce to keep things tidy
    out = (
        out.groupby(['date', 'FI_id', 'REGION_ID', 'Denom', 'Demand_type', 'Note_type'], as_index=False)
           [['Note_count', 'Note_value']].sum()
    )

    # Remove zero-value rows (as you had)
    out = out[out['Note_value'] > 0].copy()

    return out



def reader(file):
    """
    Read the data from the file and return a pandas DataFrame.
    """
    df = pd.read_csv(file)
    df = df.fillna(0)
    df['WD_SUMRY_DAY_DT'] = pd.to_datetime(df['WD_SUMRY_DAY_DT'], format='%d-%b-%y')
    #df = df[df['WD_SUMRY_DAY_DT'].dt.year.isin(years_to_include)].sort_values(by='WD_SUMRY_DAY_DT')
    df = df[
    (df["WD_SUMRY_DAY_DT"] >= start_date) & 
    (df["WD_SUMRY_DAY_DT"] <= finish_date)
].sort_values(by="WD_SUMRY_DAY_DT")
    df.rename(columns={'WD_SUMRY_DAY_DT': 'date','NOTE_TYPE_CE':'Note_type',"BANK_NOTE_DENOM_AM":'Denom','FI_ORG_ID':'FI_id','BNDS_ACTIVITY_CE':'Demand_type',
                       'WD_NOTE_VALUE_AM':'Note_value','WD_NOTE_CT':'Note_count','BOISP.ORG_INFO_DIM_T.ORG_ENG_NM':'FI_name'}, inplace=True)
    #df['Denom'] = df['Denom'].str.replace('$', '', regex=False)
    df.set_index('date', inplace=True)
    df = df.drop([ 'SOURCE_CE',
              'NOTE_CLASS_TYPE_CE','ORDER_REQUEST_TYPE_CE'], axis=1)
    df['Note_type'] = df['Note_type'].replace('NEWREG', 'New')
    df['Note_type'] = df['Note_type'].str.upper()
    df['Demand_type'] = df['Demand_type'].replace('WTH', 'W')
    df['Demand_type'] = df['Demand_type'].replace('DEP', 'D')
    df['Denom'] = df['Denom'].astype(int)
    df['Note_value'] = df['Note_value'].astype(int)
    df['Note_count'] = df['Note_count'].astype(int)
    df['FI_id'] = df['FI_id'].astype(int)
    df['Note_type']=df['Note_type'].astype(str)
    df = df[df['Note_value'] != 0]
    df_final=replace_new_with_real_and_redirect_fit(df.reset_index())
    return df_final

def get_demand_array(df_sub, denom, note_type, demand_type):
    # Now df_sub is already filtered to the correct RDP and RDC
    filtered = df_sub[
        (df_sub['Denom'] == denom) &
        (df_sub['Note_type'] == note_type.upper()) &
        (df_sub['Demand_type'] == demand_type)
    ]

    full_index = pd.date_range(start=start_date, end=finish_date, freq='B')
    #full_index = pd.DatetimeIndex(sorted(df_sub['date'].unique()))
    daily_series = (
        filtered
        .groupby('date')['Note_count']
        .sum()
        .reindex(full_index, fill_value=0)
    )

    return daily_series.to_numpy()


def fill_all_demands(df, network):
    for rdp_id, rdp in network.rdps.items():
        for rdc in rdp.rdcs:
            # Filter only rows relevant to this (RDP, RDC) pair
            df_sub = df[(df['REGION_ID'] == rdp_id) & (df['FI_id'] == rdc.id)]

            # Skip if no demand data
            if df_sub.empty:
                continue

            # Get unique combinations
            unique_keys = df_sub[['Denom', 'Note_type', 'Demand_type']].drop_duplicates()

            for _, row in unique_keys.iterrows():
                denom = row['Denom']
                note_type = row['Note_type']
                note_type = note_type.upper()  # Ensure note_type is uppercase
                demand_type = row['Demand_type']

                demand_array = get_demand_array(
                    df_sub,
                    denom=denom,
                    note_type=note_type,
                    demand_type=demand_type
                )

                rdc.set_demand(denom, note_type, demand_array, demand_type)

        # After all RDCs are filled, compute aggregate demand
        rdp.compute_all_demands()
