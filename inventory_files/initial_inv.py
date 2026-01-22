import pandas as pd
from inputs import initial_inventory_path
import pandas as pd

def set_initial_inventory(network, date_initial):
    """
    Initialize RDP inventories in `network` using an initial inventory CSV.

    Logic:
    - Read the full initial inventory file.
    - Restrict to INVENT_TYPE_CE in {KNEWREG, KFIT, KNEWSERIES}.
    - Find the latest INVENT_SNAPSHOT_DT <= date_initial that has data.
    - Use that date to set initial inventory for each RDP.
    """
    # 1. Read CSV
    df = pd.read_csv(initial_inventory_path)

    # 2. Parse date column
    df['INVENT_SNAPSHOT_DT'] = pd.to_datetime(
        df['INVENT_SNAPSHOT_DT'],
        format='%d-%b-%y',
        errors='coerce'   # bad rows -> NaT
    )

    # Drop rows with invalid dates
    df = df.dropna(subset=['INVENT_SNAPSHOT_DT'])

    # 3. Keep only the types you care about
    df = df[df['INVENT_TYPE_CE'].isin(['KNEWREG', 'KFIT', 'KNEWSERIES'])].copy()

    # 4. Normalize dates to midnight for comparison
    df['DATE_NORM'] = df['INVENT_SNAPSHOT_DT'].dt.normalize()
    date_initial = pd.to_datetime(date_initial).normalize()

    # 5. Find the latest snapshot date <= date_initial
    candidate_dates = df.loc[df['DATE_NORM'] <= date_initial, 'DATE_NORM'].unique()

    if len(candidate_dates) == 0:
        raise ValueError(
            f"No inventory snapshots found on or before {date_initial.date()} "
            "for the specified INVENT_TYPE_CE values."
        )

    effective_date = max(candidate_dates)

    # Filter to that effective snapshot date
    df_sub = df[df['DATE_NORM'] == effective_date].copy()

    #print(f"[set_initial_inventory] Using snapshot date: {effective_date.date()} "
     #     f"(requested {date_initial.date()})")

    # 6. Map INVENT_TYPE_CE codes to NEW / FIT and clean
    df_sub['INVENT_TYPE_CE'] = df_sub['INVENT_TYPE_CE'].replace({
        'KNEWREG': 'NEW',
        'KFIT': 'FIT',
        'KNEWSERIES': 'NEW'
    })
    df_sub['INVENT_TYPE_CE'] = df_sub['INVENT_TYPE_CE'].str.strip().str.upper()

    # 7. Clean REGION_ID and types
    df_sub['REGION_ID'] = df_sub['REGION_ID'].astype(str).str.strip()

    # Denominations & counts as int (avoid 5.0 vs 5 issues)
    df_sub['BANK_NOTE_DENOM_AM'] = df_sub['BANK_NOTE_DENOM_AM'].astype(float).astype(int)
    df_sub['INVENT_NOTE_CT'] = df_sub['INVENT_NOTE_CT'].astype(float).astype(int)

    # 8. Group by RDP, denom, note type
    df_grouped = (
        df_sub
        .groupby(['REGION_ID', 'BANK_NOTE_DENOM_AM', 'INVENT_TYPE_CE'])['INVENT_NOTE_CT']
        .sum()
        .reset_index()
    )

    # 9. Push into each RDP object
    for _, row in df_grouped.iterrows():
        rdp_code = row['REGION_ID']
        denom = int(row['BANK_NOTE_DENOM_AM'])
        note_type = row['INVENT_TYPE_CE']   # 'NEW' or 'FIT'
        count = int(row['INVENT_NOTE_CT'])

        if rdp_code not in network.rdps:
           # print(f"[set_initial_inventory] WARNING: REGION_ID {rdp_code} not in network.rdps, skipping")
            continue

        rdp = network.rdps[rdp_code]
        rdp.set_initial_inventory(denom, note_type, count)




