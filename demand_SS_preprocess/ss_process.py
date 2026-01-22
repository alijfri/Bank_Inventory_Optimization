import pandas as pd
import numpy as np
#from inputs import small_s_percentages

def process_ss_data(file_path, sheet_name='month'):
    """
    Process the SS policy data from the specified Excel file and sheet.

    Parameters:
    - file_path: str, path to the Excel file

    Returns:
    - pd.DataFrame: Processed DataFrame with columns 'RDP', 'RDC', 'Denom', 'Note_type', and 'S'
    """
    df_ss = pd.read_excel(file_path, sheet_name=sheet_name)
    del df_ss['date']

    # Melt the DataFrame from wide to long format
    df_long = df_ss.melt(id_vars=['RDP', 'RDC'], var_name='Note', value_name='S')

    # Extract Denomination and Note_type from the 'Note' column
    df_long[['Denom', 'Note_type']] = df_long['Note'].str.extract(r'(\d+)\s*(fit|new)')

    # Convert Denom to integer for consistency
    df_long['Denom'] = df_long['Denom'].astype(int)
    df_long['Note_type'] = df_long['Note_type'].str.upper()
    
    del df_long['Note']
    
    return df_long

def fill_sS_policy(file_path, network):
    df= process_ss_data(file_path, sheet_name='month')
    for rdp_id, rdp in network.rdps.items():
        for denom in [5, 10, 20, 50, 100]:
            for note_type in ['FIT', 'NEW']:
                total_bigS = 0
                total_small_s = 0
                small_s_pct = small_s_percentages[note_type]

                for rdc in rdp.rdcs:
                    df_sub = df[
                        (df['RDP'] == rdp_id) &
                        (df['RDC'] == rdc.id) &
                        (df['Denom'] == denom) &
                        (df['Note_type'].str.upper() == note_type)
                    ]
                    
                    
                    if df_sub.empty:
                        bigS=0
                        small_s=0
                    else:

                        bigS = df_sub['S'].values[0]
                        small_s = int(bigS * small_s_pct)

                    # Set RDC-level values
                    rdc.set_bigS(denom, note_type, bigS)
                    rdc.set_s(denom, note_type, small_s)

                    # Accumulate for RDP
                    total_bigS += bigS
                    total_small_s += small_s

                # Set RDP-level values after summing over RDCs
                
                rdp.set_bigS(denom, note_type, total_bigS)
                rdp.set_s(denom, note_type, total_small_s)
                ### Now calcualting total fit and new for each denom
        # After both FIT and NEW for this denom, compute combined total at RDP
        bigS_fit = rdp.get_bigS(denom, 'FIT') or 0
        bigS_new = rdp.get_bigS(denom, 'NEW') or 0
        s_fit = rdp.get_s(denom, 'FIT') or 0
        s_new = rdp.get_s(denom, 'NEW') or 0
        #print(f"RDP {rdp.id} - Denom: {denom}, bigS FIT: {bigS_fit}, bigS NEW: {bigS_new}, s FIT: {s_fit}, s NEW: {s_new}")
        rdp.bigS_fit_and_new[denom] = bigS_fit + bigS_new
        rdp.s_fit_and_new[denom] = s_fit + s_new

        # Also compute the combined total at each RDC
        for rdc in rdp.rdcs:
            bigS_fit = rdc.get_bigS(denom, 'FIT') or 0
            bigS_new = rdc.get_bigS(denom, 'NEW') or 0
            s_fit = rdc.get_s(denom, 'FIT') or 0
            s_new = rdc.get_s(denom, 'NEW') or 0

            rdc.bigS_fit_and_new[denom] = bigS_fit + bigS_new
            rdc.s_fit_and_new[denom] = s_fit + s_new


