import os
import pandas as pd

def save_simulation_logs(event_logs, base_dir="logs"):
    for name, records in event_logs.items():
        df = pd.DataFrame(records)

        if not df.empty and "rdp" in df.columns:
            # Save separate CSVs grouped by rdp
            for rdp_id, group in df.groupby("rdp"):
                folder = os.path.join(base_dir, rdp_id)
                os.makedirs(folder, exist_ok=True)
                
                filepath = os.path.join(folder, f"log_{name}.csv")
                group.to_csv(filepath, index=False)
        else:
            # If no rdp column, save in general folder
            folder = os.path.join(base_dir, "general")
            os.makedirs(folder, exist_ok=True)
            df.to_csv(os.path.join(folder, f"log_{name}.csv"), index=False)