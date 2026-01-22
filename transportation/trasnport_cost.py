import re

def get_cost_by_value(value_notes, row, round_trip=False):
    """
    Determine appropriate column based on the value and apply pricing rules:
    - One-way: if on boundary, pick lower range
    - Two-way: if on boundary, pick upper range
    """
    row= row.iloc[0]  # Assuming row is a DataFrame with one row
    value_notes = float(value_notes)
    if value_notes==0: return 0 
    selected_col = None

    for col in row.index:
        match = re.match(r"\$(\d+)-\$(\d+)[MK]?", col.replace(',', ''))
        if match:
            lower, upper = map(int, match.groups())
            lower *= 1_000_000
            upper *= 1_000_000

            # Boundary value
            if value_notes == lower:
                if round_trip:
                    continue  # skip lower range → use higher
                else:
                    selected_col = col
                    break

            elif lower < value_notes < upper:
                selected_col = col
                break

    if selected_col is None:
        raise ValueError(f"Value {value_notes} does not match any defined interval.")

    # Parse and return cost
    val_str = str(row[selected_col]).replace(',', '').strip()
    return float(val_str)

