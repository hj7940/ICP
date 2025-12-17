# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 01:03:55 2025

@author: User
"""

import pandas as pd

# =============================
# 1. Wczytanie danych
# =============================

df_per_dataset = pd.read_csv("top_xy_it2_85.csv", sep=" ")
df_global_avg  = pd.read_csv("top_xy_it2_85_same_avg.csv", sep=" ")

# Ujednolicenie nazw kolumn (na wypadek separatorów , / .)
numeric_cols = [
    "Mean_XY_Error",
    "Peak_Count",
    "Num_Signals_with_Peak"
]

for df in (df_per_dataset, df_global_avg):
    for c in numeric_cols:
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .astype(float)
            )

# =============================
# 2. Agregacja po Class + Peak
# =============================

agg_cols = {
    "Mean_XY_Error": "mean",
    "Peak_Count": "mean",
    "Num_Signals_with_Peak": "mean",
}

g_per_dataset = (
    df_per_dataset
    .groupby(["Class", "Peak"], as_index=False)
    .agg(agg_cols)
    .rename(columns={
        "Mean_XY_Error": "Mean_XY_Error_per_dataset",
        "Peak_Count": "Peak_Count_per_dataset",
        "Num_Signals_with_Peak": "Num_Signals_with_Peak_per_dataset",
    })
)

g_global = (
    df_global_avg
    .groupby(["Class", "Peak"], as_index=False)
    .agg(agg_cols)
    .rename(columns={
        "Mean_XY_Error": "Mean_XY_Error_global",
        "Peak_Count": "Peak_Count_global",
        "Num_Signals_with_Peak": "Num_Signals_with_Peak_global",
    })
)

# =============================
# 3. Merge + różnice
# =============================

df_cmp = g_per_dataset.merge(
    g_global,
    on=["Class", "Peak"],
    how="inner"
)

df_cmp["Δ_Mean_XY_Error"] = (
    df_cmp["Mean_XY_Error_per_dataset"]
    - df_cmp["Mean_XY_Error_global"]
)

df_cmp["Δ_Peak_Count"] = (
    df_cmp["Peak_Count_per_dataset"]
    - df_cmp["Peak_Count_global"]
)

df_cmp["Δ_Num_Signals_with_Peak"] = (
    df_cmp["Num_Signals_with_Peak_per_dataset"]
    - df_cmp["Num_Signals_with_Peak_global"]
)

# =============================
# 4. Wypisanie interpretacji
# =============================

# for _, row in df_cmp.iterrows():
#     cls = row["Class"]
#     peak = row["Peak"]

#     dx = row["Δ_Mean_XY_Error"]
#     dp = row["Δ_Peak_Count"]
#     dn = row["Δ_Num_Signals_with_Peak"]

#     print(f"\nDLA {cls} {peak}:")

#     if dx < 0:
#         print(f"- Mean_XY_Error mniejszy o średnio {abs(dx):.4f}")
#     else:
#         print(f"- Mean_XY_Error większy o średnio {dx:.4f}")

#     if dp > 0:
#         print(f"- Peak_Count większy o {dp:.2f}")
#     else:
#         print(f"- Peak_Count mniejszy o {abs(dp):.2f}")

#     if dn > 0:
#         print(f"- Num_Signals_with_Peak większy o {dn:.2f}")
#     else:
#         print(f"- Num_Signals_with_Peak mniejszy o {abs(dn):.2f}")

for _, row in df_cmp.iterrows():
    class_id = row["Class"]
    peak = row["Peak"]

    mean_xy_per = row["Mean_XY_Error_per_dataset"]
    mean_xy_glob = row["Mean_XY_Error_global"]
    peak_count_per = row["Peak_Count_per_dataset"]
    peak_count_glob = row["Peak_Count_global"]
    num_sig_per = row["Num_Signals_with_Peak_per_dataset"]
    num_sig_glob = row["Num_Signals_with_Peak_global"]

    print(f"\nDLA {class_id} {peak}:")

    # Mean_XY_Error
    if mean_xy_per < mean_xy_glob:
        print(f"- Mean_XY_Error: {mean_xy_per:.4f} (per_dataset) < {mean_xy_glob:.4f} (global_avg), mniejszy o {mean_xy_glob - mean_xy_per:.4f}")
    else:
        print(f"- Mean_XY_Error: {mean_xy_per:.4f} (per_dataset) > {mean_xy_glob:.4f} (global_avg), większy o {mean_xy_per - mean_xy_glob:.4f}")

    # # Peak_Count
    # if peak_count_per > peak_count_glob:
    #     print(f"- Peak_Count: {peak_count_per:.2f} (per_dataset) > {peak_count_glob:.2f} (global_avg), większy o {peak_count_per - peak_count_glob:.2f}")
    # else:
    #     print(f"- Peak_Count: {peak_count_per:.2f} (per_dataset) < {peak_count_glob:.2f} (global_avg), mniejszy o {peak_count_glob - peak_count_per:.2f}")

    # Num_Signals_with_Peak
    if num_sig_per > num_sig_glob:
        print(f"- Num_Signals_with_Peak: {num_sig_per:.2f} (per_dataset) > {num_sig_glob:.2f} (global_avg), większy o {num_sig_per - num_sig_glob:.2f}")
    else:
        print(f"- Num_Signals_with_Peak: {num_sig_per:.2f} (per_dataset) < {num_sig_glob:.2f} (global_avg), mniejszy o {num_sig_glob - num_sig_per:.2f}")
