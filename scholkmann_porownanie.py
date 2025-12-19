# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 22:05:02 2025

@author: User
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_handling import load_dataset, smooth_dataset
from methods import (concave, curvature, modified_scholkmann_old,
                     modified_scholkmann, 
                     line_distance, hilbert_envelope, wavelet)
from ranges import (ranges_full, ranges_pm3, ranges_whiskers, 
                    generate_ranges_for_all_files, compute_ranges_avg)
from main import peak_detection
from main import (all_methods, it2, it2_smooth_4Hz, it2_smooth_3Hz,
                  it1, it1_smooth_4Hz, it1_smooth_3Hz,
                  ranges_all_time, ranges_all_amps, compute_peak_metrics,
                  df_ranges_time, df_ranges_amps)

comparison_methods = {
    "modified_scholkmann_1-2_95": lambda sig: modified_scholkmann_old(sig, 2, 95),
    "modified_scholkmann_1-2_99": lambda sig: modified_scholkmann_old(sig, 2, 99),
    "modified_scholkmann0-5": lambda sig: modified_scholkmann(sig, 0.5),
    "modified_scholkmann1": lambda sig: modified_scholkmann(sig, 1)
}


dataset_to_test = it1
dataset_name = "it1"
range_type = "pm3"

# Pobranie odpowiednich zakresów czasowych i amplitudowych z DataFrame
time_r = df_ranges_time.loc[dataset_name, range_type]
amp_r = df_ranges_amps.loc[dataset_name, range_type]

comparison_results = []

print(f"Rozpoczynam porównanie metod Scholkmanna dla {dataset_name} ({range_type})...")

for m_name, m_func in comparison_methods.items():
    # Podmieniamy tymczasowo globalny słownik all_methods, aby peak_detection zadziałało
    all_methods[m_name] = m_func 
    
    # Wykrywanie pików
    det_res = peak_detection(
        dataset=dataset_to_test,
        method_name=m_name,
        time_ranges=time_r,
        amp_ranges=amp_r,
        ranges_name=range_type
    )
    
    # Obliczanie metryk dla każdej klasy i każdego piku
    for pk in ["P1", "P2", "P3"]:
        for cl in ["Class1", "Class2", "Class3", "Class4"]:
            metrics_df = compute_peak_metrics(det_res, pk, cl)
            metrics_df["Method_Variant"] = m_name
            comparison_results.append(metrics_df)

# 3. Agregacja wyników
df_comparison_full = pd.concat(comparison_results, ignore_index=True)

# 4. Wyświetlenie uśrednionych wyników dla porównania
df_comparison_avg = (
    df_comparison_full
    .groupby(["Method_Variant", "Class", "Peak"])
    .agg({
        "Mean_XY_Error": "mean",
        "Num_Signals_with_Peak": "first",
        "Num_Signals_in_Class": "first"
    })
    .reset_index()
)

# Obliczamy skuteczność (procent wykrytych sygnałów)
df_comparison_avg["Detection_Rate"] = (
    df_comparison_avg["Num_Signals_with_Peak"] / df_comparison_avg["Num_Signals_in_Class"]
)

# Zapis do CSV
df_comparison_avg.to_csv("scholkmann_versions_comparison_it1_pm3.csv", index=False)

print("Porównanie zakończone. Wyniki zapisano w 'scholkmann_versions_comparison_it1_pm3.csv'.")
print(df_comparison_avg.head(10))