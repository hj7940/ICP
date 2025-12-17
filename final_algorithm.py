# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 02:27:15 2025

Ostateczny algorytm
@author: Hanna Jaworska
"""

# import os
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# from data_handling import load_dataset, smooth_dataset
# from methods import (concave, curvature, modified_scholkmann, 
#                      line_distance, hilbert_envelope, wavelet)
# from ranges import (ranges_full, ranges_pm3, ranges_whiskers, 
#                     generate_ranges_for_all_files, compute_ranges_avg)
from main import (all_methods, it2, it2_smooth_4Hz, it2_smooth_3Hz,
                  ranges_all_time, ranges_all_amps, compute_peak_metrics)
# do sprawdzenia
def peak_detection_single(dataset, method_name, class_id, peak, time_ranges=None, amp_ranges=None, ranges_name=None):
    """
    Wykrywa pojedynczy pik w zadanej klasie.
    
    dataset : lista słowników z sygnałami
    method_name : nazwa metody detekcji
    class_id : np. "Class1"
    peak : "P1", "P2" lub "P3"
    time_ranges : słownik z zakresami czasowymi (opcjonalnie)
    amp_ranges : słownik z zakresami amplitud (opcjonalnie)
    ranges_name : nazwa zestawu zakresów
    """
    
    if method_name not in all_methods:
        raise ValueError(f"Nieznana metoda: {method_name}")

    detect = all_methods[method_name]
    
    results = []

    for item in dataset:
        if item["class"] != class_id:
            continue  # tylko wskazana klasa
        
        file = item["file"]
        sig = item["signal"]
        y = sig.iloc[:, 1].values
        t = sig.iloc[:, 0].values

        raw_peaks = np.array(detect(y), dtype=int)

        # zakres czasowy
        if time_ranges is None:
            t_start, t_end = 0, 180
        else:
            tmp = time_ranges.get(class_id, {}).get(file, {}).get(peak, (np.nan, np.nan))
            t_start, t_end = tmp if tmp and len(tmp) == 2 else (np.nan, np.nan)
        
        # zakres amplitudy
        if amp_ranges is None:
            a_start, a_end = 0, 1
        else:
            tmp = amp_ranges.get(class_id, {}).get(file, {}).get(peak, (np.nan, np.nan))
            a_start, a_end = tmp if tmp and len(tmp) == 2 else (np.nan, np.nan)

        if np.isnan(t_start):
            detected_peaks = []
        else:
            mask = (
                (t[raw_peaks] >= t_start) & (t[raw_peaks] <= t_end) &
                (y[raw_peaks] >= a_start) & (y[raw_peaks] <= a_end)
            )
            detected_peaks = raw_peaks[mask].tolist()

        results.append({
            "method": method_name,
            "ranges": ranges_name,
            "class": class_id,
            "peak": peak, 
            "file": file,
            "signal": sig,
            "peaks_ref": {peak: item.get("peaks_ref", {}).get(peak, np.nan)},
            "peaks_detected": {peak: detected_peaks}
        })

    return results

def get_ranges(range_type, ranges_dataset, ranges_all_time, ranges_all_amps):
    """
    Zwraca (time_ranges, amp_ranges) zgodnie z kontraktem range_type
    """

    if range_type == "none":
        return None, None

    if range_type == "avg":
        return (
            ranges_all_time[ranges_dataset]["avg"],
            None
        )

    # full / pm3 / whiskers
    return (
        ranges_all_time[ranges_dataset][range_type],
        ranges_all_amps[ranges_dataset][range_type]
    )

def run_variant(
    settings,                 # "a" lub "b"
    datasets_dict,            # {"it2": it2, "it2_smooth_4Hz": ..., ...}
    ranges_all_time,
    ranges_all_amps
):
    """
    Uruchamia pełny algorytm dla wariantu a lub b
    Zwraca listę wyników peak_detection_single
    """

    results_all = []


    for _, row in settings.iterrows():

        dataset = datasets_dict[row.detect_dataset]

        time_ranges, amp_ranges = get_ranges(
            range_type=row.range_type,
            ranges_dataset=row.ranges_dataset,
            ranges_all_time=ranges_all_time,
            ranges_all_amps=ranges_all_amps
        )

        res = peak_detection_single(
            dataset=dataset,
            method_name=row.method,
            class_id=row["class"],
            peak=row.peak,
            time_ranges=time_ranges,
            amp_ranges=amp_ranges,
            ranges_name=row.range_type
        )

        results_all.extend(res)

    return results_all

def compute_peak_metrics_fixed(detection_results, settings, peak_name, class_name):
    metrics_list = []
    # bierzemy tylko sygnały dla danej klasy
    class_signals = [item for item in detection_results if (
        item["class"] == class_name 
        and item["peak"] == peak_name
        and item["method"] in settings["method"].values)]
    
    print(class_signals)
    
    for item in class_signals:
        file_name = item["file"]
        signal_df = item["signal"]
        method_name = item.get("method", None)
        t = signal_df.iloc[:, 0].values
        y = signal_df.iloc[:, 1].values

        ref_idx = item["peaks_ref"].get(peak_name)
        detected = item["peaks_detected"].get(peak_name, [])

        if ref_idx is None or len(detected) == 0:
            metrics_list.append({
                "Class": class_name,
                "Peak": peak_name,
                "File": file_name,
                "Method": method_name,
                "Mean_X_Error": np.nan,
                "Mean_Y_Error": np.nan,
                "Mean_XY_Error": np.nan,
                "Min_XY_Error": np.nan,
                "Peak_Count": 0,
                "Reference_Peaks": ref_idx,
                "Detected_Peaks": list(detected)
            })
            continue

        t_detected = t[detected]
        y_detected = y[detected]

        # jeśli wykryto wiele pików P1, bierzemy średnie błędy względem wszystkich referencji
        if np.isscalar(ref_idx):
            dx = abs(t_detected - t[ref_idx])
            dy = abs(y_detected - y[ref_idx])
        else:  # ref_idx może być listą wielu indeksów
            dx = np.min([abs(t_detected - t[r]) for r in ref_idx], axis=0)
            dy = np.min([abs(y_detected - y[r]) for r in ref_idx], axis=0)

        dxy = np.sqrt(dx**2 + dy**2)

        metrics_list.append({
            "Class": class_name,
            "Peak": peak_name,
            "File": file_name,
            "Method": method_name,
            "Mean_X_Error": np.mean(dx),
            "Mean_Y_Error": np.mean(dy),
            "Mean_XY_Error": np.mean(dxy),
            "Min_XY_Error": np.min(dxy),
            "Peak_Count": len(detected),
            "Reference_Peaks": ref_idx,
            "Detected_Peaks": list(detected)
        })

    df_metrics = pd.DataFrame(metrics_list)

    # liczba sygnałów w klasie liczy się teraz względem sygnałów, które **mogą mieć dany pik**
    num_signals_in_class = len(class_signals)
    num_detected = df_metrics["Peak_Count"].gt(0).sum()
    df_metrics["Num_Signals_in_Class"] = num_signals_in_class
    df_metrics["Num_Signals_with_Peak"] = num_detected

    return df_metrics



"""
Class1 P1:
    a) avg, concave (min blad)
    b) full, concave (max. signals w/ peaks)

Class1 P2:
    a) smooth 4Hz, avg, curvature (min blad)
    b) avg, curvature (max signals w/ peaks)
    
Class1 P3:
    a,b) smooth 4Hz, full, concave
    

Class2 P1:
    a) smooth 3Hz, avg, hilbert (min blad, 249)
    b) smooth 3Hz, avg, wavelet (podobny, 250) SAME_AVG!!!!!!!!!

Class2 P2:
    a) whiskers, wavelet (min blad)
    b) smooth 4Hz, full, curvature (max sig w/ peaks)

Class2 P3:
    a) avg, curvature (min blad)
    b) whiskers, hilbert (doslownie o 2 wiecej sig w/ peaks)

Class3 P1:
    a,b) smooth 4Hz, full, wavelet LUB smooth 4Hz, whiskers, wavelet

Class3 P2:
    a) whiskers, concave (min blad)
    b) smooth 4Hz, full, concave (max sig)

Class3 P3:
    a,b) none, modified scholkmann 1/2 99 LUB none, modified scholkmann 1 99

Class4 P2:
    a,b) smooth 4Hz, none, modified scholkmann 1/2 99 LUB smooth 4Hz, none, modified scholkmann 1 99
"""


datasets = [
    (it2, "it2"),
    (it2_smooth_4Hz, "it2_smooth_4Hz"),
    (it2_smooth_3Hz, "it2_smooth_3Hz"),
]

# %% ranges
#base_dataset = it2

# ranges_all_time = {}
# for dataset, dataset_name in datasets:
#     ranges_all_time[dataset_name] = {}

#     for range_type, range_name in (
#         (ranges_full, "full"),
#         (ranges_pm3, "pm3"),
#         (ranges_whiskers, "whiskers"),
#     ):
#         (time, amp) = generate_ranges_for_all_files(dataset, range_type)
#         ranges_all_time[dataset_name][range_name] = time
#     # avg liczymy ZAWSZE z bazowego datasetu
#     ranges_all_time[dataset_name]["avg"] = compute_ranges_avg(dataset)
# df_ranges_time = pd.DataFrame.from_dict(ranges_all_time, orient="index")

# # only_two_avg_ranges = {}
# # for dataset, dataset_name in datasets:
# #     only_two_avg_ranges[dataset_name] = {}
# #     only_two_avg_ranges[dataset_name]["avg"] = compute_ranges_avg(base_dataset)
# # df_only_two_avg_ranges = pd.DataFrame.from_dict(only_two_avg_ranges, orient="index")    

# ranges_all_amps = {}
# for dataset, dataset_name in datasets:
#     ranges_all_amps[dataset_name] = {}

#     for range_type, range_name in (
#         (ranges_full, "full"),
#         (ranges_pm3, "pm3"),
#         (ranges_whiskers, "whiskers"),
#     ):
#         (time, amp) = generate_ranges_for_all_files(dataset, range_type)
#         ranges_all_amps[dataset_name][range_name] = amp
            
# df_ranges_amps = pd.DataFrame.from_dict(ranges_all_amps, orient="index")

# %% minimalizajca bledu

#c1_p1=peak_detection_single(it2, "concave", "Class1", "P1", df_ranges_time.loc["it2", "avg"], None, "avg")

# --- wariant a ---
df_variant_a = pd.DataFrame([
    # -------- Class1 --------
    ("Class1", "P1", "it2",            "it2",            "avg",  "concave"),
    ("Class1", "P2", "it2_smooth_4Hz", "it2_smooth_4Hz", "avg",  "curvature"),
    ("Class1", "P3", "it2_smooth_4Hz", "it2_smooth_4Hz", "full", "concave"),

    # -------- Class2 --------
    ("Class2", "P1", "it2_smooth_3Hz", "it2_smooth_3Hz", "avg",  "hilbert"),
    ("Class2", "P2", "it2",            "it2",            "whiskers", "wavelet"),
    ("Class2", "P3", "it2",            "it2",            "avg",      "curvature"),

    # -------- Class3 --------
    ("Class3", "P1", "it2_smooth_4Hz", "it2_smooth_4Hz", "full", "wavelet"),
    ("Class3", "P2", "it2",            "it2",            "whiskers", "concave"),
    ("Class3", "P3", "it2",            "it2",            "none", "modified_scholkmann_1-2_99"),

    # -------- Class4 --------
    ("Class4", "P2", "it2_smooth_4Hz", "it2_smooth_4Hz", "none", "modified_scholkmann_1-2_99"),
],
columns=[
    "class",
    "peak",
    "detect_dataset",
    "ranges_dataset",
    "range_type",
    "method"
])

# --- wariant b ---
df_variant_b = pd.DataFrame([
    # -------- Class1 --------
    ("Class1", "P1", "it2",            "it2",            "full", "concave"),
    ("Class1", "P2", "it2",            "it2",            "avg",  "curvature"),
    ("Class1", "P3", "it2_smooth_4Hz", "it2_smooth_4Hz", "full", "concave"),

    # -------- Class2 --------
    ("Class2", "P1", "it2_smooth_3Hz", "it2",            "avg",  "wavelet"),
    ("Class2", "P2", "it2_smooth_4Hz", "it2_smooth_4Hz", "full", "curvature"),
    ("Class2", "P3", "it2",            "it2",            "whiskers", "hilbert"),

    # -------- Class3 --------
    ("Class3", "P1", "it2_smooth_4Hz", "it2_smooth_4Hz", "full", "wavelet"),
    ("Class3", "P2", "it2_smooth_4Hz", "it2_smooth_4Hz", "full",     "concave"),
    ("Class3", "P3", "it2",            "it2",            "none", "modified_scholkmann_1-2_99"),

    # -------- Class4 --------
    ("Class4", "P2", "it2_smooth_4Hz", "it2_smooth_4Hz", "none", "modified_scholkmann_1-2_99"),
],
columns=[
    "class",
    "peak",
    "detect_dataset",
    "ranges_dataset",
    "range_type",
    "method"
])

# df_a = df_algo[df_algo["variant"] == "a"]

datasets_dict = {
    "it2": it2,
    "it2_smooth_4Hz": it2_smooth_4Hz,
    "it2_smooth_3Hz": it2_smooth_3Hz,
}

results_a = run_variant(
    df_variant_a,
    datasets_dict=datasets_dict,
    ranges_all_time=ranges_all_time,
    ranges_all_amps=ranges_all_amps
)


all_metrics = []

for class_name in ["Class1","Class2","Class3"]:
    for peak_name in ["P1","P2","P3"]:
        df_metrics = compute_peak_metrics_fixed(results_a, df_variant_a, peak_name, class_name)
        # dodajemy info o metodzie
        all_metrics.append(df_metrics) 
        
for class_name in ["Class4"]:
    for peak_name in ["P2"]:
        df_metrics = compute_peak_metrics_fixed(results_a, df_variant_a, peak_name, class_name)
        # dodajemy info o metodzie
        all_metrics.append(df_metrics) 
        
df_all_metrics = pd.concat(all_metrics, ignore_index=True)
df_avg_metrics = df_all_metrics.groupby(["Class", "Peak", "Method"]).mean(numeric_only=True).reset_index()
