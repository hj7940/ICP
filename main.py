# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 22:51:03 2025

@author: Hanna Jaworska
"""
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, hilbert, find_peaks_cwt
from data_handling import load_dataset, smooth_dataset
from methods import (concave, curvature, modified_scholkmann, 
                     line_distance, hilbert_envelope, wavelet)
from ranges import (ranges_full, ranges_pm3, ranges_whiskers, 
                    generate_ranges_for_all_files, compute_crossings)


# sprawdzone i dzialajace: import zakresow z ranges, load_dataset i smooth_dataset
# generowanie zakresow zgodnych w strukturze z crossingami
# peak detection
# tworzenie dfow z zakesami

# zgrubna - po calym dataset zeby nie mnozyc petli
def peak_detection(dataset, method_name, time_ranges=None, amp_ranges=None, 
                   ranges_name=None, peaks=("P1", "P2", "P3")):
    """
    Wykrywa piki w określonym przedziale czasowym/amplitudowym
    
    dataset : list
        Lista zawierajaca elementy:       
            
        - class : str (np. "Class1")
        
        - file : str (np. "Class1_example_0001")
        
        - signal: DataFrame (kolumny: Sample_no, ICP)
        
        - peaks_ref : dict (np. {'P1': 31, 'P2': 53, 'P3': 90})
    
        dataset = [
        {'class': 'Class1', 'file': 'Class1_example_0001', 'signal': DataFrame, 'peaks_ref': {'P1': 31, 'P2': 53, 'P3': 90}},
        {'class': 'Class1', 'file': 'Class1_example_0002', 'signal': DataFrame, 'peaks_ref': {'P1': 31, 'P2': 58, 'P3': 97}},
         ... i tak dalej
        ]
    method name - funkcja wykrywania
    ranges = (time_ranges, amps_ranges), ale mozwe byc (None, None) lub (time_ranges, None)
    """
    if method_name not in all_methods:
        raise ValueError(f"Nieznana metoda: {method_name}")

    detect = all_methods[method_name]
    
    # if ranges is not None:
    #     time_ranges, amp_ranges = ranges
    # else:
    #     time_ranges = None 
    #     amp_ranges = None
        
    results = []
    
    for item in dataset:
        class_id  = item["class"]
        file = item["file"]
        
        sig = item["signal"]
        y = sig.iloc[:, 1].values
        t = sig.iloc[:, 0].values
        
        raw_peaks = np.array(detect(y), dtype=int)
        
        detected = {}
        
        for p in peaks:
        
            # --- obsługa zakresów czasowych i amplitud ---
            if time_ranges is None:
                t_start, t_end = 0, 180
            else:
                tmp = time_ranges.get(class_id, {}).get(file, {}).get(p, (np.nan, np.nan))
                if not tmp or len(tmp) != 2:
                    t_start, t_end = np.nan, np.nan
                else:
                    t_start, t_end = tmp
        
            if amp_ranges is None:
                a_start, a_end = 0, 1
            else:
                a_start, a_end = amp_ranges.get(class_id, {}).get(file, {}).get(p, (np.nan, np.nan))
            
            # brak piku (Class4 P1/P3)
            if np.isnan(t_start):
                detected[p] = []
                continue
    

            # filtruj tylko te w zadanym zakresie czasu i amplitudy
            mask = (
                    (t[raw_peaks] >= t_start) & (t[raw_peaks] <= t_end) &
                    (y[raw_peaks] >= a_start) & (y[raw_peaks] <= a_end)
                )
    
            detected[p] = raw_peaks[mask].tolist()
            
        results.append({
            "method": method_name,
            "ranges": ranges_name,
            "class": class_id,
            "file": file,
            "signal": sig,
            "peaks_ref": item.get("peaks_ref", {}),
            "peaks_detected": detected
        })

    return results

def compute_ranges_avg(dataset):
    r_c1_c3 = compute_crossings(dataset, min_distance=8)
    r_c2_c4 = compute_crossings(dataset, min_distance=12)

    merged = {}
    for class_id in ("Class1", "Class3"):
        merged[class_id] = r_c1_c3[class_id]
    for class_id in ("Class2", "Class4"):
        merged[class_id] = r_c2_c4[class_id]

    return merged

# DO SPRAWDZENIA!!!!!!!! CZAT
def compute_peak_metrics(detection_results, peak_name, class_name):
    metrics_list = []
    class_signals = [item for item in detection_results if item["class"] == class_name]

    for item in class_signals:
        file_name = item["file"]
        signal_df = item["signal"]
        t = signal_df.iloc[:, 0].values
        y = signal_df.iloc[:, 1].values

        ref_idx = item["peaks_ref"].get(peak_name)
        detected = item["peaks_detected"].get(peak_name, [])

        if ref_idx is None or len(detected) == 0:
            metrics_list.append({
                "Class": class_name,
                "Peak": peak_name,
                "File": file_name,
                "Mean_X_Error": np.nan,
                "Mean_Y_Error": np.nan,
                "Mean_XY_Error": np.nan,
                "Min_XY_Error": np.nan,
                "Peak_Count": len(detected),
                "Reference_Peaks": ref_idx,
                "Detected_Peaks": list(detected)
            })
            continue

        t_detected = t[detected]
        y_detected = y[detected]

        dx = abs(t_detected - t[ref_idx])
        dy = abs(y_detected - y[ref_idx])
        dxy = np.sqrt(dx**2 + dy**2)

        metrics_list.append({
            "Class": class_name,
            "Peak": peak_name,
            "File": file_name,
            "Mean_X_Error": np.mean(dx),
            "Mean_Y_Error": np.mean(dy),
            "Mean_XY_Error": np.mean(dxy),
            "Min_XY_Error": np.min(dxy),
            "Peak_Count": len(detected),
            "Reference_Peaks": ref_idx,
            "Detected_Peaks": list(detected)
        })

    df_metrics = pd.DataFrame(metrics_list)
    num_signals_in_class = len(class_signals)
    num_detected = df_metrics["Peak_Count"].gt(0).sum()
    df_metrics["Num_Signals_in_Class"] = num_signals_in_class
    df_metrics["Num_Signals_with_Peak"] = num_detected

    return df_metrics


# Z CZATU!!!!!!! DO SPRAWDZENIA
def process_all_datasets(datasets, df_ranges_time, df_ranges_amps, 
                         peaks=("P1","P2","P3")):
    for dataset, dataset_name in datasets:
        # lista zakresów do przetwarzania
        # range_keys = list(ranges_time.get(dataset_name, {}).keys()) + ["none"]  # "none" na koniec
        
        # for range_name in range_keys:
        #     time_range = ranges_time[dataset_name].get(range_name) if range_name != "none" else None
            
        #     # amp_ranges idzie w parze z time_ranges, wyjątek dla avg
        #     if range_name == "avg":
        #         amp_range = None
        #     elif range_name == "none":
        #         amp_range = None
        #     else:
        #         amp_range = ranges_amp[dataset_name].get(range_name)
        
        range_keys = list(df_ranges_time.columns) + ["none"]
        
        for range_name in range_keys:
            # time_ranges i amp_ranges wyciągamy z DataFrame dla danego datasetu i kolumny
            time_range = (
                df_ranges_time.loc[dataset_name, range_name] 
                if range_name != "none" else None)
            
            # amp_ranges idzie w parze, wyjątek dla "avg" i "none"
            if range_name in ["avg", "none"]:
                amp_range = None
            else:
                amp_range = df_ranges_amps.loc[dataset_name, range_name]
            
            # Tworzymy folder
            folder_name = (
                f"{dataset_name}_{range_name if range_name is not None else 'none'}")  # ZMIANA: nazwa podfolderu
            folder_wyniki = os.path.join(wyniki_base, folder_name)  # ZMIANA: używamy wyniki_base jako nadrzędnego folderu
            os.makedirs(folder_wyniki, exist_ok=True)  # ZMIANA: tworzymy podfolder
            
            for method_name in all_methods.keys():
                all_metrics_file = (
                    os.path.join(
                        folder_wyniki, f"{method_name}_all_metrics.csv"))
                avg_metrics_file = (
                    os.path.join(
                        folder_wyniki, f"{method_name}_avg_metrics.csv"))
                
                if os.path.exists(all_metrics_file) and os.path.exists(avg_metrics_file):
                    print(f"Wczytano zapisane wyniki: {method_name} ({folder_wyniki})")
                    continue
                
                # --- detekcja pików ---
                detection_results = peak_detection(
                    dataset=dataset,
                    method_name=method_name,
                    time_ranges=time_range,
                    amp_ranges=amp_range,
                    ranges_name=range_name
                )
                
                # --- liczenie metryk ---
                all_metrics_list = []
                for peak_name in peaks:
                    for class_name in ["Class1","Class2","Class3","Class4"]:
                        df_metrics = compute_peak_metrics(detection_results, peak_name, class_name)
                        # dodajemy info o metodzie
                        df_metrics["Method"] = method_name
                        all_metrics_list.append(df_metrics)
                
                df_all_metrics = pd.concat(all_metrics_list, ignore_index=True)
                df_avg_metrics = df_all_metrics.groupby(["Class", "Peak", "Method"]).mean(numeric_only=True).reset_index()
                
                # --- zapis do CSV ---
                df_all_metrics.to_csv(all_metrics_file, index=False)
                df_avg_metrics.to_csv(avg_metrics_file, index=False)
                
                print(f"Zapisano wyniki: {method_name} ({folder_wyniki})")

# %%  lista metod 
all_methods = {
    "concave": lambda sig: concave(sig, 0, 0, None),
    "concave_d2x=-0-002": lambda sig:  concave(sig, -0.002, 0, None),
    "concave_d2x=0-002": lambda sig:  concave(sig, 0.002, 0, None),
    
    "modified_scholkmann_1_99": lambda sig: modified_scholkmann(sig, 1, 99),
    "modified_scholkmann_1_95": lambda sig: modified_scholkmann(sig, 1, 95),
    "modified_scholkmann_1-2_95": lambda sig: modified_scholkmann(sig, 2, 95),
    "modified_scholkmann_1-2_99": lambda sig: modified_scholkmann(sig, 2, 99),
    
    "curvature": lambda sig: curvature(sig, 0, 0, None),
    "line_distance_10": lambda sig: line_distance(sig, 0,"vertical", 10),
    "hilbert": lambda sig: hilbert_envelope(sig, 0),
    "wavelet": lambda sig: wavelet(sig, (1,10))
}

# %% ladowanie zestawow danych
base_path = r"ICP_pulses_it1"
it1 = load_dataset(base_path, "it1")
it1_smooth_4Hz = smooth_dataset(it1, cutoff=4, inplace=False)
it1_smooth_3Hz = smooth_dataset(it1, cutoff=3, inplace=False)

base_path_2 = r"ICP_pulses_it2"
it2 = load_dataset(base_path_2, "it2")
it2_smooth_4Hz = smooth_dataset(it2, cutoff=4, inplace=False)
it2_smooth_3Hz = smooth_dataset(it1, cutoff=3, inplace=False)

# %% zakresy (przechowywane w pandas DataFrames - osobno time i amps)
datasets = [
    (it1, "it1"),
    (it1_smooth_4Hz, "it1_smooth_4Hz"),
    (it1_smooth_3Hz, "it1_smooth_3Hz"),
    (it2, "it2"),
    (it2_smooth_4Hz, "it2_smooth_4Hz"),
    (it2_smooth_3Hz, "it2_smooth_3Hz"),
]

ranges_all_time = {}

for dataset, dataset_name in datasets:
    ranges_all_time[dataset_name] = {}

    for range_type, range_name in (
        (ranges_full, "full"),
        (ranges_pm3, "pm3"),
        (ranges_whiskers, "whiskers"),
    ):
        (time, amp) = generate_ranges_for_all_files(dataset, range_type)
        ranges_all_time[dataset_name][range_name] = time
            
    ranges_all_time[dataset_name]["avg"] = compute_ranges_avg(dataset)
df_ranges_time = pd.DataFrame.from_dict(ranges_all_time, orient="index")

ranges_all_amps = {}

for dataset, dataset_name in datasets:
    ranges_all_amps[dataset_name] = {}

    for range_type, range_name in (
        (ranges_full, "full"),
        (ranges_pm3, "pm3"),
        (ranges_whiskers, "whiskers"),
    ):
        (time, amp) = generate_ranges_for_all_files(dataset, range_type)
        ranges_all_amps[dataset_name][range_name] = amp
            
df_ranges_amps = pd.DataFrame.from_dict(ranges_all_amps, orient="index")

wyniki_base = "wyniki_main"
os.makedirs(wyniki_base, exist_ok=True)


process_all_datasets(datasets, df_ranges_time, df_ranges_amps)

# %% 

# test_avg = peak_detection(
#     dataset=it1,
#     method_name="concave",
#     time_ranges=df_ranges_time.loc["it1", "avg"],
#     amp_ranges=None,
#     ranges_name="avg"
# )

# test_whiskers = peak_detection(
#     dataset=it1,
#     method_name="concave",
#     time_ranges=df_ranges_time.loc["it1", "whiskers"],
#     amp_ranges=df_ranges_amps.loc["it1", "whiskers"],
#     ranges_name="whiskers"
# )

    
    
print("jejj jupii")
    