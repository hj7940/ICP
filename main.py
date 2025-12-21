# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 22:51:03 2025

@author: Hanna Jaworska
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_handling import load_dataset, smooth_dataset
from methods import (concave, curvature, modified_scholkmann, modified_scholkmann_old, 
                     line_distance, hilbert_envelope, wavelet)
from ranges import (ranges_full, ranges_pm3, ranges_whiskers, 
                    generate_ranges_for_all_files, compute_ranges_avg)
from all_plots import plot_all_signals_with_peaks_final, plot_concave_signals, plot_all_signals_with_peaks_by_peak_type


# sprawdzone i dzialajace: import zakresow z ranges, load_dataset i smooth_dataset
# generowanie zakresow zgodnych w strukturze z crossingami
# peak detection
# tworzenie dfow z zakesami

# zgrubna - po calym dataset zeby nie mnozyc petli
def peak_detection(dataset, method_name, time_ranges=None, amp_ranges=None, 
                   ranges_name=None, peaks=("P1", "P2", "P3"), tuned_params=None):
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
    if method_name not in all_methods and method_name != "concave_tuned":
        raise ValueError(f"Nieznana metoda: {method_name}")

    # detect = all_methods[method_name]
    
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
        
        if method_name == "concave_tuned":
            raw_peaks_all = {}
            for p in peaks:
                params = get_peak_params(class_id, p, tuned_params)
                raw_peaks_all[p] = np.array(concave(y, 0, 8, **params), dtype=int)
        else:
            detect = all_methods[method_name]
            raw_peaks = np.array(detect(y), dtype=int)
        
        # raw_peaks = np.array(detect(y), dtype=int)
        
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
            
            raw = raw_peaks_all[p] if method_name == "concave_tuned" else raw_peaks

            mask = (
                (t[raw] >= t_start) & (t[raw] <= t_end) &
                (y[raw] >= a_start) & (y[raw] <= a_end)
            )
            
            detected[p] = raw[mask].tolist()

            # # filtruj tylko te w zadanym zakresie czasu i amplitudy
            # mask = (
            #         (t[raw_peaks] >= t_start) & (t[raw_peaks] <= t_end) &
            #         (y[raw_peaks] >= a_start) & (y[raw_peaks] <= a_end)
            #     )
    
            # detected[p] = raw_peaks[mask].tolist()
            
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


def process_all_datasets(datasets, df_ranges_time, df_ranges_amps, 
                         tuned_params=None,  peaks=("P1","P2","P3")):
    
    all_results = {}
    #all_metrics_list = []
    
    assert all(p in ["P1","P2","P3"] for p in peaks), \
    f"BŁĄD: peaks={peaks}"
    
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
            print(f"==={dataset_name}_{range_name if range_name is not None else 'none'}===")
            # time_ranges i amp_ranges wyciągamy z DataFrame dla danego datasetu i kolumny
            time_range = (
                df_ranges_time.loc[dataset_name, range_name] 
                if range_name != "none" else None)
            
            # amp_ranges idzie w parze, wyjątek dla "avg" i "none"
            if range_name in ["avg", "none"]:
                amp_range = None
            else:
                amp_range = df_ranges_amps.loc[dataset_name, range_name]
            
            config_key = f"{dataset_name}_{range_name}"
            all_results[config_key] = {}
            
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
                    df_all_metrics = pd.read_csv(all_metrics_file)
                    df_avg_metrics = pd.read_csv(avg_metrics_file)
                    
                    all_results[config_key][method_name] = (
                        df_all_metrics, df_avg_metrics)
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
                all_metrics = []
                for peak_name in peaks:
                    for class_name in ["Class1","Class2","Class3","Class4"]:
                        df_metrics = compute_peak_metrics(detection_results, peak_name, class_name)
                        # dodajemy info o metodzie
                        df_metrics["Method"] = method_name
                        all_metrics.append(df_metrics) 
                        
                df_all_metrics = pd.concat(all_metrics, ignore_index=True)
                df_avg_metrics = df_all_metrics.groupby(["Class", "Peak", "Method"]).mean(numeric_only=True).reset_index()
                
                # --- zapis do CSV ---
                df_all_metrics.to_csv(all_metrics_file, index=False)
                df_avg_metrics.to_csv(avg_metrics_file, index=False)
                    
                print(f"Zapisano wyniki: {method_name} ({folder_wyniki})")
            
                all_results[config_key][method_name] = (
                    df_all_metrics, df_avg_metrics)
                
            # ===============================
            # concave_tuned 
            # ===============================
            if tuned_params is not None:
                method_name = "concave_tuned"

                all_metrics_file = os.path.join(
                    folder_wyniki, f"{method_name}_all_metrics.csv"
                )
                avg_metrics_file = os.path.join(
                    folder_wyniki, f"{method_name}_avg_metrics.csv"
                )
                
                if os.path.exists(all_metrics_file) and os.path.exists(avg_metrics_file):
                    df_all_metrics = pd.read_csv(all_metrics_file)
                    df_avg_metrics = pd.read_csv(avg_metrics_file)
                    all_results[config_key][method_name] = (
                        df_all_metrics, df_avg_metrics)
                    print(f"Wczytano zapisane wyniki: {method_name} ({folder_wyniki})")
                    continue
                
                detection_results = peak_detection(
                    dataset=dataset,
                    method_name="concave_tuned",
                    time_ranges=time_range,
                    amp_ranges=amp_range,
                    ranges_name=range_name,
                    tuned_params=tuned_params
                )

                all_metrics = []
                for peak_name in peaks:
                    for class_name in ["Class1", "Class2", "Class3", "Class4"]:
                        df = compute_peak_metrics(
                            detection_results, peak_name, class_name
                        )
                        df["Method"] = method_name
                        all_metrics.append(df)

                df_all_metrics = pd.concat(all_metrics, ignore_index=True)
                df_avg_metrics = (
                    df_all_metrics
                    .groupby(["Class", "Peak", "Method"])
                    .mean(numeric_only=True)
                    .reset_index()
                )

                df_all_metrics.to_csv(all_metrics_file, index=False)
                df_avg_metrics.to_csv(avg_metrics_file, index=False)
                print(f"Zapisano wyniki: {method_name} ({folder_wyniki})")
            
                all_results[config_key][method_name] = (
                    df_all_metrics, df_avg_metrics
                )
        
    return all_results


def top10_configs(df, peak_name, class_name, metric="Mean_XY_Error"):
    """
    Zwraca top 5 konfiguracji dla danego piku i klasy z uśrednionych wyników df_avg
    """
    df_filtered = df[(df["Peak"] == peak_name) & (df["Class"] == class_name)]
    
    # Sortujemy po metryce rosnąco (najmniejszy błąd najlepiej)
    top10 = df_filtered.nsmallest(10, metric)
    return top10

def merge_identical_configs_before_top(df):
    metric_cols = [c for c in df.columns if c not in ["Config", "Method"]]

    return (
        df
        .groupby(metric_cols, dropna=False, as_index=False)
        .agg({
            "Config": lambda x: ",".join(sorted(set(x)))
        })
    )

# -------------------
# funkcja pomocnicza do wyznaczenia parametrów z tuned_params
# -------------------
def get_peak_params(class_id, peak_name, tuned_params):
    row = tuned_params[
        (tuned_params["Class"] == class_id) & 
        (tuned_params["Peak"] == peak_name)]
    if row.empty:
        return {}
    
    prom_lower = row["prom_lower"].values[0]
    prom_upper = row["prom_upper"].values[0]

    # prominencja nie może być < 0
    prom_lower = max(0.0, prom_lower)
    return {
        "height": [row["h_lower"].values[0], row["h_upper"].values[0]],
        "prominence": [prom_lower, prom_upper]
    }

def compare_concave_methods(dataset, tuned_params):
    # results_default = peak_detection(dataset, "concave", 
    #                                  df_ranges_time.loc["it1", "full"], 
    #                                  df_ranges_amps.loc["it1", "full"], 
    #                                  ranges_name="full")
    
    # results_tuned = []
    # for item in dataset:
    #     class_id = item["class"]
    #     file = item["file"]
    #     sig = item["signal"]
    #     y = sig.iloc[:, 1].values

    #     detected = {}
    #     for p in ["P1", "P2", "P3"]:
    #         params = get_peak_params(class_id, p, tuned_params)
    #         detected[p] = concave(y, **params).tolist()

    #     results_tuned.append({
    #         "method": "concave_tuned",
    #         "class": class_id,
    #         "file": file,
    #         "signal": sig,
    #         "peaks_ref": item.get("peaks_ref", {}),
    #         "peaks_detected": detected
    #     })
    results_default = peak_detection(
    dataset=it1,
    method_name="concave",
    time_ranges=df_ranges_time.loc["it1", "full"],
    amp_ranges=df_ranges_amps.loc["it1", "full"],
    ranges_name="full"
    )
    
    # concave tuned + full
    results_tuned = peak_detection(
        dataset=it1,
        method_name="concave_tuned",
        time_ranges=df_ranges_time.loc["it1", "full"],
        amp_ranges=df_ranges_amps.loc["it1", "full"],
        ranges_name="full",
        tuned_params=tuned_params
    )
    
    return results_default, results_tuned


# %%  lista metod 
all_methods = {
    "concave": lambda sig: concave(sig, 0, d2x_threshold=0, min_len=8, height=0, prominence=0),
    "concave_d2x=-0-002": lambda sig:  concave(sig, -0.002, d2x_threshold=0, min_len=8, height=0, prominence=0),
    "concave_d2x=0-002": lambda sig:  concave(sig, 0.002, 0, d2x_threshold=0, min_len=8, height=0, prominence=0),
    
    # "modified_scholkmann_1_99": lambda sig: modified_scholkmann_old(sig, 1, 99),
    # "modified_scholkmann_1_95": lambda sig: modified_scholkmann_old(sig, 1, 95),
    # "modified_scholkmann_1-2_95": lambda sig: modified_scholkmann_old(sig, 2, 95),
    # "modified_scholkmann_1-2_99": lambda sig: modified_scholkmann_old(sig, 2, 99),
  
    "modified_scholkmann0-5": lambda sig: modified_scholkmann(sig, 0.5),
    "modified_scholkmann1": lambda sig: modified_scholkmann(sig, 1),
    
    "curvature": lambda sig: curvature(sig, 0, 8),
    "line_distance_8": lambda sig: line_distance(sig, 0,"vertical", 8),
    "line_perpendicular_8": lambda sig: line_distance(sig, 0, "perpendicular", 8),
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
it2_smooth_3Hz = smooth_dataset(it2, cutoff=3, inplace=False)

# %% zakresy (przechowywane w pandas DataFrames - osobno time i amps)
datasets = [
    (it1, "it1"),
    (it1_smooth_4Hz, "it1_smooth_4Hz"),
    (it1_smooth_3Hz, "it1_smooth_3Hz"),
    (it2, "it2"),
    (it2_smooth_4Hz, "it2_smooth_4Hz"),
    (it2_smooth_3Hz, "it2_smooth_3Hz"),
]

# %% ujednolicenie struktury zakresow dla time i amps oraz policzenie 
# zakresow na podstawie sredniej wolnej i szybkiej
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
    # avg liczymy ZAWSZE z bazowego datasetu
    if dataset_name.startswith("it1"):
        base_dataset = it1
    elif dataset_name.startswith("it2"):
        base_dataset = it2
    else:
        raise ValueError(f"Nieznany dataset: {dataset_name}")
    ranges_all_time[dataset_name]["avg"] = compute_ranges_avg(base_dataset)
    # ranges_all_time[dataset_name]["avg"] = compute_ranges_avg(dataset)
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

tuned_params = pd.read_csv("tuned_params.csv")

# %% liczenie pikow
if __name__ == "__main__":
    
    wyniki_base = "wyniki_nowe_same_new_filter"
    os.makedirs(wyniki_base, exist_ok=True)
    
    
    # results = process_all_datasets(datasets, df_ranges_time, df_ranges_amps, tuned_params=tuned_params)
    
    # det = peak_detection(
    # dataset=it1,
    # method_name="concave_tuned",
    # time_ranges=df_ranges_time.loc["it1", "full"],
    # amp_ranges=df_ranges_amps.loc["it1", "full"],
    # ranges_name="full",
    # tuned_params=tuned_params
    # )
    
    # plot_all_signals_with_peaks_by_peak_type(
    #     detection_results=det,
    #     method_name="concave_tuned",
    #     ranges_name="full"
    # )
    
    det = peak_detection(
    dataset=it1,
    method_name="concave_tuned",
    time_ranges=None,
    amp_ranges=None,
    ranges_name="none",
    tuned_params=tuned_params
    )
    
    plot_all_signals_with_peaks_by_peak_type(
        detection_results=det,
        method_name="concave_tuned",
        ranges_name="none"
    )
    # plot_concave_signals(it1)
    
    # it2_results = {k: v for k, v in results.items() if k.startswith("it2")}
    
    # dfs_avg = []
    # for config_name, method_dict in it2_results.items():
    #     for method_name, (df_all, df_avg) in method_dict.items():
    #         df = df_avg.copy()
    #         df["Config"] = f"{config_name}_{method_name}"
    #         dfs_avg.append(df)
    # df_it2_avg = pd.concat(dfs_avg, ignore_index=True)
    
    
    # peaks = ["P1","P2","P3"]
    # classes = ["Class1","Class2","Class3","Class4"]
    
    # top_xy_dfs = {}
    # top_minxy_dfs = {}
    
    # min_fraction = 0.85
    # # max_XY_Error = 30
    
    # # Tworzenie osobnych DF-ów
    # for class_id in classes:
    #     for pk in peaks:
    #         # --- filtr po udziale sygnałów ---
    #         df_filtered = df_it2_avg[
    #             (df_it2_avg["Peak"] == pk) &
    #             (df_it2_avg["Class"] == class_id) &
    #             (df_it2_avg["Num_Signals_with_Peak"] / df_it2_avg["Num_Signals_in_Class"] >= min_fraction)
    #             #(df_it2_avg["Mean_XY_Error"] <= max_XY_Error) &
    #             #(df_it2_avg["Peak_Count"] <= 10)
    #         ].copy()
            
    #         error_cols = ["Mean_XY_Error", "Min_XY_Error"]
            
    #         df_filtered[error_cols] = df_filtered[error_cols].round(10)
    #         #print(df_filtered.columns.tolist())
    #         df_merged = merge_identical_configs_before_top(df_filtered)
            
    #         # --- Mean_XY_Error ---
    #         df_top_xy = top10_configs(df_merged, pk, class_id, metric="Mean_XY_Error")
    #         top_xy_dfs[f"{class_id}_{pk}"] = df_top_xy
    
    #         # --- Min_XY_Error ---
    #         df_top_minxy = top10_configs(df_merged, pk, class_id, metric="Min_XY_Error")
    #         top_minxy_dfs[f"{class_id}_{pk}"] = df_top_minxy
    
    # cols_to_keep = [c for c in df_it2_avg.columns if c not in ["Method", "Mean_X_Error", "Mean_Y_Error"]]
    
    # df_top_xy_all = pd.concat(top_xy_dfs.values(), ignore_index=True)
    # df_top_xy_all[cols_to_keep].to_csv("top_xy_it2_85_same_avg.csv", sep=' ', index=False)
    
    # df_top_minxy_all = pd.concat(top_minxy_dfs.values(), ignore_index=True)
    # df_top_minxy_all[cols_to_keep].to_csv("top_min_xy_it2_85_same_avg.csv", sep=' ', index=False)
    
    
# -------- WYNIKI - kombinacje parametrow IT1 -----------------
    # it1_results = {k: v for k, v in results.items() if k.startswith("it1")}
    
    # dfs_avg = []
    # for config_name, method_dict in it1_results.items():
    #     for method_name, (df_all, df_avg) in method_dict.items():
    #         df = df_avg.copy()
    #         df["Config"] = f"{config_name}_{method_name}"
    #         dfs_avg.append(df)
    # df_it1_avg = pd.concat(dfs_avg, ignore_index=True)
    
    
    # peaks = ["P1","P2","P3"]
    # classes = ["Class1","Class2","Class3","Class4"]
    
    # top_xy_dfs = {}
    # top_minxy_dfs = {}
    
    # min_fraction = 0.90
    # # max_XY_Error = 30
    
    # # Tworzenie osobnych DF-ów
    # for class_id in classes:
    #     for pk in peaks:
    #         # --- filtr po udziale sygnałów ---
    #         df_filtered = df_it1_avg[
    #             (df_it1_avg["Peak"] == pk) &
    #             (df_it1_avg["Class"] == class_id) &
    #             (df_it1_avg["Num_Signals_with_Peak"] / df_it1_avg["Num_Signals_in_Class"] >= min_fraction)
    #             #(df_it1_avg["Mean_XY_Error"] <= max_XY_Error) &
    #             #(df_it1_avg["Peak_Count"] <= 10)
    #         ].copy()
            
    #         error_cols = ["Mean_XY_Error", "Min_XY_Error"]
            
    #         df_filtered[error_cols] = df_filtered[error_cols].round(10)
    #         #print(df_filtered.columns.tolist())
    #         df_merged = merge_identical_configs_before_top(df_filtered)
            
    #         # --- Mean_XY_Error ---
    #         df_top_xy = top10_configs(df_merged, pk, class_id, metric="Mean_XY_Error")
    #         top_xy_dfs[f"{class_id}_{pk}"] = df_top_xy
    
    #         # --- Min_XY_Error ---
    #         df_top_minxy = top10_configs(df_merged, pk, class_id, metric="Min_XY_Error")
    #         top_minxy_dfs[f"{class_id}_{pk}"] = df_top_minxy
    
    # cols_to_keep = [c for c in df_it1_avg.columns if c not in ["Method", "Mean_X_Error", "Mean_Y_Error"]]
    
    # df_top_xy_all = pd.concat(top_xy_dfs.values(), ignore_index=True)
    # df_top_xy_all[cols_to_keep].to_csv("top_xy_it1_90_nowe_same_new_filter.csv", sep=' ', index=False)
    
    # df_top_minxy_all = pd.concat(top_minxy_dfs.values(), ignore_index=True)
    # df_top_minxy_all[cols_to_keep].to_csv("top_min_xy_it1_90_same_avg.csv", sep=' ', index=False)
    
# ---------------- test ---------------------- 
    # test_avg = peak_detection(
    #     dataset=it1,
    #     method_name="concave",
    #     time_ranges=df_ranges_time.loc["it1", "avg"],
    #     amp_ranges=None,
    #     ranges_name="avg"
    # )
    
    # test_avg_results = compute_peak_metrics(test_avg, "P1", "Class1")
    
    # test_whiskers = peak_detection(
    #     dataset=it1,
    #     method_name="concave",
    #     time_ranges=df_ranges_time.loc["it1", "whiskers"],
    #     amp_ranges=df_ranges_amps.loc["it1", "whiskers"],
    #     ranges_name="whiskers"
    # )
    
# ------------------- scholkman - porowanainie z nowa wersja -------------
    # comparison_methods = {
    #     "modified_scholkmann_1-2_95": lambda sig: modified_scholkmann_old(sig, 2, 95),
    #     "modified_scholkmann_1-2_99": lambda sig: modified_scholkmann_old(sig, 2, 99),
    #     "modified_scholkmann0-5": lambda sig: modified_scholkmann(sig, 0.5),
    #     "modified_scholkmann1": lambda sig: modified_scholkmann(sig, 1)
    # }


    # dataset_to_test = it1
    # dataset_name = "it1"
    # range_type = "pm3"

    # # Pobranie odpowiednich zakresów czasowych i amplitudowych z DataFrame
    # time_r = df_ranges_time.loc[dataset_name, range_type]
    # amp_r = df_ranges_amps.loc[dataset_name, range_type]

    # comparison_results = []

    # print(f"Rozpoczynam porównanie metod Scholkmanna dla {dataset_name} ({range_type})...")

    # for m_name, m_func in comparison_methods.items():
    #     # Podmieniamy tymczasowo globalny słownik all_methods, aby peak_detection zadziałało
    #     all_methods[m_name] = m_func 
        
    #     # Wykrywanie pików
    #     det_res = peak_detection(
    #         dataset=dataset_to_test,
    #         method_name=m_name,
    #         time_ranges=time_r,
    #         amp_ranges=amp_r,
    #         ranges_name=range_type
    #     )
        
    #     # Obliczanie metryk dla każdej klasy i każdego piku
    #     for pk in ["P1", "P2", "P3"]:
    #         for cl in ["Class1", "Class2", "Class3", "Class4"]:
    #             metrics_df = compute_peak_metrics(det_res, pk, cl)
    #             metrics_df["Method_Variant"] = m_name
    #             comparison_results.append(metrics_df)

    # # 3. Agregacja wyników
    # df_comparison_full = pd.concat(comparison_results, ignore_index=True)

    # # 4. Wyświetlenie uśrednionych wyników dla porównania
    # df_comparison_avg = (
    #     df_comparison_full
    #     .groupby(["Method_Variant", "Class", "Peak"])
    #     .agg({
    #         "Mean_XY_Error": "mean",
    #         "Num_Signals_with_Peak": "first",
    #         "Num_Signals_in_Class": "first"
    #     })
    #     .reset_index()
    # )

    # # Obliczamy skuteczność (procent wykrytych sygnałów)
    # df_comparison_avg["Detection_Rate"] = (
    #     df_comparison_avg["Num_Signals_with_Peak"] / df_comparison_avg["Num_Signals_in_Class"]
    # )

    # # Zapis do CSV
    # df_comparison_avg.to_csv("scholkmann_versions_comparison_it1_pm3.csv", index=False)

    # print("Porównanie zakończone. Wyniki zapisano w 'scholkmann_versions_comparison_it1_pm3.csv'.")
    # print(df_comparison_avg.head(10))    
    
# --------------- concave - porownanie dla roznego prominence, peak width
    
    
    # concave_comparison = {"concave": concave}
    # results_default, results_tuned = compare_concave_methods(dataset=it1, tuned_params=tuned_params)
    
    # # -------------------
    # # obliczanie metryk
    # # -------------------
    # metrics_default = pd.concat([
    #     compute_peak_metrics(results_default, p, c)
    #     for c in ["Class1", "Class2", "Class3", "Class4"]
    #     for p in ["P1", "P2", "P3"]
    # ])
    # metrics_default["Method"] = "concave_default"
    
    # metrics_tuned = pd.concat([
    #     compute_peak_metrics(results_tuned, p, c)
    #     for c in ["Class1", "Class2", "Class3", "Class4"]
    #     for p in ["P1", "P2", "P3"]
    # ])
    # metrics_tuned["Method"] = "concave_tuned"
    # metrics_all = pd.concat([metrics_default, metrics_tuned], ignore_index=True)
    
    # metrics_avg = (
    #     metrics_all
    #     .groupby(["Method", "Class", "Peak"])
    #     .agg({
    #         "Mean_X_Error": "mean",
    #         "Mean_Y_Error": "mean",
    #         "Mean_XY_Error": "mean",
    #         "Min_XY_Error": "mean",
    #         "Peak_Count": "mean",
    #         "Num_Signals_in_Class": "first",
    #         "Num_Signals_with_Peak": "mean"
    #     })
    #     .reset_index()
    #     )
    
    # metrics_avg["Detection_Rate"] = metrics_avg["Num_Signals_with_Peak"] / metrics_avg["Num_Signals_in_Class"]

    # # Zapis do CSV
    # metrics_avg.to_csv("concave_comparison_avg_it1_2.csv", index=False)
    
    # print("Uśrednione wyniki zapisane w 'concave_comparison_avg_it1.csv'")
    # print(metrics_avg.head(10))

# ---------- koniec ----------------
    print("jejj jupii")
    