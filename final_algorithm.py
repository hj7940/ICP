# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 02:27:15 2025

Ostateczny algorytm
@author: Hanna Jaworska

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
    
"""

# import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


def combine_peaks_by_file(results, expected_peaks=("P1", "P2", "P3")):
    """
    Łączy słowniki z wynikami detekcji pików po 'file',
    aby każdy plik miał wszystkie piki w jednym słowniku.
    
    results: lista słowników (po jednym pikie)
    expected_peaks: tuple z oczekiwanymi pikami (domyślnie P1,P2,P3)
    
    Zwraca: lista słowników w formacie results_a
    """
    combined = {}
    
    for item in results:
        file_name = item["file"]
        method = item.get("method", None)
        class_id = item["class"]
        sig = item["signal"]
        
        if file_name not in combined:
            combined[file_name] = {
                "class": class_id,
                "file": file_name,
                "signal": sig,
                "method": method,
                "peaks_ref": {peak: np.nan for peak in expected_peaks},
                "peaks_detected": {peak: [] for peak in expected_peaks}
            }
        
        peak_name = item["peak"]
        if peak_name in expected_peaks:
            combined[file_name]["peaks_ref"][peak_name] = item["peaks_ref"].get(peak_name, np.nan)
            combined[file_name]["peaks_detected"][peak_name] = item["peaks_detected"].get(peak_name, [])
    
    # Zamiana słownika na listę słowników
    return list(combined.values())

def compute_peak_metrics_all_peaks(detection_results, class_name):
    """
    Łączy metryki wszystkich pików P1, P2, P3 dla każdego pliku/metody.
    Zwraca DataFrame z osobnymi błędami i liczby pików dla P1,P2,P3,
    Signals_with_all_Peaks i Num_Signals_in_Class.
    Obsługuje brakujące piki w klasach.
    """
    expected_peaks = ["P1", "P2", "P3"]

    # filtrowanie po klasie
    class_signals = [item for item in detection_results if item["class"] == class_name]

    # grupujemy po file i metodzie
    grouped = {}
    for item in class_signals:
        key = (item["file"], item["method"])
        if key not in grouped:
            grouped[key] = {}
        grouped[key][item["peak"]] = item

    metrics_list = []

    for (file_name, method_name), peaks_dict in grouped.items():
        row = {
            "Class": class_name,
            "File": file_name,
            "Method": method_name,
        }

        signals_all_detected = True  # czy wszystkie oczekiwane piki są wykryte w tym pliku

        for peak in expected_peaks:
            item = peaks_dict.get(peak)
            if item is None:
                # brak piku w danym pliku
                row.update({
                    f"Mean_X_Error_{peak}": np.nan,
                    f"Mean_Y_Error_{peak}": np.nan,
                    f"Mean_XY_Error_{peak}": np.nan,
                    f"Min_XY_Error_{peak}": np.nan,
                    f"Peak_Count_{peak}": 0
                })
                signals_all_detected = False
                continue

            ref_idx = item["peaks_ref"].get(peak)
            detected = item["peaks_detected"].get(peak, [])

            sig_df = item["signal"]
            t = sig_df.iloc[:, 0].values
            y = sig_df.iloc[:, 1].values

            if ref_idx is None or len(detected) == 0:
                dx = dy = dxy = np.array([np.nan])
                peak_count = 0
                signals_all_detected = False
            else:
                t_detected = t[detected]
                y_detected = y[detected]
                if np.isscalar(ref_idx):
                    dx = abs(t_detected - t[ref_idx])
                    dy = abs(y_detected - y[ref_idx])
                else:
                    dx = np.min([abs(t_detected - t[r]) for r in ref_idx], axis=0)
                    dy = np.min([abs(y_detected - y[r]) for r in ref_idx], axis=0)
                dxy = np.sqrt(dx**2 + dy**2)
                peak_count = len(detected)

            row.update({
                f"Mean_X_Error_{peak}": np.mean(dx),
                f"Mean_Y_Error_{peak}": np.mean(dy),
                f"Mean_XY_Error_{peak}": np.mean(dxy),
                f"Min_XY_Error_{peak}": np.min(dxy),
                f"Peak_Count_{peak}": peak_count
            })

        row["Signals_with_all_Peaks"] = int(signals_all_detected)
        metrics_list.append(row)

    df_metrics = pd.DataFrame(metrics_list)

    # Num_Signals_in_Class = liczba wszystkich plików w klasie
    df_metrics["Num_Signals_in_Class"] = len(class_signals)

    return df_metrics



def plot_files_in_class(detection_results, class_name):
    # Filtrujemy tylko pliki z danej klasy
    class_files = [item for item in detection_results if item["class"] == class_name]
    n_files = len(class_files)
    
    # Tworzymy siatkę subplots (np. max 3 kolumn)
    ncols = 5
    nrows = (n_files + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3*nrows), squeeze=False)
    axes = axes.flatten()
    
    # Kolory dla pików
    peak_colors = {'P1': 'red', 'P2': 'green', 'P3': 'blue'}
    peak_colors_d = {'P1': 'orange', 'P2': 'yellow', 'P3': 'cyan'}
    
    for idx, item in enumerate(class_files):
        ax = axes[idx]
        df = item["signal"]
        t = df.iloc[:, 0].values
        y = df.iloc[:, 1].values
        
        # Rysujemy sygnał
        ax.plot(t, y, color='black', lw=1)
        
        # Piki referencyjne
        for p, ref_idx in item['peaks_ref'].items():
            ax.scatter(t[ref_idx], y[ref_idx], color=peak_colors[p], marker='o', s=50, alpha=1.0, label=f"{p} ref" if idx==0 else "")
        
        # Piki wykryte
        for p, detected_idx in item['peaks_detected'].items():
            ax.scatter(t[detected_idx], y[detected_idx], color=peak_colors_d[p], marker='x', s=50, alpha=1.0, label=f"{p} detected" if idx==0 else "")
        
        ax.set_title(item['file'])
        ax.set_xlabel('Sample')
        ax.set_ylabel('ICP')
    
    # Usuń puste subplots
    for j in range(idx+1, len(axes)):
        fig.delaxes(axes[j])
    
    # Dodaj legendę tylko raz
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.tight_layout()
    plt.show()


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

results_combined = combine_peaks_by_file(results_a)

plot_files_in_class(results_combined, "Class1")
plot_files_in_class(results_combined, "Class2")
plot_files_in_class(results_combined, "Class3")
plot_files_in_class(results_combined, "Class2")

# all_metrics = []

# for class_name in ["Class1","Class2","Class3","Class4"]:
#         df_metrics = compute_peak_metrics_all_peaks(results_a, class_name)
#         # dodajemy info o metodzie
#         all_metrics.append(df_metrics) 
        
# # for class_name in ["Class4"]:
# #     for peak_name in ["P2"]:
# #         df_metrics = compute_peak_metrics_fixed(results_a, df_variant_a, peak_name, class_name)
# #         # dodajemy info o metodzie
# #         all_metrics.append(df_metrics) 

# # for class_name in ["Class1","Class2","Class3","Class4"]:
# #     df_metrics = compute_peak_metrics_all_peaks_combined(results_a, class_name)
# #     all_metrics.append(df_metrics)
# df_all_metrics = pd.concat(all_metrics, ignore_index=True)
# df_avg_metrics = df_all_metrics.groupby(["Class"]).mean(numeric_only=True).reset_index()
