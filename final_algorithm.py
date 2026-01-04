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

import copy
import pandas as pd
import numpy as np
# from data_handling import load_dataset, smooth_dataset
# from methods import (concave, curvature, modified_scholkmann, 
#                      line_distance, hilbert_envelope, wavelet)
# from ranges import (ranges_full, ranges_pm3, ranges_whiskers, 
#                     generate_ranges_for_all_files, compute_ranges_avg)
import math
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from main import (all_methods, it2, it2_smooth_4Hz, it2_smooth_3Hz,
                  it1, it1_smooth_4Hz, it1_smooth_3Hz,
                  ranges_all_time, ranges_all_amps)
from all_plots import (plot_files_in_class, plot_upset_classic_postproc_new, 
                       plot_signal_pre_post, plot_signal_with_concave_areas, 
                       plot_peak_detection_pie_new, plot_selected_files)

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
        sig_raw=item["signal_raw"]
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
            "signal_raw": sig_raw,
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
        sig_raw=item["signal_raw"]
        
        if file_name not in combined:
            combined[file_name] = {
                "class": class_id,
                "file": file_name,
                "signal": sig,
                "signal_raw": sig_raw,
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



# def postprocess_peaks_new(results_combined):
#     """
#     Postprocess peaks_detected zgodnie z regułami DALSZE DZIAŁANIA.
#     """

#     results_post=copy.deepcopy(results_combined)

#     for item in results_post:
#         class_id = item["class"]
#         peaks = item["peaks_detected"]

#         # pomocnicze
#         def mean_peak(p):
#             return [int(round(np.mean(p)))] if len(p) > 0 else []

#         def first_peak(p):
#             return [int(p[0])] if len(p) > 0 else []

#         def last_peak(p):
#             return [int(p[-1])] if len(p) > 0 else []
        
#         def middle_peak(p):
#             """
#             Środkowy element listy.
#             Dla parzystej liczby → wcześniejszy z dwóch środkowych.
#             """
#             if len(p) == 0:
#                 return []
#             idx = (len(p) - 1) // 2
#             return [int(p[idx])]
        
#         def second_peak(p):
#             """
#             Drugi element listy (drugi w czasie).
#             Jeśli <2 elementy → [].
#             """
#             if len(p) >= 2:
#                 return [int(p[1])]
#             return p
                
#         def _finalize_peaks(peaks):
#             """
#             []  -> np.nan
#             [x] -> x
#             """
#             out = {}
#             for k, v in peaks.items():
#                 if len(v) == 0:
#                     out[k] = np.nan
#                 else:
#                     out[k] = int(v[0])
#             return out

#         # =========================
#         # -------- Class1 ---------
#         # =========================
#         if class_id == "Class1":

            
#             # P3: jeśli dwa → zostaw późniejszy
#             # ZMIANA! drugi
#             peaks["P3"] = second_peak(peaks["P3"])

#             # P1: jeśli kilka → średnia
#             peaks["P1"] = first_peak(peaks["P1"])

#             # P2: jesli kilka - sredni (brak info)
#             peaks["P2"] = first_peak(peaks["P2"])

#             # jeśli P2 i P3 zbyt blisko → usuń oba
#             if peaks["P2"] and peaks["P3"]:
#                 if abs(peaks["P3"][0] - peaks["P2"][0]) <= 5:
#                     peaks["P2"] = []
#                     peaks["P3"] = []
#                     # peaks["P3"] = peaks["P3"][1:] # zmiana
            
#             # --- NOWY WARUNEK (Class1): P3 pomiędzy dwoma P2 ---
#             if len(peaks["P2"]) == 2 and len(peaks["P3"]) == 1:
#                 x, y = min(peaks["P2"]), max(peaks["P2"])
#                 z = peaks["P3"][0]
#                 if x < z < y:
#                     peaks["P2"] = []
            
#             # warunek P1 < P2 < P3
#             if peaks["P1"] and peaks["P2"] and peaks["P3"]:
#                 if not (peaks["P1"][0] < peaks["P2"][0] < peaks["P3"][0]):
#                     # jeśli P3 jest najpóźniejszy → zostaw tylko P3
#                     if peaks["P3"][0] > max(peaks["P1"][0], peaks["P2"][0]):
#                         peaks["P1"] = []
#                         peaks["P2"] = []
#                     else:
#                         peaks["P1"] = []
#                         peaks["P2"] = []
#                         peaks["P3"] = []

#         # =========================
#         # -------- Class2 ---------
#         # =========================
#         elif class_id == "Class2":


#             # # jeśli P2 i P3 zbyt blisko → usuń P3
#             # if peaks["P2"] and peaks["P3"]:
#             #     p2_val = peaks["P2"][0]
#             #     peaks["P3"] = [p for p in peaks["P3"] if abs(p - p2_val) > 5]
            
#             # P3: jeśli 2 → drugi, jeśli >2 → środkowy
#             # if len(peaks["P3"]) == 2:
#             #     peaks["P3"] = [peaks["P3"][1]]
#             # else:
#             #     peaks["P3"] = middle_peak(peaks["P3"])
            
#             peaks["P3"] = second_peak(peaks["P3"])
            
#             # if peaks["P3"]:
#             #     peaks["P2"] = [p for p in peaks["P2"] if p <= peaks["P3"][0]]
#             # # P2: ostatni, a jeśli dokładnie 3 → środkowy
#             # if len(peaks["P2"]) == 3:
#             #     peaks["P2"] = middle_peak(peaks["P2"])
#             # else:
#             #     peaks["P2"] = last_peak(peaks["P2"])
            
#             peaks["P2"] = first_peak(peaks["P2"])
            
#             # P1: pierwszy
#             peaks["P1"] = first_peak(peaks["P1"])
            
#             # jeśli P2 i P3 zbyt blisko → usuń oba
#             if peaks["P2"] and peaks["P3"]:
#                 if abs(peaks["P3"][0] - peaks["P2"][0]) <= 5:
#                     peaks["P2"] = []
#                     peaks["P3"] = []
#                     # peaks["P3"] = peaks["P3"][1:] # zmiana
                    
#             # warunek P1 < P2 < P3
#             if peaks["P1"] and peaks["P2"] and peaks["P3"]:
#                 if not (peaks["P1"][0] < peaks["P2"][0] < peaks["P3"][0]):
#                     if peaks["P3"][0] > max(peaks["P1"][0], peaks["P2"][0]):
#                         peaks["P1"] = []
#                         peaks["P2"] = []
#                     else:
#                         peaks["P1"] = []
#                         peaks["P2"] = []
#                         peaks["P3"] = []

#         # =========================
#         # -------- Class3 ---------
#         # =========================
#         # elif class_id == "Class3":

            
#             # P3: średnia ZMIANA drugi
#             # peaks["P3"] = last_peak(peaks["P3"])
#             peaks["P3"] = second_peak(peaks["P3"])

#             # P1: first -> zmiana na last
#             # peaks["P1"] = first_peak(peaks["P1"])
#             peaks["P1"] = last_peak(peaks["P1"])

#             # P2: pierwszy
#             peaks["P2"] = first_peak(peaks["P2"])

#             # # jeśli P2 i P3 zbyt blisko → usuń P3
#             # if peaks["P2"] and peaks["P3"]:
#             #     if abs(peaks["P3"][0] - peaks["P2"][0]) <= 5:
#             #         peaks["P3"] = []
#             # jeśli P2 i P3 zbyt blisko → usuń oba
#             if peaks["P2"] and peaks["P3"]:
#                 if abs(peaks["P3"][0] - peaks["P2"][0]) <= 5:
#                     peaks["P2"] = []
#                     peaks["P3"] = []
#                     # peaks["P3"] = peaks["P3"][1:] # zmiana
            
#             if peaks["P1"] and peaks["P2"] and peaks["P3"]:
#                 if not (peaks["P1"][0] < peaks["P2"][0] < peaks["P3"][0]):
#                     if peaks["P3"][0] > max(peaks["P1"][0], peaks["P2"][0]):
#                         peaks["P1"] = []
#                         peaks["P2"] = []
#                     else:
#                         peaks["P1"] = []
#                         peaks["P2"] = []
#                         peaks["P3"] = []

#         # =========================
#         # -------- Class4 ---------
#         # =========================
#         elif class_id == "Class4":
#             # tylko P2, jeśli kilka → średnia
#             peaks["P2"] = middle_peak(peaks["P2"])
#             peaks["P1"] = []
#             peaks["P3"] = []
            
#         item["peaks_detected"] = _finalize_peaks(peaks)

#     return results_post


def postprocess_peaks(results_combined):
    """
    Postprocess peaks_detected zgodnie z regułami DALSZE DZIAŁANIA.
    """

    results_post=copy.deepcopy(results_combined)

    for item in results_post:
        class_id = item["class"]
        peaks = item["peaks_detected"]

        # pomocnicze
        def mean_peak(p):
            return [int(round(np.mean(p)))] if len(p) > 0 else []

        def first_peak(p):
            return [int(p[0])] if len(p) > 0 else []

        def last_peak(p):
            return [int(p[-1])] if len(p) > 0 else []
        
        def middle_peak(p):
            """
            Środkowy element listy.
            Dla parzystej liczby → wcześniejszy z dwóch środkowych.
            """
            if len(p) == 0:
                return []
            idx = (len(p) - 1) // 2
            return [int(p[idx])]
        
        def second_peak(p):
            """
            Drugi element listy (drugi w czasie).
            Jeśli <2 elementy → [].
            """
            if len(p) >= 2:
                return [int(p[1])]
            return p
                
	
        
        def _finalize_peaks(peaks):
            """
            []  -> np.nan
            [x] -> x
            """
            out = {}
            for k, v in peaks.items():
                if len(v) == 0:
                    out[k] = np.nan
                else:
                    out[k] = int(v[0])
            return out
        
        # ==========================================================
        # NOWY WARUNEK WSTĘPNY: Kolizja P1 i P3 (dla wszystkich klas)
        # ==========================================================
        if peaks["P1"] and peaks["P3"]:
            # Usuwamy tylko te piki P3, które są zbyt blisko jakiegokolwiek piku P1
            peaks["P3"] = [
                p3 for p3 in peaks["P3"] 
                if not any(abs(p3 - p1) <= 6 for p1 in peaks["P1"])
            ]
            
        # =========================
        # -------- Class1 ---------
        # =========================
        if class_id == "Class1":
            # P3: jeśli dwa → zostaw późniejszy
            peaks["P3"] = second_peak(peaks["P3"])

            # P1: jeśli kilka → średnia
            peaks["P1"] = middle_peak(peaks["P1"])

            # P2: jesli kilka - pierwszy
            peaks["P2"] = middle_peak(peaks["P2"])

            # jeśli P2 i P3 zbyt blisko → usuń oba
            if peaks["P2"] and peaks["P3"]:
                if abs(peaks["P3"][0] - peaks["P2"][0]) <= 5:
                    peaks["P2"] = []
                    peaks["P3"] = []
            
            # --- NOWY WARUNEK (Class1): P3 pomiędzy dwoma P2 ---
            if len(peaks["P2"]) == 2 and len(peaks["P3"]) == 1:
                x, y = min(peaks["P2"]), max(peaks["P2"])
                z = peaks["P3"][0]
                if x < z < y:
                    peaks["P2"] = []
            
            # warunek P1 < P2 < P3
            if peaks["P1"] and peaks["P2"] and peaks["P3"]:
                if not (peaks["P1"][0] < peaks["P2"][0] < peaks["P3"][0]):
                    # jeśli P3 jest najpóźniejszy → zostaw tylko P3
                    if peaks["P3"][0] > max(peaks["P1"][0], peaks["P2"][0]):
                        peaks["P1"] = []
                        peaks["P2"] = []
                    else:
                        peaks["P1"] = []
                        peaks["P2"] = []
                        peaks["P3"] = []

        # =========================
        # -------- Class2 ---------
        # =========================
        elif class_id == "Class2":
            # P3: jeśli 2 → drugi, jeśli >2 → środkowy SPRAWDZONE - JEST OK
            peaks["P3"] = second_peak(peaks["P3"])

            # if peaks["P3"]:
            #     peaks["P2"] = [p for p in peaks["P2"] if p <= peaks["P3"][0]]
            # P2: ostatni, a jeśli dokładnie 3 → środkowy
            # if len(peaks["P2"]) == 3:
            peaks["P2"] = middle_peak(peaks["P2"]) # OK
            # else:
                # peaks["P2"] = first_peak(peaks["P2"])
                # 
            # P1: pierwszy
            peaks["P1"] = middle_peak(peaks["P1"])
            
            # jeśli P2 i P3 zbyt blisko → usuń oba
            if peaks["P2"] and peaks["P3"]:
                if abs(peaks["P3"][0] - peaks["P2"][0]) <= 5:
                    peaks["P2"] = []
                    peaks["P3"] = []

            # warunek P1 < P2 < P3
            if peaks["P1"] and peaks["P2"] and peaks["P3"]:
                if not (peaks["P1"][0] < peaks["P2"][0] < peaks["P3"][0]):
                    if peaks["P3"][0] > max(peaks["P1"][0], peaks["P2"][0]):
                        peaks["P1"] = []
                        peaks["P2"] = []
                    else:
                        peaks["P1"] = []
                        peaks["P2"] = []
                        peaks["P3"] = []

        # =========================
        # -------- Class3 ---------
        # =========================
        elif class_id == "Class3":
            # P3: średnia DRUGI SPRAWDZONE OK
            peaks["P3"] = second_peak(peaks["P3"])

            # P1: średnia
            peaks["P1"] = second_peak(peaks["P1"])

            # P2: pierwszy
            peaks["P2"] = first_peak(peaks["P2"])

            # jeśli P2 i P3 zbyt blisko → usuń P3
            if peaks["P2"] and peaks["P3"]:
                if abs(peaks["P3"][0] - peaks["P2"][0]) <= 5:
                    peaks["P3"] = []
            
            if peaks["P1"] and peaks["P2"] and peaks["P3"]:
                if not (peaks["P1"][0] < peaks["P2"][0] < peaks["P3"][0]):
                    if peaks["P3"][0] > max(peaks["P1"][0], peaks["P2"][0]):
                        peaks["P1"] = []
                        peaks["P2"] = []
                    else:
                        peaks["P1"] = []
                        peaks["P2"] = []
                        peaks["P3"] = []

        # =========================
        # -------- Class4 ---------
        # =========================
        elif class_id == "Class4":
            # tylko P2, jeśli kilka → średnia
            peaks["P2"] = first_peak(peaks["P2"])
            peaks["P1"] = []
            peaks["P3"] = []
            
        item["peaks_detected"] = _finalize_peaks(peaks)

    return results_post

# def compute_peak_metrics_all_peaks(detection_results, class_name):
#     """
#     Łączy metryki wszystkich pików P1, P2, P3 dla każdego pliku/metody.
#     Zwraca DataFrame z osobnymi błędami i liczby pików dla P1,P2,P3,
#     Signals_with_all_Peaks i Num_Signals_in_Class.
#     Obsługuje brakujące piki w klasach.
#     """
#     expected_peaks = ["P1", "P2", "P3"]

#     # filtrowanie po klasie
#     class_signals = [item for item in detection_results if item["class"] == class_name]

#     # grupujemy po file i metodzie
#     grouped = {}
#     for item in class_signals:
#         key = (item["file"], item["method"])
#         if key not in grouped:
#             grouped[key] = {}
#         grouped[key][item["peak"]] = item

#     metrics_list = []

#     for (file_name, method_name), peaks_dict in grouped.items():
#         row = {
#             "Class": class_name,
#             "File": file_name,
#             "Method": method_name,
#         }

#         signals_all_detected = True  # czy wszystkie oczekiwane piki są wykryte w tym pliku

#         for peak in expected_peaks:
#             item = peaks_dict.get(peak)
#             if item is None:
#                 # brak piku w danym pliku
#                 row.update({
#                     f"Mean_X_Error_{peak}": np.nan,
#                     f"Mean_Y_Error_{peak}": np.nan,
#                     f"Mean_XY_Error_{peak}": np.nan,
#                     f"Min_XY_Error_{peak}": np.nan,
#                     f"Peak_Count_{peak}": 0
#                 })
#                 signals_all_detected = False
#                 continue

#             ref_idx = item["peaks_ref"].get(peak)
#             detected = item["peaks_detected"].get(peak, [])

#             sig_df = item["signal"]
#             t = sig_df.iloc[:, 0].values
#             y = sig_df.iloc[:, 1].values

#             if ref_idx is None or len(detected) == 0:
#                 dx = dy = dxy = np.array([np.nan])
#                 peak_count = 0
#                 signals_all_detected = False
#             else:
#                 t_detected = t[detected]
#                 y_detected = y[detected]
#                 if np.isscalar(ref_idx):
#                     dx = abs(t_detected - t[ref_idx])
#                     dy = abs(y_detected - y[ref_idx])
#                 else:
#                     dx = np.min([abs(t_detected - t[r]) for r in ref_idx], axis=0)
#                     dy = np.min([abs(y_detected - y[r]) for r in ref_idx], axis=0)
#                 dxy = np.sqrt(dx**2 + dy**2)
#                 peak_count = len(detected)

#             row.update({
#                 f"Mean_X_Error_{peak}": np.mean(dx),
#                 f"Mean_Y_Error_{peak}": np.mean(dy),
#                 f"Mean_XY_Error_{peak}": np.mean(dxy),
#                 f"Min_XY_Error_{peak}": np.min(dxy),
#                 f"Peak_Count_{peak}": peak_count
#             })

#         row["Signals_with_all_Peaks"] = int(signals_all_detected)
#         metrics_list.append(row)

#     df_metrics = pd.DataFrame(metrics_list)

#     # Num_Signals_in_Class = liczba wszystkich plików w klasie
#     df_metrics["Num_Signals_in_Class"] = len(class_signals)

#     return df_metrics


def has_peak(v):
    """Zwraca True, jeśli pik jest wykryty (nie NaN, nie pusta lista)."""
    if v is None:
        return False
    if isinstance(v, list):
        return len(v) > 0
    try:
        return not math.isnan(v)
    except TypeError:
        return True  # np. int



def select_problematic_rows(results_combined, max_dist=20):
    selected = []

    for item in results_combined:
        class_id = item["class"]
        det = item["peaks_detected"]
        ref = item["peaks_ref"]

        # --- WARUNEK 1: brak pików ---
        if class_id == "Class4":
            empty_condition = not has_peak(det.get("P2"))
        else:
            empty_condition = any(not has_peak(det.get(p)) for p in ["P1", "P2", "P3"])

        # --- WARUNEK 2: duża odległość ---
        dist_condition = any(
            has_peak(det.get(p)) and has_peak(ref.get(p)) and abs(det[p] - ref[p]) > max_dist
            for p in ["P1", "P2", "P3"]
        )

        if empty_condition or dist_condition:
            selected.append(item)

    return selected


def compute_metrics_pre_postproc(dataset, class_name):
    """
    Oblicza metryki błędów dla danych przed postprocessingiem.
    
    dataset: lista słowników (przed postproc)
    class_name: np. "Class1"
    
    Zwraca DataFrame:
        Class, Peak, Mean_X_Error, Mean_Y_Error, Mean_XY_Error, Std_XY_Error, Num_Files
    """
    expected_peaks = ["P1", "P2", "P3"] if class_name != "Class4" else ["P2"]
    metrics_list = []
    # filtrowanie po klasie
    # class_signals = [d for d in dataset if d["class"] == class_name]
    grouped = defaultdict(list)
    for d in dataset:
        if d["class"] == class_name:
            grouped[(d["file"], d["peak"])].append(d)
    n_files = 250 if class_name in ["Class1","Class2","Class3"] else 50
    tolerance = 3
    
    for peak in expected_peaks:
        dx_all, dy_all, dxy_all, peak_count_sum = [], [], [], 0
        TP_sum, FP_sum, FN_sum = 0, 0, 0

        # for item in class_signals:
        for (file_name, peak_name), items in grouped.items():  # zmiana
            if peak_name != peak:                              # zmiana
                continue                                       # zmiana

            # sig = item["signal_raw"]
            # t = sig.iloc[:,0].values
            # y = sig.iloc[:,1].values

            # ref_idx = item["peaks_ref"].get(peak)
            # detected = item["peaks_detected"].get(peak, [])
            detected_all = []                                      # zmiana
            for it in items:                                       # zmiana
                detected_all.extend(it["peaks_detected"].get(peak, []))  # zmiana
            item0 = items[0]                                       # zmiana
            sig = item0["signal_raw"]                              # zmiana
            t = sig.iloc[:, 0].values                              # zmiana
            y = sig.iloc[:, 1].values                              # zmiana
            ref_idx = item0["peaks_ref"].get(peak)                 # zmiana

            # if ref_idx is None or len(detected) == 0:
            #     FN_sum += 1 
            #     continue
            if ref_idx is None or len(detected_all) == 0:           # zmiana
                FN_sum += 1                                         # zmiana
                continue                                            # zmiana

            # # dopasowanie wielu wykrytych do referencji
            # if np.isscalar(ref_idx):
            #     dx = np.abs(t[detected] - t[ref_idx])
            #     dy = np.abs(y[detected] - y[ref_idx])
            # else:
            #     dx = np.sum([np.abs(t[detected] - t[r]) for r in ref_idx], axis=0)
            #     dy = np.sum([np.abs(y[detected] - y[r]) for r in ref_idx], axis=0)

            # dxy = np.sqrt(dx**2 + dy**2)
            dx = np.abs(t[detected_all] - t[ref_idx])               # zmiana
            dy = np.abs(y[detected_all] - y[ref_idx])               # zmiana
            dxy = np.sqrt(dx**2 + dy**2)                             # zmiana
            
            TP = int(np.any(dx <= tolerance))                        # zmiana
            FN = int(TP == 0)                                        # zmiana
            FP = int(np.sum(dx > tolerance))                         # zmiana
            
            TP_sum += TP
            FP_sum += FP
            FN_sum += FN

            # agregacja po pliku: średnia jeśli wykryto kilka
            dx_all.append(np.mean(dx))
            dy_all.append(np.mean(dy))
            dxy_all.append(np.mean(dxy))
            
            peak_count_sum += len(detected_all)

        metrics_list.append({
            "Class": class_name,
            "Peak": peak,
            "Mean_X_Error": np.mean(dx_all) if dx_all else np.nan,
            "Mean_Y_Error": np.mean(dy_all) if dy_all else np.nan,
            "Mean_XY_Error": np.mean(dxy_all) if dxy_all else np.nan,
            "Std_XY_Error": np.std(dxy_all) if dxy_all else np.nan,
            "TP": TP_sum,    # zmiana
            "FP": FP_sum,    # zmiana
            "FN": FN_sum,     # zmiana
            "%_Files_With_Peak": len(dx_all)/n_files,
            "Peak_Count": peak_count_sum
        })

    return pd.DataFrame(metrics_list)


def compute_metrics_postproc(dataset, class_name):
    """
    Oblicza metryki błędów dla danych po postprocessingiem.
    
    dataset: lista słowników (po postproc)
    class_name: np. "Class1"
    
    Zwraca DataFrame:
        Class, Peak, Mean_X_Error, Mean_Y_Error, Mean_XY_Error, Std_XY_Error, Num_Files
    """
    expected_peaks = ["P1", "P2", "P3"] if class_name != "Class4" else ["P2"]
    metrics_list = []
    class_signals = [d for d in dataset if d["class"] == class_name]
    n_files = 250 if class_name in ["Class1","Class2","Class3"] else 50
    tolerance = 3
    
    for peak in expected_peaks:
        dx_all, dy_all, dxy_all, peak_count_sum = [], [], [], 0
        TP_sum, FP_sum, FN_sum = 0, 0, 0

        for item in class_signals:
            sig = item["signal_raw"]
            t = sig.iloc[:,0].values
            y = sig.iloc[:,1].values

            ref_idx = item["peaks_ref"].get(peak)
            detected_val = item["peaks_detected"].get(peak)

            # w postproc to albo int albo NaN
            if ref_idx is None or detected_val is None or (isinstance(detected_val, float) and math.isnan(detected_val)):
                FN_sum += 1
                continue

            if np.isscalar(ref_idx):
                dx = abs(t[detected_val] - t[ref_idx])
                dy = abs(y[detected_val] - y[ref_idx])
            else:
                dx = np.sum([abs(t[detected_val] - t[r]) for r in ref_idx])
                dy = np.sum([abs(y[detected_val] - y[r]) for r in ref_idx])

            dxy = np.sqrt(dx**2 + dy**2)
            
            TP = int(dx <= tolerance)
            FN = int(TP == 0)
            FP = int(dx > tolerance)
            
            TP_sum += TP
            FP_sum += FP
            FN_sum += FN

            dx_all.append(dx)
            dy_all.append(dy)
            dxy_all.append(dxy)
            
            peak_count_sum += 1

        metrics_list.append({
            "Class": class_name,
            "Peak": peak,
            "Mean_X_Error": np.mean(dx_all) if dx_all else np.nan,
            "Mean_Y_Error": np.mean(dy_all) if dy_all else np.nan,
            "Mean_XY_Error": np.mean(dxy_all) if dxy_all else np.nan,
            "Std_XY_Error": np.std(dxy_all) if dxy_all else np.nan,
            "TP": TP_sum,    # zmiana
            "FP": FP_sum,    # zmiana
            "FN": FN_sum,     # zmiana
            "%_Files_With_Peak": len(dx_all)/n_files,
            "Peak_Count": peak_count_sum
        })

    return pd.DataFrame(metrics_list)


# =========================
# 1. Funkcja do obliczania błędów XY dla pojedynczego piku
# =========================
def compute_dxy_per_peak(results_combined, expected_peaks=("P1","P2","P3")):
    """
    Zwraca DataFrame z błędami XY dla każdego pliku i piku.
    
    results_combined: lista słowników po postprocess
    expected_peaks: tuple oczekiwanych pików
    """
    records = []
    for item in results_combined:
        file_name = item["file"]
        class_id = item.get("class", "Unknown")
        sig = item["signal_raw"]
        t = sig.iloc[:,0].values
        y = sig.iloc[:,1].values
        for peak in expected_peaks:
            ref_idx = item["peaks_ref"].get(peak)
            detected_val = item["peaks_detected"].get(peak)
            
            # obsługa braków
            if ref_idx is None or detected_val is None or (isinstance(detected_val,float) and np.isnan(detected_val)):
                dxy = np.nan
            else:
                dx = abs(t[detected_val] - t[ref_idx])
                dy = abs(y[detected_val] - y[ref_idx])
                dxy = np.sqrt(dx**2 + dy**2)
            
            records.append({
                "file": file_name,
                "class": class_id,
                "peak": peak,
                "dxy": dxy
            })
    return pd.DataFrame(records)

# =========================
# 2. Funkcja do łączenia dwóch wariantów w pary
# =========================
def merge_dxy_for_wilcoxon(df1, df2):
    # Dodajemy "class" do listy kluczy (on=)
    merged = pd.merge(df1, df2, on=["file", "class", "peak"], suffixes=("_new","_new_simpl"))
    merged = merged.dropna(subset=["dxy_new","dxy_new_simpl"])
    return merged

# =========================
# 3. Funkcja do wykonania testu Wilcoxona dla wszystkich pików
# =========================
# def wilcoxon_test_all_peaks(merged_df):
#     """
#     merged_df: DataFrame z kolumnami ['file','peak','dxy_new','dxy_new_simpl']
#     Zwraca DataFrame z wynikami testu dla każdego piku.
#     """
#     results = []
#     for peak in merged_df["peak"].unique():
#         sub = merged_df[merged_df["peak"]==peak]
#         if len(sub) < 5:  # zbyt mało danych
#             p_value = np.nan
#             stat = np.nan
#         else:
#             stat, p_value = wilcoxon(sub["dxy_new"], sub["dxy_new_simpl"])
#         results.append({
#             "peak": peak,
#             "n_pairs": len(sub),
#             "wilcoxon_stat": stat,
#             "p_value": p_value
#         })
    # return pd.DataFrame(results)
    
def wilcoxon_test_by_class_and_peak(merged_df):
    """
    merged_df: DataFrame z kolumnami ['file', 'class', 'peak', 'dxy_new', 'dxy_new_simpl']
    """
    results = []
    # Grupowanie po dwóch kolumnach: klasie i piku
    for (class_id, peak), sub in merged_df.groupby(["class", "peak"]):
        
        # Obliczamy różnice
        diff = sub["dxy_new"] - sub["dxy_new_simpl"]
        
        # Sprawdzamy, czy wszystkie różnice są zerem (identyczne wyniki)
        if (diff == 0).all():
            p_value = 1.0  # identyczne rozkłady
            stat = 0.0
        elif len(sub) < 5:
            p_value = np.nan
            stat = np.nan
        else:
            # zero_method="pratt" pomaga, gdy jest dużo identycznych par
            try:
                stat, p_value = wilcoxon(sub["dxy_new"], sub["dxy_new_simpl"], zero_method="pratt")
            except:
                stat, p_value = np.nan, np.nan

        results.append({
            "class": class_id,
            "peak": peak,
            "n_pairs": len(sub),
            "wilcoxon_stat": stat,
            "p_value": p_value
        })
    return pd.DataFrame(results)

# =========================
# 4. Kompletny pipeline
# =========================
def run_wilcoxon_pipeline(results_new_pp, results_new_s_pp, expected_peaks=("P1","P2","P3")):
    df_new = compute_dxy_per_peak(results_new_pp, expected_peaks)
    df_simpl = compute_dxy_per_peak(results_new_s_pp, expected_peaks)
    merged = merge_dxy_for_wilcoxon(df_new, df_simpl)
    wilcoxon_results = wilcoxon_test_by_class_and_peak(merged)
    return wilcoxon_results
#, df_new, df_simpl


def prepare_metrics_df(df, version_name):
    cols = [
        "Class",
        "Peak",
        "Mean_X_Error",
        "Mean_Y_Error",
        "Mean_XY_Error",
        "TP",
        "FP",
        "FN",
    ]
    
    out = df[cols].copy()
    out["Version"] = version_name
    return out


# def closest_peak_position_stats(results_new):
#     """
#     Analiza: gdy wykryto >1 pik, który (kolejność) jest najbliżej referencji.
    
#     Zwraca DataFrame:
#         Class | Peak | Position | Count | Fraction
#     """
#     records = []

#     for item in results_new:
#         class_id = item["class"]
#         peak = item["peak"]
#         detected = item["peaks_detected"].get(peak, [])
#         ref_idx = item["peaks_ref"].get(peak)

#         if ref_idx is None or not isinstance(detected, list):
#             continue

#         if len(detected) <= 1:
#             continue  # interesują nas tylko przypadki z wieloma detekcjami

#         sig = item["signal_raw"]
#         t = sig.iloc[:, 0].values

#         # odległości czasowe
#         distances = [abs(t[d] - t[ref_idx]) for d in detected]
#         closest_pos = int(np.argmin(distances))  # 0 = pierwszy, 1 = drugi, ...

#         records.append({
#             "Class": class_id,
#             "Peak": peak,
#             "Position": closest_pos + 1  # 1-based (czytelniej do pracy)
#         })

#     df = pd.DataFrame(records)

#     summary = (
#         df
#         .groupby(["Class", "Peak", "Position"])
#         .size()
#         .reset_index(name="Count")
#     )

#     summary["Fraction"] = (
#         summary["Count"]
#         / summary.groupby(["Class", "Peak"])["Count"].transform("sum")
#     )

#     return summary.sort_values(["Class", "Peak", "Position"])

def closest_peak_position_stats(results_new, tol=3):
    """
    Analiza: gdy wykryto >1 pik, który (kolejność) jest najbliżej referencji,
    z ignorowaniem różnic <= tol próbek.

    Zwraca DataFrame:
        Class | Peak | Position | Count | Fraction
    """
    records = []

    for item in results_new:
        class_id = item["class"]
        peak = item["peak"]
        detected = item["peaks_detected"].get(peak, [])
        ref_idx = item["peaks_ref"].get(peak)

        if ref_idx is None or not isinstance(detected, list):
            continue

        if len(detected) <= 1:
            continue

        # odległości w próbkach (bez czasu ciągłego)
        distances = [abs(d - ref_idx) for d in detected]

        # sortowanie wg odległości
        order = np.argsort(distances)
        best = order[0]
        second = order[1]

        # ignoruj kosmetyczne różnice
        if abs(distances[best] - distances[second]) <= tol:
            continue

        records.append({
            "Class": class_id,
            "Peak": peak,
            "Position": int(best) + 1  # 1-based
        })

    df = pd.DataFrame(records)

    if df.empty:
        return df

    summary = (
        df
        .groupby(["Class", "Peak", "Position"])
        .size()
        .reset_index(name="Count")
    )

    summary["Fraction"] = (
        summary["Count"]
        / summary.groupby(["Class", "Peak"])["Count"].transform("sum")
    )

    return summary.sort_values(["Class", "Peak", "Position"])



"""
Class1 P1:
    a) avg, concave (min blad) 
    b) full, concave (max. signals w/ peaks)
    
Class1 P2:
    a) smooth 4Hz, avg, curvature (min blad) 
    b) avg, curvature (max signals w/ peaks)   
    ----------- ZMIANA ----------------------
        a,b) smooth 4Hz, avg, curvature 
    
Class1 P3:
    a,b) smooth 4Hz, full, concave   

    
Class2 P1:
    a) smooth 3Hz, avg, hilbert (min blad, 249)
    b) smooth 3Hz, avg, wavelet (podobny, 250) SAME_AVG!!!!!!!!!
    -------- ZMIANA -------------------
        a,b) smooth 3Hz, avg, hilbert
    
Class2 P2:
    a) whiskers, wavelet (min blad)
    b) smooth 4Hz, full, curvature (max sig w/ peaks)
    ----------- ZMIANA -------------
    a) smooth 4Hz, whiskers, line distance (vertical lub perpendicular)
    b) smooth 4Hz, full, line distance (vertical lub perpendicular)
    
Class2 P3:
    a) avg, curvature (min blad)
    b) whiskers, hilbert (doslownie o 2 wiecej sig w/ peaks)
    ----------- ZMIANA -----------
    a) full, hilbert
    b) pm3, hilbert
    

Class3 P1:
    a,b) smooth 4Hz, full, wavelet LUB smooth 4Hz, whiskers, wavelet // (AVG)
    
Class3 P2:
    a) whiskers, concave (min blad)
    b) smooth 4Hz, full, concave (max sig)
    ----------- ZMIANA ------------
    a) smooth 4Hz, avg, hilbert // <----- tylko b?
    b) smooth 4Hz, avg, wavelet

Class3 P3:
    a,b) none, modified scholkmann 1/2 99 LUB none, modified scholkmann 1 99
    ------------ ZMIANA ------------
    a) full modified scholkmann 1/2 99
    b) none modified scholkmann 1/2 99

Class4 P2:
    a,b) smooth 4Hz, none, modified scholkmann 1/2 99 LUB smooth 4Hz, none, modified scholkmann 1 99
    ---------- ZMIANA ------------
    a) full, conncave d2x=0,002
    b) pm3, concave d2x=0,002
    
DALSZE DZIALANIA:
Class1: 
    - jesli sa dwa P3 to wziac tylko pozniejszy i usunac pierwszy (np. [80, 96] -> [96])
    - jesli dwa P2 to pierwszy
    - jesli P2 i P3 sa w odleglosci mniejszej niz 3 probki to usun P2 i P3 (zostaw puste listy)
    - jesli kilka P1 to srednia 
    PO WYKONANIU TYCH OPERCAJI:
    - jesli nie jest spelniony warunek P1<P2<P3 (na osi X!) to usun P1, P2 i P3
    (jesli P1>P2 ale P3 jest najpozniejsze to zostaw P3)
    
Class2:
    - jesli jest kilka P3 to srednia
    - jesli jest kilka P2 to wziac pierwszy
    - jesli sa dwa/kilka P1 to wziac pierwszy
    PO WYKONANIU TYCH OPERCAJI:
    - jesli nie jest spelniony warunek P1<P2<P3 (na osi X!) to usun P1, P2 i P3
    (jesli P1>P2 ale P3 jest najpozniejsze to zostaw P3)
    
Class3:
    - jesli jest kilka P3 to srednia
    - jesli jest kilka P1 to srednia
    - jesli jest kilka P2 to wziac pierwszy
    - jesli P2 i P3 sa w odleglosci mniejszej niz 3 probki to usun P3
    
Class4:
    - jesli jest kilka P2 to srednia
    
"""

# --- wariant NOWY ---
df_new = pd.DataFrame([
    # -------- Class1 --------
    ("Class1", "P1", "it2",            "it2_smooth_4Hz", "avg", "concave"), # lub 0,002
    ("Class1", "P2", "it2_smooth_4Hz", "it2_smooth_4Hz", "avg",  "line_distance_3"),
    ("Class1", "P3", "it2",            "it2_smooth_4Hz", "avg", "concave"), # lub 0,002

    # -------- Class2 --------
    ("Class2", "P1", "it2",            "it2_smooth_4Hz", "avg",  "concave"),
    ("Class2", "P2", "it2",            "it2_smooth_4Hz", "avg", "concave"), # full hilbert tez ok
    ("Class2", "P3", "it2",            "it2_smooth_4Hz", "full", "hilbert"), # whiskers hilbert tez ok

    # -------- Class3 --------
    ("Class3", "P1", "it2_smooth_4Hz", "it2_smooth_4Hz", "avg", "wavelet"),
    ("Class3", "P2", "it2",            "it2_smooth_4Hz", "avg",  "concave"),
    ("Class3", "P3", "it2",            "it2_smooth_4Hz", "full", "hilbert"), # whiskers hilbert tez ok

    # -------- Class4 --------
    ("Class4", "P2", "it2", "it2_smooth_4Hz", "full", "concave_d2x=0-002"),
],
columns=[
    "class",
    "peak",
    "detect_dataset",
    "ranges_dataset",
    "range_type",
    "method"
])

df_new_simplified = pd.DataFrame([
    # -------- Class1 --------
    ("Class1", "P1", "it2",            "it2_smooth_4Hz", "avg", "concave"), # lub 0,002
    ("Class1", "P2", "it2",            "it2_smooth_4Hz", "avg",  "concave"),
    ("Class1", "P3", "it2",            "it2_smooth_4Hz", "avg", "concave"), # lub 0,002

    # -------- Class2 --------
    ("Class2", "P1", "it2",            "it2_smooth_4Hz", "avg",  "concave"),
    ("Class2", "P2", "it2",            "it2_smooth_4Hz", "avg", "concave"), # full hilbert tez ok
    ("Class2", "P3", "it2",            "it2_smooth_4Hz", "full", "hilbert"), # whiskers hilbert tez ok

    # -------- Class3 --------
    ("Class3", "P1", "it2",            "it2_smooth_4Hz", "avg", "concave"),
    ("Class3", "P2", "it2",            "it2_smooth_4Hz", "avg",  "concave"),
    ("Class3", "P3", "it2",            "it2_smooth_4Hz", "full", "hilbert"), # whiskers hilbert tez ok

    # -------- Class4 --------
    ("Class4", "P2", "it2",            "it2_smooth_4Hz", "full", "concave"), # lub concave
],
columns=[
    "class",
    "peak",
    "detect_dataset",
    "ranges_dataset",
    "range_type",
    "method"
])

datasets_dict = {
    "it2": it2,
    "it2_smooth_4Hz": it2_smooth_4Hz,
    "it2_smooth_3Hz": it2_smooth_3Hz,
}


out_dir = "rysunki_4_01"
# %% ================= DRUGI ZESTAW (IT2) ====================
# results_a_it2 = run_variant(
#     df_variant_a,
#     datasets_dict=datasets_dict,
#     ranges_all_time=ranges_all_time,
#     ranges_all_amps=ranges_all_amps)

# results_combined_a_it2 = combine_peaks_by_file(results_a_it2)
# results_combined_a_it2_pp = postprocess_peaks(results_combined_a_it2)

# df_pogladowe_a_it2_pp = pd.DataFrame([
#     {
#         "file": d["file"],
#         "peaks_ref": d["peaks_ref"],
#         "peaks_detected": d["peaks_detected"],
# }
#     for d in results_combined_a_it2_pp])



# results_b_it2 = run_variant(
#     df_variant_b,
#     datasets_dict=datasets_dict,
#     ranges_all_time=ranges_all_time,
#     ranges_all_amps=ranges_all_amps)

# results_combined_b_it2 = combine_peaks_by_file(results_b_it2)
# results_combined_b_it2_pp = postprocess_peaks(results_combined_b_it2)

# df_pogladowe_b_it2_pp = pd.DataFrame([
#     {
#         "file": d["file"],
#         "peaks_ref": d["peaks_ref"],
#         "peaks_detected": d["peaks_detected"],
#     }
#     for d in results_combined_b_it2_pp])

# %% ============== OSTATECZNE STARCIE ===============
results_new = run_variant(
    df_new,
    datasets_dict=datasets_dict,
    ranges_all_time=ranges_all_time,
    ranges_all_amps=ranges_all_amps)


# rows = []

# for item in results_new:
#     class_id = item["class"]
#     file_name = item["file"]
    
#     for peak, detected in item["peaks_detected"].items():
#         if class_id == "Class4" and peak != "P2":
#             continue
        
#         n_det = len(detected) if isinstance(detected, list) else 0
        
#         rows.append({
#             "Class": class_id,
#             "Peak": peak,
#             "File": file_name,
#             "n_detections": n_det
#         })

# df_counts = pd.DataFrame(rows)

# def detection_category(n):
#     if n == 1:
#         return "1"
#     elif n == 2:
#         return "2"
#     elif n >= 3:
#         return "≥3"
#     else:
#         return "0"   # brak detekcji – opcjonalnie

# df_counts["Category"] = df_counts["n_detections"].apply(detection_category)

# summary = (
#     df_counts
#     .groupby(["Class", "Peak", "Category"])
#     .size()
#     .unstack(fill_value=0)
#     .reset_index()
# )

# print(summary)


results_combined_new = combine_peaks_by_file(results_new)
results_combined_new_pp = postprocess_peaks(results_combined_new)

df_pogladowe_new = pd.DataFrame([
    {
        "file": d["file"],
        "peaks_ref": d["peaks_ref"],
        "peaks_detected": d["peaks_detected"],
    }
    for d in results_combined_new])

df_pogladowe_new_pp = pd.DataFrame([
    {
        "file": d["file"],
        "peaks_ref": d["peaks_ref"],
        "peaks_detected": d["peaks_detected"],
    }
    for d in results_combined_new_pp])


classes = ["Class1", 
           "Class2", 
           "Class3", "Class4"
           ]

# for cls in classes:
#     plot_files_in_class(results_combined_new, cls)

df_metrics_pre = pd.concat([compute_metrics_pre_postproc(results_new, cls) for cls in classes], ignore_index=True)
df_metrics_post = pd.concat([compute_metrics_postproc(results_combined_new_pp, cls) for cls in classes], ignore_index=True)
# df_metrics_pre.to_csv("02_01_metrics_pre.csv", index=False)
# df_metrics_post.to_csv("02_01metrics_post.csv", index=False)



mean_dxy = df_metrics_post["Mean_XY_Error"].mean()

mean_dx = df_metrics_post['Mean_X_Error'].mean()

mean_dy = df_metrics_post['Mean_Y_Error'].mean()

# Sumowanie TP, FP, FN po wszystkich wierszach
TP_total = df_metrics_post['TP'].sum()
FP_total = df_metrics_post['FP'].sum()
FN_total = df_metrics_post['FN'].sum()

# Czułość (sensitivity / recall)
sensitivity = 100 * TP_total / (TP_total + FN_total) if (TP_total + FN_total) > 0 else np.nan

# Precyzja (precision)
precision = 100 * TP_total / (TP_total + FP_total) if (TP_total + FP_total) > 0 else np.nan

print(f"Średni błąd: {mean_dxy:.3f}")
print(f"Średni błąd: {mean_dx:.3f}")
print(f"Średni błąd: {mean_dy:.3f}")
print(f"Czułość (Sensitivity %): {sensitivity:.2f}%")
print(f"Precyzja (Precision %): {precision:.2f}%")
"""

# sumowanie wartości TP, FP, FN
TP_total = df_metrics_post['TP'].sum()
FP_total = df_metrics_post['FP'].sum()
FN_total = df_metrics_post['FN'].sum()

# średni błąd XY uśredniony między pikami
mean_dxy = df_metrics_post['Mean_XY_Error'].mean()

# czułość / sensitivity
sensitivity = 100 * TP_total / (TP_total + FN_total)

# precyzja / precision
precision = 100 * TP_total / (TP_total + FP_total)

# wyświetlenie krok po kroku
print("ŚREDNI BŁĄD XY (uśredniony między pikami):")
print(f"Mean_dxy = sum(Mean_XY_Error dla wszystkich pików) / liczba pików")
print(f"Mean_dxy = {mean_dxy:.3f}\n")

print("CZUŁOŚĆ / SENSITIVITY:")
print(f"Sensitivity = TP / (TP + FN) * 100%")
print(f"Sensitivity = {TP_total} / ({TP_total} + {FN_total}) * 100%")
print(f"Sensitivity = {sensitivity:.2f}%\n")

print("PRECYZJA / PRECISION:")
print(f"Precision = TP / (TP + FP) * 100%")
print(f"Precision = {TP_total} / ({TP_total} + {FP_total}) * 100%")
print(f"Precision = {precision:.2f}%")

"""

# plot_selected_files(
#     detection_results=results_combined_new_pp,
#     class_name="Class1",
#     files_to_plot=["Class1_it2_example_0119", 
#                    "Class1_it2_example_0083",
#                    "Class1_it2_example_0080",
#                    "Class1_it2_example_0109"],
#     ncols=2
# )
# plt.savefig("rysunki/final_detection_examples_k1.pdf", 
#             bbox_inches=None)

# plot_selected_files(
#     detection_results=results_combined_new_pp,
#     class_name="Class2",
#     files_to_plot=["Class2_it2_example_0022", 
#                    "Class2_it2_example_0002",
#                    "Class2_it2_example_0058"],
#     ncols=3
# )
# plt.savefig("rysunki/final_detection_examples_k2.pdf", 
#             bbox_inches=None)

# plot_selected_files(
#     detection_results=results_combined_new_pp,
#     class_name="Class3",
#     files_to_plot=["Class3_it2_example_0004", 
#                    "Class3_it2_example_0105",
#                    "Class3_it2_example_0047"],
#     ncols=3
# )
# plt.savefig("rysunki/final_detection_examples_k3.pdf", 
#             bbox_inches=None)


# plot_selected_files(
#     detection_results=results_combined_new_pp,
#     class_name="Class4",
#     files_to_plot=["Class4_it2_example_0033", 
#                    "Class4_it2_example_0024",
#                    "Class4_it2_example_0019"],
    
#     ncols=3
# )
# plt.savefig("rysunki/final_detection_examples_k4.pdf", 
#             bbox_inches=None)

classes = ["Class1", "Class2", "Class3"]
for class_id in classes:
    # plot_upset_classic_postproc_new(results_combined_new, class_id)
    plot_upset_classic_postproc_new(results_combined_new_pp, class_id)
    plt.savefig(
        os.path.join(out_dir, f"upset_{class_id}_post.pdf"),
        format="pdf",
        bbox_inches="tight"
    )

plot_peak_detection_pie_new(results_combined_new_pp, "Class4")
plt.savefig(
    os.path.join(out_dir, "pie_Class4_P2_post.pdf"),
    format="pdf",
    bbox_inches="tight"
)


results_new_simpl = run_variant(
    df_new_simplified,
    datasets_dict=datasets_dict,
    ranges_all_time=ranges_all_time,
    ranges_all_amps=ranges_all_amps)


# rows = []

# for item in results_new:
#     class_id = item["class"]
#     file_name = item["file"]
    
#     for peak, detected in item["peaks_detected"].items():
#         if class_id == "Class4" and peak != "P2":
#             continue
        
#         n_det = len(detected) if isinstance(detected, list) else 0
        
#         rows.append({
#             "Class": class_id,
#             "Peak": peak,
#             "File": file_name,
#             "n_detections": n_det
#         })

# df_counts = pd.DataFrame(rows)

# def detection_category(n):
#     if n == 1:
#         return "1"
#     elif n == 2:
#         return "2"
#     elif n >= 3:
#         return "≥3"
#     else:
#         return "0"   # brak detekcji – opcjonalnie

# df_counts["Category"] = df_counts["n_detections"].apply(detection_category)

# summary2 = (
#     df_counts
#     .groupby(["Class", "Peak", "Category"])
#     .size()
#     .unstack(fill_value=0)
#     .reset_index()
# )

# print(summary2)


results_combined_new_s = combine_peaks_by_file(results_new_simpl)
results_combined_new_s_pp = postprocess_peaks(results_combined_new_s)

df_pogladowe_new_s = pd.DataFrame([
    {
        "file": d["file"],
        "peaks_ref": d["peaks_ref"],
        "peaks_detected": d["peaks_detected"],
    }
    for d in results_combined_new_s])

df_pogladowe_new_s_pp = pd.DataFrame([
    {
        "file": d["file"],
        "peaks_ref": d["peaks_ref"],
        "peaks_detected": d["peaks_detected"],
    }
    for d in results_combined_new_s_pp])

classes = ["Class1", "Class2", "Class3",
           "Class4"
           ]
# for cls in classes:
#     plot_files_in_class(results_combined_new, cls)

# df_closest = closest_peak_position_stats(results_new)
# print(df_closest)



df_metrics_pre_s = pd.concat([compute_metrics_pre_postproc(results_new_simpl, cls) for cls in classes], ignore_index=True)
df_metrics_post_s = pd.concat([compute_metrics_postproc(results_combined_new_s_pp, cls) for cls in classes], ignore_index=True)
# df_metrics_pre.to_csv("27_12_metrics_pre_simpl.csv", index=False)
# df_metrics_post.to_csv("27_12_metrics_post_simpl.csv", index=False)

# ========== JEDEN PLIK WYNIKOWY ==================

df_fit_pre  = prepare_metrics_df(
    df_metrics_pre,
    "dopasowany – przed postprocessingiem"
)

df_fit_post = prepare_metrics_df(
    df_metrics_post,
    "dopasowany – po postprocessingu"
)

df_s_pre = prepare_metrics_df(
    df_metrics_pre_s,
    "uproszczony – przed postprocessingiem"
)

df_s_post = prepare_metrics_df(
    df_metrics_post_s,
    "uproszczony – po postprocessingu"
)

df_metrics_all = pd.concat(
    [df_fit_pre, df_fit_post, df_s_pre, df_s_post],
    ignore_index=True
)

df_metrics_all = df_metrics_all[
    ["Class", "Peak", "Version",
     "Mean_X_Error", "Mean_Y_Error", "Mean_XY_Error",
     "TP", "FP", "FN"]
]

version_order = [
    "dopasowany – przed postprocessingiem",
    "dopasowany – po postprocessingu",
    "uproszczony – przed postprocessingiem",
    "uproszczony – po postprocessingu"
]

df_metrics_all["Version"] = pd.Categorical(
    df_metrics_all["Version"],
    categories=version_order,
    ordered=True
)


df_metrics_all = df_metrics_all.sort_values(
    ["Class", "Peak", "Version"]
).reset_index(drop=True)

df_metrics_all.to_csv("NEW_metrics_4_01.csv", index=False)




# =========== WILCOXON ====================
# ============================================

# expected_peaks = ["P1","P2","P3"]
# wilcoxon_results = run_wilcoxon_pipeline(results_combined_new_pp, results_combined_new_s_pp, expected_peaks)
# print(wilcoxon_results)

# # 2. Obliczenie wyniku GLOBALNEGO (dla całej pracy)
# # Łączymy wszystkie pary dxy z wyników szczegółowych w jeden wektor
# df_new = compute_dxy_per_peak(results_combined_new_pp, expected_peaks)
# df_simpl = compute_dxy_per_peak(results_combined_new_s_pp, expected_peaks)
# merged_global = merge_dxy_for_wilcoxon(df_new, df_simpl)

# # Wykonujemy test na całości danych

# stat_glob, p_glob = wilcoxon(merged_global["dxy_new"], 
#                              merged_global["dxy_new_simpl"], 
#                              zero_method="pratt")

# print(f"\n--- WYNIK GLOBALNY ---")
# print(f"Suma par do testu: {len(merged_global)}")
# print(f"p-value globalne: {p_glob:.20f}")

# # Test jednostronny: czy błąd 'new' jest ISTOTNIE MNIEJSZY niż 'simpl'?
# stat, p_less = wilcoxon(merged_global["dxy_new"], 
#                         merged_global["dxy_new_simpl"], 
#                         alternative='less', 
#                         zero_method="pratt")

# if p_less < 0.05:
#     print("Algorytm dopasowany ma istotnie mniejsze błędy (ma przewagę).")
# else: print("Łojojo") 

# tp_total_new = df_metrics_post['TP'].sum()
# tp_total_simpl = df_metrics_post_s['TP'].sum()

# summary = {
#     "Dopasowany": {
#         "TP_total": tp_total_new, 
#         "Mean_Err": df_new['dxy'].mean()
#     },
#     "Uproszczony": {
#         "TP_total": tp_total_simpl, 
#         "Mean_Err": df_simpl['dxy'].mean()
#     }
# }

# print(f"Suma TP (Dopasowany): {tp_total_new}")
# print(f"Suma TP (Uproszczony): {tp_total_simpl}")

# diff = merged_global["dxy_new"] - merged_global["dxy_new_simpl"]
# print(diff.median(), diff.mean())


# merged_before = pd.merge(
#     df_new,
#     df_simpl,
#     on=["file","class","peak"],
#     suffixes=("_new","_new_simpl")
# )

# merged_after = merged_before.dropna(
#     subset=["dxy_new","dxy_new_simpl"]
# )

# print(len(merged_before))
# print(len(merged_after))


# def plot_error_comparison(merged_df):
#     plt.figure(figsize=(8, 8))
    
#     # Dane do wykresu
#     x = merged_df["dxy_new_simpl"]
#     y = merged_df["dxy_new"]
    
#     # Wykres rozrzutu
#     plt.scatter(x, y, alpha=0.5, color='royalblue', label='Pary detekcji')
    
#     # Linia y = x (linia odniesienia)
#     max_val = max(x.max(), y.max())
#     plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', label='y = x (Błędy równe)')
    
#     # Formatowanie
#     plt.title('Porównanie błędów lokalizacji pików: Wariant dopasowany vs uproszczony', fontsize=12)
#     plt.xlabel('Błąd wariantu uproszczonego [dxy_simpl]', fontsize=10)
#     plt.ylabel('Błąd wariantu dopasowanego [dxy_new]', fontsize=10)
    
#     # Dodanie legendy i siatki
#     plt.legend()
#     plt.grid(True, linestyle=':', alpha=0.6)
    
#     # Opcjonalnie: wypełnienie obszaru pod linią (obszar sukcesu Twojego algorytmu)
#     plt.fill_between([0, max_val], [0, max_val], color='green', alpha=0.05, label='Przewaga wariantu dopasowanego')
    
#     plt.tight_layout()
#     plt.show()

# # Wywołanie
# plot_error_comparison(merged_global)


# =====================================================


# for cls in classes:
#     plot_files_in_class(results_combined_new_s_pp, cls)

classes = ["Class1", "Class2", "Class3"]
for class_id in classes:
    # plot_upset_classic_postproc(results_combined_new_s, class_id)
    plot_upset_classic_postproc_new(results_combined_new_s_pp, class_id)
    plt.savefig(
        os.path.join(out_dir, f"upset_{class_id}_post_simpl.pdf"),
        format="pdf",
        bbox_inches="tight"
    )
    
plot_peak_detection_pie_new(results_combined_new_s_pp, "Class4")
plt.savefig(
    os.path.join(out_dir, "pie_Class4_P2_post_simpl.pdf"),
    format="pdf",
    bbox_inches="tight"
)

# # wariant a
# df_metrics_a_pre = pd.concat([compute_metrics_pre_postproc(results_a_it2, cls) for cls in classes], ignore_index=True)
# df_metrics_a_post = pd.concat([compute_metrics_postproc(results_combined_a_it2_pp, cls) for cls in classes], ignore_index=True)

# # wariant b
# df_metrics_b_pre = pd.concat([compute_metrics_pre_postproc(results_b_it2, cls) for cls in classes], ignore_index=True)
# df_metrics_b_post = pd.concat([compute_metrics_postproc(results_combined_b_it2_pp, cls) for cls in classes], ignore_index=True)

# df_metrics_a_pre.to_csv("metrics_a_pre.csv", index=False)
# df_metrics_a_post.to_csv("metrics_a_post.csv", index=False)
# df_metrics_b_pre.to_csv("metrics_b_pre.csv", index=False)
# df_metrics_b_post.to_csv("metrics_b_post.csv", index=False)

# plot_files_in_class(results_combined_a_it2_pp, "Class1")
# plot_files_in_class(results_combined_a_it2_pp, "Class2")
# plot_files_in_class(results_combined_a_it2_pp, "Class3")
# plot_files_in_class(results_combined_a_it2_pp, "Class4")

# plot_files_in_class(results_combined_b_it2_pp, "Class1")
# plot_files_in_class(results_combined_b_it2_pp, "Class2")
# plot_files_in_class(results_combined_b_it2_pp, "Class3")
# plot_files_in_class(results_combined_b_it2_pp, "Class4")

# classes = ["Class1", "Class2", "Class3",]
# for class_id in classes:
#     plot_upset_classic_postproc(results_combined_a_it2_pp, class_id)
#     plot_upset_classic_postproc(results_combined_b_it2_pp, class_id)

# df_pogladowe_a_it2_pp['Class'] = df_pogladowe_a_it2_pp['file'].str.split('_').str[0]


# plot_signal_pre_post(
#     results_combined_new,
#     results_combined_new_pp,
#     file_name="Class2_it2_example_0048"
# )

# plt.savefig("rysunki/pp_example_1.pdf", 
#             bbox_inches=None)

# plot_signal_pre_post(
#     results_combined_new,
#     results_combined_new_pp,
#     file_name="Class1_it2_example_0202"
# )

# plt.savefig("rysunki/pp_example_2.pdf", 
#             bbox_inches=None)


# # plot_signal_pre_post(
# #     results_combined_new,
# #     results_combined_new_pp,
# #     file_name="Class2_it2_example_0150"
# # )

# # plot_signal_pre_post(
# #     results_combined_new,
# #     results_combined_new_pp,
# #     file_name="Class3_it2_example_0162"
# # )

# plot_signal_pre_post(
#     results_combined_new,
#     results_combined_new_pp,
#     file_name="Class3_it2_example_0020"
# )

# plt.savefig("rysunki/pp_example_3.pdf", 
#             bbox_inches=None)


# plot_signal_with_concave_areas(it1, "Class1_example_0028")



# mask = (
#     (df_pogladowe_a_it2_pp["Class"] == "Class3") &
#     df_pogladowe_a_it2_pp["peaks_detected"].apply(
#         lambda d:
#             not (isinstance(d.get("P1"), float) and math.isnan(d.get("P1"))) and
#             not (isinstance(d.get("P2"), float) and math.isnan(d.get("P2"))) and
#             not (isinstance(d.get("P3"), float) and math.isnan(d.get("P3")))
#     )
# )

# df_all_peaks_class3 = df_pogladowe_a_it2_pp[mask]

# print(df_all_peaks_class3[["file", "peaks_detected"]])
# print(f"\nLiczba sygnałów Class3 z wykrytymi P1, P2 i P3: {len(df_all_peaks_class3)}")

# mask_complement = (
#     (df_pogladowe_a_it2_pp["Class"] == "Class3") &
#     df_pogladowe_a_it2_pp["peaks_detected"].apply(
#         lambda d:
#             (isinstance(d.get("P1"), float) and math.isnan(d.get("P1"))) or
#             (isinstance(d.get("P2"), float) and math.isnan(d.get("P2"))) or
#             (isinstance(d.get("P3"), float) and math.isnan(d.get("P3")))
#     )
# )

# df_not_all_peaks_class3 = df_pogladowe_a_it2_pp[mask_complement]

# print(df_not_all_peaks_class3[["file", "peaks_detected"]])
# print(f"\nLiczba sygnałów Class3 BEZ kompletu P1–P2–P3: {len(df_not_all_peaks_class3)}")
# mask = (df_pogladowe_a_it2_pp["Class"] == "Class3") & df_pogladowe_a_it2_pp["peaks_detected"].apply(
#     lambda d: isinstance(d.get("P1"), float) and math.isnan(d.get("P1"))
# )

# df_missing_p1 = df_pogladowe_a_it2_pp[mask]

# # Wyświetlamy
# print(df_missing_p1)

# plot_peak_detection_pie(results_combined_a_it2_pp, "Class4", peak="P2")
# plot_peak_detection_pie(results_combined_b_it2_pp, "Class4", peak="P2")


# mask = df_pogladowe_a_pp["peaks_detected"].apply(
#     lambda d: any(len(d[p]) > 1 for p in ["P2"])
# )
# rows_multi_a = df_pogladowe_a_pp[mask]
# print(rows_multi_a)
# mask = df_pogladowe_a_pp["peaks_detected"].apply(
#     lambda d: any(len(d[p]) > 1 for p in ["P1", "P2", "P3"])
# )
# rows_multi_a = df_pogladowe_a_pp[mask]
# print(rows_multi_a)

# count_a = df_pogladowe_a["peaks_detected"].apply(
#     lambda d: len(d["P1"]) == 0 or len(d["P2"]) == 0 or len(d["P3"]) == 0
# ).sum()

# empty_lists_count_a = df_pogladowe_a["peaks_detected"].apply(
#     lambda d: (len(d["P1"]) == 0) + (len(d["P2"]) == 0) + (len(d["P3"]) == 0)
# ).sum()

# print(count_a, empty_lists_count_a)

