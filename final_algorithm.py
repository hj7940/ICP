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

import math
from main import (all_methods, it2, it2_smooth_4Hz, it2_smooth_3Hz,
                  it1, it1_smooth_4Hz, it1_smooth_3Hz,
                  ranges_all_time, ranges_all_amps, compute_peak_metrics)
from all_plots import plot_upset_for_class, plot_files_in_class
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



def postprocess_peaks(results_combined):
    """
    Postprocess peaks_detected zgodnie z regułami DALSZE DZIAŁANIA.
    """

    results_post=results_combined

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

        # =========================
        # -------- Class1 ---------
        # =========================
        if class_id == "Class1":
            # P3: jeśli dwa → zostaw późniejszy
            peaks["P3"] = last_peak(peaks["P3"])

            # P1: jeśli kilka → średnia
            peaks["P1"] = mean_peak(peaks["P1"])

            # P2: jesli kilka - pierwszy
            peaks["P2"] = first_peak(peaks["P2"])

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
            # P3: średnia
            peaks["P3"] = mean_peak(peaks["P3"])

            # P2: pierwszy
            peaks["P2"] = first_peak(peaks["P2"])

            # P1: pierwszy
            peaks["P1"] = first_peak(peaks["P1"])
            
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
            # P3: średnia
            peaks["P3"] = mean_peak(peaks["P3"])

            # P1: średnia
            peaks["P1"] = mean_peak(peaks["P1"])

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
            peaks["P2"] = mean_peak(peaks["P2"])
            peaks["P1"] = []
            peaks["P3"] = []
            
        item["peaks_detected"] = _finalize_peaks(peaks)

    return results_post


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



    

def has_peak(v):
    """Zwraca True, jeśli pik jest wykryty (nie NaN)."""
    return v is not None and not math.isnan(v)



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
    a,b) smooth 4Hz, full, wavelet LUB smooth 4Hz, whiskers, wavelet
    
Class3 P2:
    a) whiskers, concave (min blad)
    b) smooth 4Hz, full, concave (max sig)
    ----------- ZMIANA ------------
    a) smooth 4Hz, avg, hilbert
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

# datasets = [
#     (it2, "it2"),
#     (it2_smooth_4Hz, "it2_smooth_4Hz"),
#     (it2_smooth_3Hz, "it2_smooth_3Hz"),
# ]

#c1_p1=peak_detection_single(it2, "concave", "Class1", "P1", df_ranges_time.loc["it2", "avg"], None, "avg")
 # -------------------- lekka krakasa, moznaby rzec ----------------
# --- wariant a ---
# df_variant_a = pd.DataFrame([
#     # -------- Class1 --------
#     ("Class1", "P1", "it2",            "it2",            "avg",  "concave"),
#     ("Class1", "P2", "it2_smooth_4Hz", "it2_smooth_4Hz", "avg",  "curvature"),
#     ("Class1", "P3", "it2_smooth_4Hz", "it2_smooth_4Hz", "full", "concave"),

#     # -------- Class2 --------
#     ("Class2", "P1", "it2_smooth_3Hz", "it2_smooth_3Hz", "avg",  "hilbert"),
#     ("Class2", "P2", "it2",            "it2",            "whiskers", "wavelet"),
#     ("Class2", "P3", "it2",            "it2",            "avg",      "curvature"),

#     # -------- Class3 --------
#     ("Class3", "P1", "it2_smooth_4Hz", "it2_smooth_4Hz", "full", "wavelet"),
#     ("Class3", "P2", "it2",            "it2",            "whiskers", "concave"),
#     ("Class3", "P3", "it2",            "it2",            "none", "modified_scholkmann_1-2_99"),

#     # -------- Class4 --------
#     ("Class4", "P2", "it2_smooth_4Hz", "it2_smooth_4Hz", "none", "modified_scholkmann_1-2_99"),
# ],
# columns=[
#     "class",
#     "peak",
#     "detect_dataset",
#     "ranges_dataset",
#     "range_type",
#     "method"
# ])

# # --- wariant b ---
# df_variant_b = pd.DataFrame([
#     # -------- Class1 --------
#     ("Class1", "P1", "it2",            "it2",            "full", "concave"),
#     ("Class1", "P2", "it2",            "it2",            "avg",  "curvature"),
#     ("Class1", "P3", "it2_smooth_4Hz", "it2_smooth_4Hz", "full", "concave"),

#     # -------- Class2 --------
#     ("Class2", "P1", "it2_smooth_3Hz", "it2",            "avg",  "wavelet"),
#     ("Class2", "P2", "it2_smooth_4Hz", "it2_smooth_4Hz", "full", "curvature"),
#     ("Class2", "P3", "it2",            "it2",            "whiskers", "hilbert"),

#     # -------- Class3 --------
#     ("Class3", "P1", "it2_smooth_4Hz", "it2_smooth_4Hz", "full", "wavelet"),
#     ("Class3", "P2", "it2_smooth_4Hz", "it2_smooth_4Hz", "full",     "concave"),
#     ("Class3", "P3", "it2",            "it2",            "none", "modified_scholkmann_1-2_99"),

#     # -------- Class4 --------
#     ("Class4", "P2", "it2_smooth_4Hz", "it2_smooth_4Hz", "none", "modified_scholkmann_1-2_99"),
# ],
# columns=[
#     "class",
#     "peak",
#     "detect_dataset",
#     "ranges_dataset",
#     "range_type",
#     "method"
# ])

# --- wariant a ---
df_variant_a = pd.DataFrame([
    # -------- Class1 --------
    ("Class1", "P1", "it2",            "it2",            "avg",  "concave"),
    ("Class1", "P2", "it2_smooth_4Hz", "it2_smooth_4Hz", "avg",  "curvature"),
    ("Class1", "P3", "it2_smooth_4Hz", "it2_smooth_4Hz", "full", "concave"),

    # -------- Class2 --------
    ("Class2", "P1", "it2_smooth_3Hz", "it2_smooth_3Hz", "avg",  "hilbert"),
    ("Class2", "P2", "it2_smooth_4Hz", "it2_smooth_4Hz", "whiskers", "line_distance_10"),
    ("Class2", "P3", "it2",            "it2",            "full",      "hilbert"),

    # -------- Class3 --------
    ("Class3", "P1", "it2_smooth_4Hz", "it2_smooth_4Hz", "full", "wavelet"),
    ("Class3", "P2", "it2_smooth_4Hz", "it2_smooth_4Hz", "avg", "hilbert"),
    ("Class3", "P3", "it2",            "it2",            "full", "modified_scholkmann_1-2_99"),

    # -------- Class4 --------
    ("Class4", "P2", "it2", "it2", "full", "concave_d2x=0-002"),
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
    ("Class1", "P2", "it2_smooth_4Hz", "it2_smooth_4Hz", "avg",  "curvature"),
    ("Class1", "P3", "it2_smooth_4Hz", "it2_smooth_4Hz", "full", "concave"),

    # -------- Class2 --------
    ("Class2", "P1", "it2_smooth_3Hz", "it2_smooth_3Hz", "avg",  "hilbert"),
    ("Class2", "P2", "it2_smooth_4Hz", "it2_smooth_4Hz", "full", "line_distance_10"),
    ("Class2", "P3", "it2",            "it2",            "pm3", "hilbert"),

    # -------- Class3 --------
    ("Class3", "P1", "it2_smooth_4Hz", "it2_smooth_4Hz", "full", "wavelet"),
    ("Class3", "P2", "it2_smooth_4Hz", "it2_smooth_4Hz", "avg",  "wavelet"),
    ("Class3", "P3", "it2",            "it2",            "none", "modified_scholkmann_1-2_99"),

    # -------- Class4 --------
    ("Class4", "P2", "it2", "it2", "pm3", "concave_d2x=0-002"),
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
    # "it1": it1,
    # "it1_smooth_4Hz": it1_smooth_4Hz,
    # "it1_smooth_3Hz": it1_smooth_3Hz,
    "it2": it2,
    "it2_smooth_4Hz": it2_smooth_4Hz,
    "it2_smooth_3Hz": it2_smooth_3Hz,
}

#%% A IT1
# results_a = run_variant(
#     df_variant_a,
#     datasets_dict=datasets_dict,
#     ranges_all_time=ranges_all_time,
#     ranges_all_amps=ranges_all_amps
# )


# results_combined_a = combine_peaks_by_file(results_a)
# # results_combined_a_pp = postprocess_peaks(results_combined_a)


# df_pogladowe_a_pp = pd.DataFrame([
#     {
#         "file": d["file"],
#         "peaks_ref": d["peaks_ref"],
#         "peaks_detected": d["peaks_detected"],
#     }
#     for d in results_combined_a
# ])

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


# plot_files_in_class(results_combined, "Class1")
# plot_files_in_class(results_combined, "Class2")
# plot_files_in_class(results_combined, "Class3")
# plot_files_in_class(results_combined, "Class4")

# %% B
# results_b = run_variant(
#     df_variant_a,
#     datasets_dict=datasets_dict,
#     ranges_all_time=ranges_all_time,
#     ranges_all_amps=ranges_all_amps
# )


# results_combined_b = combine_peaks_by_file(results_b)
# # results_combined_b_pp = postprocess_peaks(results_combined_b)

# df_pogladowe_b_pp = pd.DataFrame([
#     {
#         "file": d["file"],
#         "peaks_ref": d["peaks_ref"],
#         "peaks_detected": d["peaks_detected"],
#     }
#     for d in results_combined_b
# ])

# mask = df_pogladowe_b_pp["peaks_detected"].apply(
#     lambda d: any(len(d[p]) > 1 for p in ["P1", "P2", "P3"])
# )

# rows_multi_b = df_pogladowe_b_pp[mask]
# print(rows_multi_b)

# count_b = df_pogladowe_b["peaks_detected"].apply(
#     lambda d: len(d["P1"]) == 0 or len(d["P2"]) == 0 or len(d["P3"]) == 0
# ).sum()

# empty_lists_count_b = df_pogladowe_b["peaks_detected"].apply(
#     lambda d: (len(d["P1"]) == 0) + (len(d["P2"]) == 0) + (len(d["P3"]) == 0)
# ).sum()

# print(count_b, empty_lists_count_b)

# plot_files_in_class(results_combined_b, "Class1")
# plot_files_in_class(results_combined_b, "Class2")
# plot_files_in_class(results_combined_b, "Class3")
# plot_files_in_class(results_combined_b, "Class4")

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

# %% ================= DRUGI ZESTAW (IT2) ====================
results_a_it2 = run_variant(
    df_variant_a,
    datasets_dict=datasets_dict,
    ranges_all_time=ranges_all_time,
    ranges_all_amps=ranges_all_amps
)


results_combined_a_it2 = combine_peaks_by_file(results_a_it2)
results_combined_a_it2_pp = postprocess_peaks(results_combined_a_it2)


df_pogladowe_a_it2_pp = pd.DataFrame([
    {
        "file": d["file"],
        "peaks_ref": d["peaks_ref"],
        "peaks_detected": d["peaks_detected"],
    }
    for d in results_combined_a_it2_pp
])

classes = ["Class1", "Class2", "Class3",]
for cls in classes:
    plot_upset_for_class(results_combined_a_it2_pp, cls)

# plot_files_in_class(results_combined_a_it2_pp, "Class1")
# plot_files_in_class(results_combined_a_it2_pp, "Class2")
# plot_files_in_class(results_combined_a_it2_pp, "Class3")
# plot_files_in_class(results_combined_a_it2_pp, "Class4")

results_b_it2 = run_variant(
    df_variant_b,
    datasets_dict=datasets_dict,
    ranges_all_time=ranges_all_time,
    ranges_all_amps=ranges_all_amps
)


results_combined_b_it2 = combine_peaks_by_file(results_b_it2)
results_combined_b_it2_pp = postprocess_peaks(results_combined_b_it2)


df_pogladowe_b_it2_pp = pd.DataFrame([
    {
        "file": d["file"],
        "peaks_ref": d["peaks_ref"],
        "peaks_detected": d["peaks_detected"],
    }
    for d in results_combined_b_it2_pp
])




# classes = ["Class1", "Class2", "Class3",]
# for cls in classes:
#     plot_upset_for_class(results_combined_b_it2_pp, cls)


# plot_files_in_class(results_combined_b_it2_pp, "Class1")
# plot_files_in_class(results_combined_b_it2_pp, "Class2")
# plot_files_in_class(results_combined_b_it2_pp, "Class3")
# plot_files_in_class(results_combined_b_it2_pp, "Class4")

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


# plot_files_in_class(results_combined, "Class1")
# plot_files_in_class(results_combined, "Class2")
# plot_files_in_class(results_combined, "Class3")
# plot_files_in_class(results_combined, "Class4")