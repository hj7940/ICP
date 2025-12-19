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
import copy
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
                  ranges_all_time, ranges_all_amps)
from all_plots import (plot_files_in_class, 
                       plot_upset_classic_postproc, plot_signal_pre_post,
                       plot_signal_with_concave_areas)

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


def plot_peak_detection_pie(results_combined, class_id, peak="P2"):
    """
    Rysuje wykres kołowy pokazujący ile sygnałów w danej klasie
    ma wykryty dany pik, a ile nie.

    results_combined : lista słowników po postprocess_peaks
    class_id : str, np. "Class4"
    peak : str, nazwa piku, np. "P1", "P2", "P3"
    """
    # filtrujemy tylko daną klasę
    class_results = [d for d in results_combined if d["class"] == class_id]

    # liczba sygnałów z wykrytym pikiem
    has_peak_count = sum(1 for d in class_results if d.get("peaks_detected") and not math.isnan(d["peaks_detected"].get(peak, float('nan'))))
    no_peak_count = len(class_results) - has_peak_count

    # dane do wykresu
    labels = [f"{peak} wykryty", f"{peak} nie wykryty"]
    sizes = [has_peak_count, no_peak_count]
    colors = ["#66b3ff", "#ff9999"]

    plt.figure(figsize=(6,6))
    plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    plt.title(f"Klasa {class_id[-1]}: wykrycie {peak} w sygnałach")
    plt.show()
    
    
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
    class_signals = [d for d in dataset if d["class"] == class_name]
    n_files = 250 if class_name in ["Class1","Class2","Class3"] else 50
    
    for peak in expected_peaks:
        dx_all, dy_all, dxy_all, peak_count_sum = [], [], [], 0

        for item in class_signals:
            sig = item["signal"]
            t = sig.iloc[:,0].values
            y = sig.iloc[:,1].values

            ref_idx = item["peaks_ref"].get(peak)
            detected = item["peaks_detected"].get(peak, [])

            if ref_idx is None or len(detected) == 0:
                continue

            # dopasowanie wielu wykrytych do referencji
            if np.isscalar(ref_idx):
                dx = np.abs(t[detected] - t[ref_idx])
                dy = np.abs(y[detected] - y[ref_idx])
            else:
                dx = np.sum([np.abs(t[detected] - t[r]) for r in ref_idx], axis=0)
                dy = np.sum([np.abs(y[detected] - y[r]) for r in ref_idx], axis=0)

            dxy = np.sqrt(dx**2 + dy**2)

            # agregacja po pliku: średnia jeśli wykryto kilka
            dx_all.append(np.mean(dx))
            dy_all.append(np.mean(dy))
            dxy_all.append(np.mean(dxy))
            
            peak_count_sum += len(detected)

        metrics_list.append({
            "Class": class_name,
            "Peak": peak,
            "Mean_X_Error": np.mean(dx_all) if dx_all else np.nan,
            "Mean_Y_Error": np.mean(dy_all) if dy_all else np.nan,
            "Mean_XY_Error": np.mean(dxy_all) if dxy_all else np.nan,
            "Std_XY_Error": np.std(dxy_all) if dxy_all else np.nan,
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

    for peak in expected_peaks:
        dx_all, dy_all, dxy_all, peak_count_sum = [], [], [], 0

        for item in class_signals:
            sig = item["signal"]
            t = sig.iloc[:,0].values
            y = sig.iloc[:,1].values

            ref_idx = item["peaks_ref"].get(peak)
            detected_val = item["peaks_detected"].get(peak)

            # w postproc to albo int albo NaN
            if ref_idx is None or detected_val is None or (isinstance(detected_val, float) and math.isnan(detected_val)):
                continue

            if np.isscalar(ref_idx):
                dx = abs(t[detected_val] - t[ref_idx])
                dy = abs(y[detected_val] - y[ref_idx])
            else:
                dx = np.sum([abs(t[detected_val] - t[r]) for r in ref_idx])
                dy = np.sum([abs(y[detected_val] - y[r]) for r in ref_idx])

            dxy = np.sqrt(dx**2 + dy**2)
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
            "%_Files_With_Peak": len(dx_all)/n_files,
            "Peak_Count": peak_count_sum
        })

    return pd.DataFrame(metrics_list)




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


# --- wariant a ---
df_variant_a = pd.DataFrame([
    # -------- Class1 --------
    ("Class1", "P1", "it1",            "it1",            "avg",  "concave"),
    ("Class1", "P2", "it1_smooth_4Hz", "it1_smooth_4Hz", "avg",  "curvature"),
    ("Class1", "P3", "it1_smooth_4Hz", "it1_smooth_4Hz", "full", "concave"),

    # -------- Class2 --------
    ("Class2", "P1", "it1_smooth_3Hz", "it1_smooth_3Hz", "avg",  "hilbert"),
    ("Class2", "P2", "it1_smooth_4Hz", "it1_smooth_4Hz", "whiskers", "line_distance_10"),
    ("Class2", "P3", "it1",            "it1",            "full",      "hilbert"),

    # -------- Class3 --------
    ("Class3", "P1", "it1_smooth_4Hz", "it1_smooth_4Hz", "full", "wavelet"),
    ("Class3", "P2", "it1_smooth_4Hz", "it1_smooth_4Hz", "avg", "hilbert"),
    ("Class3", "P3", "it1",            "it1",            "full", "modified_scholkmann_1-2_99"),

    # -------- Class4 --------
    ("Class4", "P2", "it1", "it1", "full", "concave_d2x=0-002"),
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
    ("Class1", "P1", "it1",            "it1",            "full", "concave"),
    ("Class1", "P2", "it1_smooth_4Hz", "it1_smooth_4Hz", "avg",  "curvature"),
    ("Class1", "P3", "it1_smooth_4Hz", "it1_smooth_4Hz", "full", "concave"),

    # -------- Class2 --------
    ("Class2", "P1", "it1_smooth_3Hz", "it1_smooth_3Hz", "avg",  "hilbert"),
    ("Class2", "P2", "it1_smooth_4Hz", "it1_smooth_4Hz", "full", "line_distance_10"),
    ("Class2", "P3", "it1",            "it1",            "pm3", "hilbert"),

    # -------- Class3 --------
    ("Class3", "P1", "it1_smooth_4Hz", "it1_smooth_4Hz", "full", "wavelet"),
    ("Class3", "P2", "it1_smooth_4Hz", "it1_smooth_4Hz", "avg",  "wavelet"),
    ("Class3", "P3", "it1",            "it1",            "none", "modified_scholkmann_1-2_99"),

    # -------- Class4 --------
    ("Class4", "P2", "it1", "it1", "pm3", "concave_d2x=0-002"),
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
    "it1": it1,
    "it1_smooth_4Hz": it1_smooth_4Hz,
    "it1_smooth_3Hz": it1_smooth_3Hz,
    # "it2": it2,
    # "it2_smooth_4Hz": it2_smooth_4Hz,
    # "it2_smooth_3Hz": it2_smooth_3Hz,
}



# %% ================= DRUGI ZESTAW (IT2) ====================
results_a_it2 = run_variant(
    df_variant_a,
    datasets_dict=datasets_dict,
    ranges_all_time=ranges_all_time,
    ranges_all_amps=ranges_all_amps)

results_combined_a_it2 = combine_peaks_by_file(results_a_it2)
results_combined_a_it2_pp = postprocess_peaks(results_combined_a_it2)

df_pogladowe_a_it2_pp = pd.DataFrame([
    {
        "file": d["file"],
        "peaks_ref": d["peaks_ref"],
        "peaks_detected": d["peaks_detected"],
}
    for d in results_combined_a_it2_pp])



results_b_it2 = run_variant(
    df_variant_b,
    datasets_dict=datasets_dict,
    ranges_all_time=ranges_all_time,
    ranges_all_amps=ranges_all_amps)

results_combined_b_it2 = combine_peaks_by_file(results_b_it2)
results_combined_b_it2_pp = postprocess_peaks(results_combined_b_it2)

df_pogladowe_b_it2_pp = pd.DataFrame([
    {
        "file": d["file"],
        "peaks_ref": d["peaks_ref"],
        "peaks_detected": d["peaks_detected"],
    }
    for d in results_combined_b_it2_pp])


# classes = ["Class1", "Class2", "Class3", "Class4"]

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

df_pogladowe_a_it2_pp['Class'] = df_pogladowe_a_it2_pp['file'].str.split('_').str[0]


# plot_signal_pre_post(
#     results_combined_a_it2,
#     results_combined_a_it2_pp,
#     file_name="Class1_example_0050"
# )

plot_signal_with_concave_areas(it1, "Class1_example_0028")



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

