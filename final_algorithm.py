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




def wilcoxon_safe(x, y):
    # jeśli wszystkie różnice są zerowe, p-value = 1.0
    if np.all(x == y):
        return 0.0, 1.0
    else:
        try:
            stat, p = wilcoxon(x, y, alternative="two-sided")
            return stat, p
        except ValueError:
            return np.nan, np.nan


def build_error_dataframe(results_postproc):
    """
    Zwraca DataFrame:
    Filename | Class | Peak | Mean_XY_Error
    Jedna obserwacja = (plik, pik)
    """

    rows = []

    for item in results_postproc:
        file = item["file"]
        cls = item["class"]
        sig = item["signal_raw"]
        t = sig.iloc[:, 0].values
        y = sig.iloc[:, 1].values

        for peak, det in item["peaks_detected"].items():
            ref = item["peaks_ref"].get(peak)

            if ref is None or det is None or (isinstance(det, float) and np.isnan(det)):
                continue

            dx = abs(t[det] - t[ref])
            dy = abs(y[det] - y[ref])
            dxy = np.sqrt(dx**2 + dy**2)

            rows.append({
                "Filename": file,
                "Class": cls,
                "Peak": peak,
                "Mean_XY_Error": dxy
            })

    return pd.DataFrame(rows)



def prepare_paired_dataset(
    df_fitted,
    df_simplified,
    value_col="Mean_XY_Error",
    id_cols=("Filename", "Class", "Peak")
):
    """
    Łączy wyniki dwóch wersji algorytmu w pary (plik, pik),
    automatycznie usuwając obserwacje niekompletne.
    """

    df_fitted = df_fitted[list(id_cols) + [value_col]].copy()
    df_simplified = df_simplified[list(id_cols) + [value_col]].copy()

    df_fitted = df_fitted.rename(columns={value_col: "Error_fitted"})
    df_simplified = df_simplified.rename(columns={value_col: "Error_simplified"})

    df_merged = pd.merge(
        df_fitted,
        df_simplified,
        on=list(id_cols),
        how="inner"
    )

    return df_merged

def wilcoxon_option_A(df_pairs):
    """
    Wykonuje test Wilcoxona na wszystkich parach (plik, pik)
    bez rozdzielania na klasy.
    """

    errors_fitted = df_pairs["Error_fitted"]
    errors_simplified = df_pairs["Error_simplified"]
    
    diff_all = df_pairs["Error_fitted"] - df_pairs["Error_simplified"]


    stat, p_value = wilcoxon_safe(
        errors_fitted,
        errors_simplified,
        # alternative="two-sided"
    )

    stats = {
        "N_total_pairs": len(df_pairs),
        
        # "Mean_fitted": np.mean(errors_fitted),
        "Median_fitted": np.median(errors_fitted),
        "Q1_fitted": np.percentile(errors_fitted, 25),
        "Q3_fitted": np.percentile(errors_fitted, 75),
        
        # "Mean_simplified": np.mean(errors_simplified),
        "Median_simplified": np.median(errors_simplified),
        "Q1_simplified": np.percentile(errors_simplified, 25),
        "Q3_simplified": np.percentile(errors_simplified, 75),
        # "Wilcoxon_stat": stat,
        
        "Median_diff": np.median(diff_all),
        "p_value": p_value
    }

    return stats

def wilcoxon_by_class_and_peak(df_pairs):
    """
    Wilcoxon osobno dla każdej klasy i piku.
    """

    results = []

    for (cls, peak), group in df_pairs.groupby(["Class", "Peak"]):
        if len(group) < 5:
            continue  # zbyt mało par – brak sensu statystycznego
        
        diff = group["Error_fitted"] - group["Error_simplified"]
        
        stat, p = wilcoxon_safe(
            group["Error_fitted"],
            group["Error_simplified"],
            # alternative="two-sided"
        )

        results.append({
            "Class": cls,
            "Peak": peak,
            "N_used": len(group),
            # "Mean_fitted": np.mean(group["Error_fitted"]),
            "Median_fitted": np.median(group["Error_fitted"]),
            "Q1_fitted": np.percentile(group["Error_fitted"], 25),
            "Q3_fitted": np.percentile(group["Error_fitted"], 75),
            # "Mean_simplified": np.mean(group["Error_simplified"]),
            "Median_simplified": np.median(group["Error_simplified"]),
            "Q1_simplified": np.percentile(group["Error_simplified"], 25),
            "Q3_simplified": np.percentile(group["Error_simplified"], 75),
            
            "Median_diff": np.median(diff),
            "p_value": p
        })

    return pd.DataFrame(results)

def add_summary_row(df_by_class, df_pairs):
    """
    Dodaje ostatni wiersz sumaryczny do tabeli df_by_class,
    z uwzględnieniem średniego błędu.
    """
    # Wilcoxon na wszystkich parach
    summary_stats = wilcoxon_option_A(df_pairs)
    
    # Obliczamy średnie błędy po wszystkich parach
    # mean_fitted = df_pairs["Error_fitted"].mean()
    # mean_simplified = df_pairs["Error_simplified"].mean()
    
    summary_row = {
        "Class": "ALL",
        "Peak": "ALL",
        "N_used": summary_stats["N_total_pairs"],
        # "Mean_fitted": mean_fitted,
        "Median_fitted": summary_stats["Median_fitted"],
        "Q1_fitted": summary_stats["Q1_fitted"],
        "Q3_fitted": summary_stats["Q3_fitted"],
        # "Mean_simplified": mean_simplified,
        "Median_simplified": summary_stats["Median_simplified"],
        "Q1_simplified": summary_stats["Q1_simplified"],
        "Q3_simplified": summary_stats["Q3_simplified"],
        "Median_diff": summary_stats["Median_diff"],
        "p_value": summary_stats["p_value"]
    }
    
    return pd.concat([df_by_class, pd.DataFrame([summary_row])], ignore_index=True)



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


# mean_dxy = df_metrics_post["Mean_XY_Error"].mean()

# mean_dx = df_metrics_post['Mean_X_Error'].mean()

# mean_dy = df_metrics_post['Mean_Y_Error'].mean()

# # Sumowanie TP, FP, FN po wszystkich wierszach
# TP_total = df_metrics_post['TP'].sum()
# FP_total = df_metrics_post['FP'].sum()
# FN_total = df_metrics_post['FN'].sum()

# # Czułość (sensitivity / recall)
# sensitivity = 100 * TP_total / (TP_total + FN_total) if (TP_total + FN_total) > 0 else np.nan

# # Precyzja (precision)
# precision = 100 * TP_total / (TP_total + FP_total) if (TP_total + FP_total) > 0 else np.nan

# print(f"Średni błąd: {mean_dxy:.3f}")
# print(f"Średni błąd: {mean_dx:.3f}")
# print(f"Średni błąd: {mean_dy:.3f}")
# print(f"Czułość (Sensitivity %): {sensitivity:.2f}%")
# print(f"Precyzja (Precision %): {precision:.2f}%")
"""
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
#     # ncols=2
# )
# plt.savefig("rysunki/final_detection_examples_k1.pdf", 
#             bbox_inches=None)

# plot_selected_files(
#     detection_results=results_combined_new_pp,
#     class_name="Class2",
#     files_to_plot=["Class2_it2_example_0022", 
#                    "Class2_it2_example_0002",
#                    "Class2_it2_example_0058"],
#     # ncols=3
# )
# plt.savefig("rysunki/final_detection_examples_k2.pdf", 
#             bbox_inches=None)

# plot_selected_files(
#     detection_results=results_combined_new_pp,
#     class_name="Class3",
#     files_to_plot=["Class3_it2_example_0004", 
#                    "Class3_it2_example_0105",
#                    "Class3_it2_example_0047"],
#     # ncols=3
# )
# plt.savefig("rysunki/final_detection_examples_k3.pdf", 
#             bbox_inches=None)


# plot_selected_files(
#     detection_results=results_combined_new_pp,
#     class_name="Class4",
#     files_to_plot=["Class4_it2_example_0033", 
#                    "Class4_it2_example_0024",
#                    "Class4_it2_example_0019"],
    
#     # ncols=3
# )
# plt.savefig("rysunki/final_detection_examples_k4.pdf", 
#             bbox_inches=None)

"""
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

"""
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

# df_fit_pre  = prepare_metrics_df(
#     df_metrics_pre,
#     "dopasowany – przed postprocessingiem"
# )

# df_fit_post = prepare_metrics_df(
#     df_metrics_post,
#     "dopasowany – po postprocessingu"
# )

# df_s_pre = prepare_metrics_df(
#     df_metrics_pre_s,
#     "uproszczony – przed postprocessingiem"
# )

# df_s_post = prepare_metrics_df(
#     df_metrics_post_s,
#     "uproszczony – po postprocessingu"
# )

# df_metrics_all = pd.concat(
#     [df_fit_pre, df_fit_post, df_s_pre, df_s_post],
#     ignore_index=True
# )

# df_metrics_all = df_metrics_all[
#     ["Class", "Peak", "Version",
#      "Mean_X_Error", "Mean_Y_Error", "Mean_XY_Error",
#      "TP", "FP", "FN"]
# ]

# version_order = [
#     "dopasowany – przed postprocessingiem",
#     "dopasowany – po postprocessingu",
#     "uproszczony – przed postprocessingiem",
#     "uproszczony – po postprocessingu"
# ]

# df_metrics_all["Version"] = pd.Categorical(
#     df_metrics_all["Version"],
#     categories=version_order,
#     ordered=True
# )


# df_metrics_all = df_metrics_all.sort_values(
#     ["Class", "Peak", "Version"]
# ).reset_index(drop=True)

# df_metrics_all.to_csv("NEW_metrics_4_01.csv", index=False)



# for cls in classes:
#     plot_files_in_class(results_combined_new_s_pp, cls)
"""
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
"""

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


df_fitted = build_error_dataframe(results_combined_new_pp)
df_simplified = build_error_dataframe(results_combined_new_s_pp)

df_pairs = prepare_paired_dataset(
    df_fitted,
    df_simplified,
    value_col="Mean_XY_Error",
    id_cols=("Filename", "Class", "Peak")
)


# 2. OPCJA A – jeden test, jeden wiersz
summary_A = wilcoxon_option_A(df_pairs)

# 3. (opcjonalnie) pełna tabela 4.9
table_49 = wilcoxon_by_class_and_peak(df_pairs)

table_49 = add_summary_row(table_49, df_pairs)
