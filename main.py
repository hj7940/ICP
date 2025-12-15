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
from moving_average import compute_crossings, smooth_dataset
from data_handling import load_dataset, smooth_dataset
from methods import concave, curvature, modified_scholkmann, line_distance, hilbert_envelope, wavelet
from ranges import ranges_full, ranges_pm3, ranges_whiskers, generate_ranges_for_all_files


# sprawdzone i dzialajace: import zakresow z ranges, load_dataset i smooth_dataset
# generowanie zakresow zgodnych w strukturze z crossingami
# peak detection

# zgrubna - po calym dataset zeby nie mnozyc petli
def peak_detection(dataset, method_name, ranges=None, 
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
    
    if ranges is not None:
        time_ranges, amp_ranges = ranges
    else:
        time_ranges = None 
        amp_ranges = None
        
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
                t_start, t_end = time_ranges[class_id][file][p]
        
            if amp_ranges is None:
                a_start, a_end = 0, 1
            else:
                a_start, a_end = amp_ranges[class_id][file][p]
            
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




# %% 
all_methods = {
    "concave": lambda sig: concave(sig, 0, 0, None),
    "concave_d2x=-0,002": lambda sig:  concave(sig, -0.002, 0, None),
    "concave_d2x=0,002": lambda sig:  concave(sig, 0.002, 0, None),
    
    "modified_scholkmann_1_99": lambda sig: modified_scholkmann(sig, 1, 99),
    "modified_scholkmann_1_95": lambda sig: modified_scholkmann(sig, 1, 95),
    "modified_scholkmann_1-2_95": lambda sig: modified_scholkmann(sig, 2, 95),
    "modified_scholkmann_1-2_99": lambda sig: modified_scholkmann(sig, 2, 99),
    
    "curvature": lambda sig: curvature(sig, 0, 0, None),
    "line_distance_10": lambda sig: line_distance(sig, 0,"vertical", 10),
    "hilbert": lambda sig: hilbert_envelope(sig, 0),
    "wavelet": lambda sig: wavelet(sig, (1,10))
}

base_path = r"ICP_pulses_it1"
it1 = load_dataset(base_path, "it1")
# data_raw_it1 = data_it1
it1_smooth_4Hz = smooth_dataset(it1, cutoff=4, inplace=False)
it1_smooth_3Hz = smooth_dataset(it1, cutoff=3, inplace=False)

base_path_2 = r"ICP_pulses_it2"
it2 = load_dataset(base_path_2, "it2")
# data_raw_it2 = data_it2
it2_smooth_4Hz = smooth_dataset(it2, cutoff=4, inplace=False)
it2_smooth_3Hz = smooth_dataset(it1, cutoff=3, inplace=False)


# ujednolicenie struktury zakresow stalych oraz wyznaczonych 
# za pomoca sredniej ruchomej
for dataset_type, dataset_name in ((it1, "it1"), (it2, "it2")):
    for range_type, range_name in ((ranges_full, "ranges_full"), 
                                   (ranges_pm3, "ranges_pm3"), 
                                   (ranges_whiskers, "ranges_whiskers")):
        vars()[f"{range_name}_{dataset_name}"] = generate_ranges_for_all_files(dataset_type, range_type)

all_results = []
for m in all_methods:
    test = peak_detection(it1, m, ranges_full_it1, "full")
    test2 = peak_detection(it1, m, ranges_whiskers_it1, "whiskers")
    test3 = peak_detection(it1, m, None, "none")
    all_results.append(test)
    all_results.append(test2)
    all_results.append(test3)
    
    
print("jejj jupii")
    