# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 02:27:15 2025

Ostateczny algorytm
@author: Hanna Jaworska
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_handling import load_dataset, smooth_dataset
from methods import (concave, curvature, modified_scholkmann, 
                     line_distance, hilbert_envelope, wavelet)
from ranges import (ranges_full, ranges_pm3, ranges_whiskers, 
                    generate_ranges_for_all_files, compute_crossings)

# z czatu - do sprawdzenia
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

    
    results = []

    for item in dataset:
        if item["class"] != class_id:
            continue  # tylko wskazana klasa
        
        file = item["file"]
        sig = item["signal"]
        y = sig.iloc[:, 1].values
        t = sig.iloc[:, 0].values

        raw_peaks = np.array(method_name(y), dtype=int)

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
            "file": file,
            "signal": sig,
            "peaks_ref": item.get("peaks_ref", {}),
            "peak_detected": {peak: detected_peaks}
        })

    return results

"""
Class1 P1:
    - avg, concave (min blad)
    - full, concave (max. signals w/ peaks)

Class1 P2:
    - avg, curvature (max signals w/ peaks)
    - smooth 4Hz, avg, curvature (min blad)

Class1 P3:
    - smooth 4Hz, full, concave
    

Class2 P1:
    - smooth 3Hz, avg, hilbert (min blad, 249)
    - smooth 3Hz, avg, wavelet (podobny, 250) SAME_AVG!!!!!!!!!

Class2 P2:
    - whiskers, wavelet (min blad)
    - smooth 4Hz, full, curvature (max sig w/ peaks)

Class2 P3:
    - avg, curvature (min blad)
    - whiskers, hilbert (doslownie o 2 wiecej sig w/ peaks)

Class3 P1:
    - smooth 4Hz, full, wavelet LUB smooth 4Hz, whiskers, wavelet

Class3 P2:
    - whiskers, concave (min blad)
    - smooth 4Hz, full, concave (max sig)

Class3 P3:
    - none, modified scholkmann 1/2 99 LUB none, modified scholkmann 1 99

Class4 P2:
    - smooth 4Hz, none, modified scholkmann 1/2 99 LUB smooth 4Hz, none, modified scholkmann 1 99
"""

base_path_2 = r"ICP_pulses_it2"
it2 = load_dataset(base_path_2, "it2")
it2_smooth_4Hz = smooth_dataset(it2, cutoff=4, inplace=False)
it2_smooth_3Hz = smooth_dataset(it2, cutoff=3, inplace=False)