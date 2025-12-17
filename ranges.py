# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 22:38:24 2025

Zakresy (dicts) w ktorych funkcja ma szukac danego piku danej klasy
Wyznaczone na podstawie it1 oraz funkcja generujaca zakrwsy dla kazdego pliku 
w datasecie.
Rodzaje:
    - full (minimum - maximum, łacznie z wartosciami odstajacymi)
    - pm3 (minimum - maximum +- 3 probki)
    - whiskers (Q1-1.5*IQR - Q3+1.5*IQR)
Zakresy w osi x (numery probek): time_
Zakresy w osi y (amplituda): amps_

ranges_ - krotki (time_ , amps_ )
@author: Hanna Jaworska
"""

import numpy as np
import pandas as pd

def generate_ranges_for_all_files(dataset, ranges, peaks=("P1", "P2", "P3")):
    """
    ranges = (time_ranges, amp_ranges)
    Cel: ujednolicenie formatu
    """

    time_in, amp_in = ranges

    time_out = {}
    amp_out  = {}

    for item in dataset:
        class_id  = item["class"]
        file = item["file"]

        time_out.setdefault(class_id, {})
        amp_out.setdefault(class_id, {})

        time_out[class_id][file] = {}
        amp_out[class_id][file]  = {}

        for p in peaks:
            # TIME
            try:
                time_out[class_id][file][p] = time_in[class_id][p]
            except KeyError:
                time_out[class_id][file][p] = (np.nan, np.nan)

            # AMPLITUDE
            try:
                amp_out[class_id][file][p] = amp_in[class_id][p]
            except KeyError:
                amp_out[class_id][file][p] = (np.nan, np.nan)

    return (time_out, amp_out)


def compute_crossings(dataset, window_fast=2, window_slow=4,
                              min_distance=0, max_crossings=6):
    """
    Wyznacza pukty przeciecia sredniej ruchomej wolnej i sredniej ruchomej szybkiej
    
    Parameters
    ----------
    dataset : list
        zestaw danych
    window_fast : int 
        liczba probek wliczana do szybkiej sredniej ruchomej
    widow_slow : int
        liczba probek wliczana do wolnej sredniej ruchomej
    min_distance : int (optional)
        minimalna odleglosc od poprzedniego punktu przeciecia (domyslnie 0)
    max_crossings : int (optional)
        liczba po ktorej zatrzymuje sie wykrywanie p. przeciecia (domyslnie 6)
        
    Returns
    ------
    cross : dict
        Klucz - nazwa klasy (np. "Class1"), wartosc - slownik zawierajacy nazwe pliku (
            np. Class1_it2_example_0001) oraz liste p. przeciecia (int)
    
    """
    crossings_by_class = {}

    classes = sorted({item["class"] for item in dataset})
    for class_id in classes:
        crossings_by_class[class_id] = {}
    
    amp_thresholds = {
        "Class1": 0.40,
        "Class2": 0.30,
        "Class3": 0.10,
        "Class4": 0.40,
    }

    for class_id in classes:
        
        items = [item for item in dataset if item["class"] == class_id]
        class_id_threshold = amp_thresholds.get(class_id, 0.0)
        
        #n_crossings_list = []
        
        for item in items:
            sig = item["signal"].iloc[:,1].values
            
            xf = pd.Series(sig).rolling(window_fast, min_periods=1, center=True).mean().to_numpy()
            xs = pd.Series(sig).rolling(window_slow, min_periods=1, center=True).mean().to_numpy()
            
            crossings = []
            # zeby pierwszy crossing nie byl odrzucony przez warunek min_distance
            last_cross = -min_distance
            found_first_valid = False
            kept_count = 0
            
            for i in range(1, len(sig)):
                crossed = ((xf[i] >= xs[i] and xf[i-1] < xs[i-1]) or
                           (xf[i] <= xs[i] and xf[i-1] > xs[i-1]))
                
                if crossed and (i - last_cross >= min_distance):
                    amp = sig[i]
                    
                    if not found_first_valid:
                        if amp >= class_id_threshold:
                            # pierwszy crossing spełniający próg
                            found_first_valid = True
                        else:
                            # za mała amplituda -> odrzucamy
                            continue
                        
                    crossings.append(i)
                    last_cross = i
                    kept_count += 1
                    
                    if kept_count >= max_crossings:
                        break
                    
            peaks_dict = {
                "P1": tuple(crossings[0:2]), # indeksy 0,1
                "P2": tuple(crossings[2:4]), # indeksy 2,3
                "P3": tuple(crossings[4:6]), # indeksy 4,5
            }

            crossings_by_class[class_id][item["file"]] = peaks_dict
        
    return crossings_by_class


def compute_ranges_avg(dataset):
    r_c1_c3 = compute_crossings(dataset, min_distance=8)
    r_c2_c4 = compute_crossings(dataset, min_distance=12)

    merged = {}
    for class_id in ("Class1", "Class3"):
        merged[class_id] = r_c1_c3[class_id]
    for class_id in ("Class2", "Class4"):
        merged[class_id] = r_c2_c4[class_id]

    return merged


time_pm3 = {
    "Class1": {"P1": (17, 73), "P2": (34, 107), "P3": (60, 143)},
    "Class2": {"P1": (17, 59), "P2": (36, 98), "P3": (61, 138)},
    "Class3": {"P1": (13, 42), "P2": (32, 83), "P3": (49, 116)},
    "Class4": {"P2": (32, 74)},}
# zaokraglone w gore 
time_full = {
    "Class1": {"P1": (20, 70), "P2": (38, 104), "P3": (63, 140)},
    "Class2": {"P1": (20, 56), "P2": (39, 86), "P3": (64, 135)},
    "Class3": {"P1": (16, 39), "P2": (35, 80), "P3": (52, 113)},
    "Class4": {"P2": (35, 71)},}

time_whiskers = {
    "Class1": {"P1": (20, 48), "P2": (38, 85.5), "P3": (63, 132)},
    "Class2": {"P1": (20, 48), "P2": (39, 82), "P3": (64, 121)},
    "Class3": {"P1": (17, 37), "P2": (37, 77), "P3": (52, 109)},
    "Class4": {"P2": (49, 71)},}


amps_pm3 = {
    "Class1": {"P1": (0.864, 1.0), "P2": (0.315, 0.937), "P3": (0.127, 0.952)},
    "Class2": {"P1": (0.778, 1.0), "P2": (0.711, 1.0), "P3": (0.24, 1.0)},
    "Class3": {"P1": (0.185, 0.941), "P2": (0.732, 1.0), "P3": (0.616, 1.0)},
    "Class4": {"P2": (0.905, 1.0)}}

amps_full = {
    "Class1": {"P1": (0.914, 0.995), "P2": (0.365, 0.887), "P3": (0.177, 0.902)},
    "Class2": {"P1": (0.828, 0.997), "P2": (0.761, 0.998), "P3": (0.29, 0.997)},
    "Class3": {"P1": (0.235, 0.891), "P2": (0.782, 0.998), "P3": (0.666, 0.998)},
    "Class4": {"P2": (0.955, 0.998)},}

amps_whiskers = {
    "Class1": {"P1": (0.938, 0.995), "P2": (0.474, 0.887), "P3": (0.219, 0.902)},
    "Class2": {"P1": (0.842, 0.997), "P2": (0.898, 0.998), "P3": (0.303, 0.997)},
    "Class3": {"P1": (0.242, 0.891), "P2": (0.944, 0.998), "P3": (0.697, 0.998)},
    "Class4": {"P2": (0.985, 0.998)},}


ranges_pm3 = (time_pm3, amps_pm3)
ranges_full = (time_full, amps_full)
ranges_whiskers = (time_whiskers, amps_whiskers)

if __name__ == "__main__":
    print(ranges_pm3, ranges_full, ranges_whiskers)