# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 01:08:05 2025

@author: User
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.signal import find_peaks, hilbert, find_peaks_cwt
import itertools
from single_peak_detection import load_data

def detect_ma_crossings(signal, window_fast=5, window_slow=20,
                        min_distance=30, lookback=20):
    """
    Wykrywanie fast/slow MA crossing.
    Zwraca listę zakresów (start_idx, end_idx) = (crossing - lookback, crossing).

    x: sygnał
    window_fast: okno szybkiej średniej
    window_slow: okno wolnej średniej
    min_distance: minimalny odstęp w próbkach między crossingami
    lookback: zakres "wstecz" od crossing do analizy piku
    """

    # --- średnie ruchome ---
    xf = pd.Series(signal).rolling(window_fast, min_periods=1).mean().to_numpy()
    xs = pd.Series(signal).rolling(window_slow, min_periods=1).mean().to_numpy()

    crossings = []
    last_cross = -min_distance

    for i in range(1, len(signal)):
        # przecięcie: fast przebija slow
        crossed = ((xf[i] >= xs[i] and xf[i-1] < xs[i-1]) or
                   (xf[i] <= xs[i] and xf[i-1] > xs[i-1]))

        if crossed and (i - last_cross >= min_distance):
            start = max(0, i - lookback)
            crossings.append((start, i))
            last_cross = i

    return crossings

base_path = r"C:\Users\User\OneDrive\Dokumenty\praca inżynierska\ICP_pulses_it1"
data = load_data(base_path)

pd.options.display.float_format = '{:.4f}'.format