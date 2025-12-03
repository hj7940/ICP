# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 01:08:05 2025

@author: User
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import glob
# import os
# from scipy.signal import find_peaks, hilbert, find_peaks_cwt
# import itertools
from single_peak_detection import load_data
import math


# %% ------- FUNKCJE ----------

def compute_crossings_summary(dataset, window_fast=2, window_slow=10,
                              min_distance=30, lookback=20, 
                              min_crossing_index=15, max_crossing_index=(160)):
    """
    Liczy MA crossingi dla wszystkich sygnałów w dataset.
    Zwraca:
        - crossings_by_class: dict z listą crossingów dla każdego sygnału
        - summary_df: DataFrame z liczbą crossingów w każdej klasie i parametrami
    """
    crossings_by_class = {}
    summary_rows = []

    classes = sorted({item["class"] for item in dataset})
    for cls in classes:
        crossings_by_class[cls] = []

    for cls in classes:
        # wszystkie sygnały danej klasy
        items = [item for item in dataset if item["class"] == cls]
        num_signals = len(items)
        n_crossings_list = []
        
        n_crossings_list = []
        
        for item in items:
            sig = item["signal"].iloc[:,1].values
            xf = pd.Series(sig).rolling(window_fast, min_periods=1).mean().to_numpy()
            xs = pd.Series(sig).rolling(window_slow, min_periods=1).mean().to_numpy()

            crossings = []
            last_cross = -min_distance

            for i in range(1, len(sig)):
                crossed = ((xf[i] >= xs[i] and xf[i-1] < xs[i-1]) or
                           (xf[i] <= xs[i] and xf[i-1] > xs[i-1]))
                if crossed and (i - last_cross >= min_distance) and (max_crossing_index > i > min_crossing_index):
                    start = max(0, i - lookback)
                    crossings.append((start, i))
                    last_cross = i

            crossings_by_class[cls].append({
                "File": item["file"],
                "Crossings": crossings
            })
            n_crossings_list.append(len(crossings))
            # total_crossings += len(crossings)
            # avg_crossings = total_crossings / num_signals if num_signals > 0 else 0
        
        avg_crossings = np.mean(n_crossings_list) if n_crossings_list else 0
        min_crossings = np.min(n_crossings_list) if n_crossings_list else 0
        max_crossings = np.max(n_crossings_list) if n_crossings_list else 0
        
        # podsumowanie dla klasy
        summary_rows.append({
            "Class": cls,
            "Num_Signals": num_signals,
            "Avg_Crossings": avg_crossings,
            "Min_Crossings": min_crossings,
            "Max_Crossings": max_crossings,
            "window_fast": window_fast,
            "window_slow": window_slow,
            "min_distance": min_distance,
            "lookback": lookback
        })

    summary_df = pd.DataFrame(summary_rows)
    return crossings_by_class, summary_df



def plot_crossings_overview(dataset, crossings_by_class, mode="lines",
                            hist_bins=50, alpha_line=0.15):

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    classes = sorted(crossings_by_class.keys())

    for ax, cls in zip(axes, classes):
        items = [item for item in dataset if item["class"] == cls]
        if not items:
            ax.set_visible(False)
            continue

        t = items[0]["signal"].iloc[:, 0].values
        
        all_x = []
        # sygnały
        for item in items:
            x = item["signal"].iloc[:, 1].values
            ax.plot(t, x, color="black", alpha=0.07)
            all_x.append(x)
        entries = crossings_by_class[cls]
        
        # --- średnia krzywa fali ICP ---
        mean_signal = np.mean(all_x, axis=0)
        ax.plot(t, mean_signal, color="blue", linewidth=2.0, label="Średni sygnał")
        ax.margins(y=0, x=0)

        if mode == "lines":
            for entry in entries:
                for start, end in entry["Crossings"]:
                    ax.axvline(t[end], color="red", alpha=alpha_line)

        elif mode == "hist":
            times = [
                t[end]
                for entry in entries
                for (start, end) in entry["Crossings"]
            ]
            if times:
                ax2 = ax.twinx()
                ax2.hist(times, bins=hist_bins, alpha=0.35, color="red")
                ax2.set_ylabel("Cross count")

        ax.set_title(cls)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    
def plot_crossings_single_signal(dataset, crossings_by_class, cls, idx,
                                      window_fast=2, window_slow=10,
                                      show_lines=True, show_points=True):

    """
    Rysuje sygnał z klasy `cls` o indeksie `idx` wraz z:
        - fast i slow średnimi ruchomymi
        - wykrytymi crossingami MA
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    # wybór sygnału z danej klasy
    items = [item for item in dataset if item["class"] == cls]
    if idx >= len(items):
        raise IndexError(f"Klasa {cls} ma tylko {len(items)} sygnałów")

    item = items[idx]
    t = item["signal"].iloc[:, 0].values
    x = item["signal"].iloc[:, 1].values
    file_name = item["file"]

    # fast i slow MA
    xf = pd.Series(x).rolling(window_fast, min_periods=1).mean().to_numpy()
    xs = pd.Series(x).rolling(window_slow, min_periods=1).mean().to_numpy()

    # znajdź odpowiadający wpis crossingów
    class_entries = crossings_by_class[cls]
    entry = next(e for e in class_entries if e["File"] == file_name)
    crossings = entry["Crossings"]

    # Rysowanie
    plt.figure(figsize=(12, 6))
    plt.plot(t, x, color="black", label="Signal")
    plt.plot(t, xf, color="blue", linestyle="--", linewidth=1.5, label=f"Fast MA ({window_fast})")
    plt.plot(t, xs, color="orange", linestyle="--", linewidth=1.5, label=f"Slow MA ({window_slow})")

    # crossingi
    if show_lines:
        for start, end in crossings:
            plt.axvline(t[end], color="red", alpha=0.5)
    if show_points:
        for start, end in crossings:
            plt.scatter(t[end], x[end], color="red", s=50, zorder=5)

    plt.title(f"{cls} — {file_name} — {len(crossings)} crossings")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def sweep_min_distance_all_classes(dataset, min_distance_values,
                                   window_fast=5, window_slow=20,
                                   lookback=20, min_crossing_index=15,
                                   max_crossing_index=160):
    """
    Sweep po min_distance dla wszystkich klas.
    Zwraca DataFrame z min, max i średnią crossingów dla każdej klasy.
    Tworzy wykres z 4 subplotsami (po jednej klasie).
    """
    classes = sorted({item["class"] for item in dataset})
    summary_rows = []

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for ax, cls in zip(axes, classes):
        items = [item for item in dataset if item["class"] == cls]
        # num_signals = len(items)

        min_list, max_list, avg_list = [], [], []

        for min_dist in min_distance_values:
            n_crossings_list = []

            for item in items:
                sig = item["signal"].iloc[:,1].values
                xf = pd.Series(sig).rolling(window_fast, min_periods=1).mean().to_numpy()
                xs = pd.Series(sig).rolling(window_slow, min_periods=1).mean().to_numpy()

                crossings = []
                last_cross = -min_dist

                for i in range(1, len(sig)):
                    crossed = ((xf[i] >= xs[i] and xf[i-1] < xs[i-1]) or
                               (xf[i] <= xs[i] and xf[i-1] > xs[i-1]))
                    if crossed and (i - last_cross >= min_dist) and (min_crossing_index < i < max_crossing_index):
                        start = max(0, i - lookback)
                        crossings.append((start, i))
                        last_cross = i

                n_crossings_list.append(len(crossings))

            min_c = np.min(n_crossings_list) if n_crossings_list else 0
            max_c = np.max(n_crossings_list) if n_crossings_list else 0
            avg_c = np.mean(n_crossings_list) if n_crossings_list else 0

            min_list.append(min_c)
            max_list.append(max_c)
            avg_list.append(avg_c)

            # zapis do DataFrame
            summary_rows.append({
                "Class": cls,
                "min_distance": min_dist,
                "Min_Crossings": min_c,
                "Max_Crossings": max_c,
                "Avg_Crossings": avg_c,
                "window_fast": window_fast,
                "window_slow": window_slow,
                "lookback": lookback
            })

        # --- wykresy ---
        ax.plot(min_distance_values, min_list, 'o-', label="Min crossings")
        ax.plot(min_distance_values, max_list, 's-', label="Max crossings")
        ax.plot(min_distance_values, avg_list, '^-', label="Avg crossings")
        ax.set_title(cls)
        ax.set_xlabel("min_distance")
        ax.set_ylabel("Liczba crossingów")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.show()

    summary_df = pd.DataFrame(summary_rows)
    return summary_df


def plot_all_signals_class_with_peaks(dataset, cls, peak_colors={"P1":"green","P2":"orange","P3":"red"}):
    """
    Rysuje wszystkie sygnały z danej klasy w osobnych subplotach.
    Każdy sygnał ma zaznaczone referencyjne piki P1/P2/P3.
    """
    items = [item for item in dataset if item["class"] == cls]
    n_signals = len(items)
    
    if n_signals == 0:
        print(f"Brak sygnałów dla klasy {cls}")
        return

    # --- ustawienia siatki ---
    n_cols = 5
    n_rows = math.ceil(n_signals / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 2*n_rows), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, item in enumerate(items):
        t = item["signal"].iloc[:,0].values
        x = item["signal"].iloc[:,1].values
        axes[i].plot(t, x, color='blue')
        
        # --- piki referencyjne ---
        peaks = item.get("peaks_ref", {})
        for p_name, idx in peaks.items():
            if idx is not None and 0 <= idx < len(t):
                axes[i].plot(t[idx], x[idx], 'o', color=peak_colors.get(p_name, 'black'), alpha=0.8)
        
        axes[i].set_title(f"{item['file']}")
        axes[i].grid(True, alpha=0.3)

    # Wyłącz puste subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Klasa {cls} – wszystkie sygnały z pikami referencyjnymi", fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_signals_with_single_crossing(dataset, crossings_by_class, classes_to_plot=[1,2,3],
                                      peak_colors={"P1":"green","P2":"orange","P3":"red"}):
    """
    Wyplotowanie wszystkich sygnałów, w których wykryto dokładnie 1 crossing,
    dla klas podanych w classes_to_plot.
    Każdy sygnał w osobnym subplotie z referencyjnymi pikami i crossingiem.
    """
    import math

    classes_to_plot = [f"Class{i}" for i in classes_to_plot]

    for cls in classes_to_plot:
        items = [item for item in dataset if item["class"] == cls]
        # filtrujemy sygnały z dokładnie 1 crossingiem
        selected_items = []
        for item in items:
            entry = next(e for e in crossings_by_class[cls] if e["File"] == item["file"])
            if len(entry["Crossings"]) == 1:
                selected_items.append((item, entry["Crossings"][0]))

        n_signals = len(selected_items)
        if n_signals == 0:
            print(f"Klasa {cls}: brak sygnałów z dokładnie 1 crossingiem")
            continue

        # --- ustawienia siatki ---
        n_cols = 5
        n_rows = math.ceil(n_signals / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 2*n_rows), sharex=True, sharey=True)
        axes = axes.flatten()

        for i, (item, crossing) in enumerate(selected_items):
            t = item["signal"].iloc[:,0].values
            x = item["signal"].iloc[:,1].values
            axes[i].plot(t, x, color='blue')

            # referencyjne piki
            peaks = item.get("peaks_ref", {})
            for p_name, idx in peaks.items():
                if idx is not None and 0 <= idx < len(t):
                    axes[i].plot(t[idx], x[idx], 'o', color=peak_colors.get(p_name, 'black'), alpha=0.8)

            # wykryty crossing
            start, end = crossing
            axes[i].axvline(t[end], color="red", alpha=0.6)
            axes[i].scatter(t[end], x[end], color="red", s=40)

            axes[i].set_title(f"{item['file']}")
            axes[i].grid(True, alpha=0.3)

        # Wyłącz puste subplots
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f"Klasa {cls} — sygnały z 1 crossingiem", fontsize=16)
        plt.tight_layout()
        plt.show()


# %% ------------ EXECUTION --------------
base_path = r"C:\Users\User\OneDrive\Dokumenty\praca inżynierska\ICP_pulses_it1"
data = load_data(base_path)

pd.options.display.float_format = '{:.4f}'.format

cross, summary = compute_crossings_summary(
    data,
    window_fast=2,
    window_slow=5,
    min_distance=15,
    lookback=20,
    min_crossing_index=15,
    max_crossing_index=160
)

print(summary)
# plot_crossings_overview(data, cross, mode="lines")
# plot_crossings_overview(data, cross, mode="hist")

# plot_crossings_single_signal(data, cross, "Class2", 20)


# min_distances = [20, 25, 30, 40, 50]
# summary_df = sweep_min_distance_all_classes(data, min_distances)
# print(summary_df)

# plot_all_signals_class_with_peaks(data, "Class1")

plot_signals_with_single_crossing(data, cross)