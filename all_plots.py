# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 00:37:04 2025

Wszystkie wykresy 
@author: Hanna Jaworska
"""

from collections import Counter
from upsetplot import UpSet, from_memberships
import itertools
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def has_peak(v):
    """Zwraca True, jeśli pik jest wykryty (nie NaN)."""
    return v is not None and not math.isnan(v)



def peak_signature(peaks):
    """Zwraca tuple wykrytych pików spośród P1, P2, P3"""
    return tuple(p for p in ["P1", "P2", "P3"] if has_peak(peaks.get(p)))


def plot_upset_for_class(results_combined, class_name):
    # zbieramy raw combo counts
    combos = [peak_signature(r["peaks_detected"]) for r in results_combined if r["class"] == class_name]

    # wszystkie możliwe kombinacje trzech kategorii
    all_combos = []
    for r in itertools.product([False, True], repeat=3):
        # tuple np. (True,False,True) -> maska
        combo_names = tuple(c for flag, c in zip(r, ["P1","P2","P3"]) if flag)
        all_combos.append(combo_names)

    # zlicz
    counts = Counter(combos)

    # budujemy serię z MultiIndex
    index = []
    values = []
    for combo in all_combos:
        # bool tuple do indeksu
        bool_idx = tuple([c in combo for c in ["P1","P2","P3"]])
        index.append(bool_idx)
        values.append(counts.get(combo, 0))
        

    index = pd.MultiIndex.from_tuples(index, names=["P1","P2","P3"][::-1])
    series = pd.Series(values, index=index)

    # teraz UpSet
    plt.figure(figsize=(6,4))
    u = UpSet(series, show_counts=True).plot()
    
    plt.suptitle(f"Diagram UpSet – Klasa {class_name[-1]}", fontsize=12)
    for ax in plt.gcf().axes:
        if ax.get_ylabel() == "Intersection size":
            ax.set_ylabel("Liczba sygnałów", fontsize=8)
        if ax.get_xlabel() == "None":
            ax.set_xlabel("Liczba wykryć piku")  # usuwa niepotrzebne etykiety
    u['totals'].set_xlabel("Liczba wykryć piku", fontsize=8)
    plt.tight_layout(pad=5, w_pad=5, h_pad=5)
    plt.subplots_adjust(left=0.2)
    plt.show()
    
    

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
        
        for p, ref_idx in item.get("peaks_ref", {}).items(): 
            if ref_idx is None or (isinstance(ref_idx, float) and np.isnan(ref_idx)):
                continue  # ← tutaj zmiana
            if isinstance(ref_idx, (list, tuple, np.ndarray)):
                ref_idx = list(ref_idx)
            else:
                ref_idx = [ref_idx]  # ← tutaj zmiana

            ax.scatter(
                t[ref_idx],
                y[ref_idx],
                color=peak_colors.get(p, "gray"),
                marker="o",
                s=60,
                alpha=1.0,
                label=f"{p} ref" if idx == 0 else "",
            )

        # ===== PIKI WYKRYTE =====
        try:
            for p, detected_idx in item.get("peaks_detected", {}).items(): 
                if not detected_idx:
                    continue 
    
                ax.scatter(
                    t[detected_idx],
                    y[detected_idx],
                    color=peak_colors_d.get(p, "gray"),
                    marker="x",
                    s=60,
                    alpha=1.0,
                    label=f"{p} detected" if idx == 0 else "",
                )
        except:
            for p, detected_idx in item.get("peaks_detected", {}).items():
                if detected_idx is None or (isinstance(detected_idx, float) and np.isnan(detected_idx)):
                    continue
        
                detected_idx = int(detected_idx)
        
                ax.scatter(
                    [t[detected_idx]],
                    [y[detected_idx]],
                    color=peak_colors_d.get(p, "gray"),
                    marker="x",
                    s=60,
                    alpha=1.0,
                    label=f"{p} detected" if idx == 0 else "",
                )
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