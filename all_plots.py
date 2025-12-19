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
    """Zwraca True, jeśli pik jest wykryty (nie NaN, nie pusta lista)."""
    if v is None:
        return False
    if isinstance(v, list):
        return len(v) > 0
    try:
        return not math.isnan(v)
    except TypeError:
        return True  # np. int



def peak_signature(peaks):
    """Zwraca tuple wykrytych pików spośród P1, P2, P3"""
    return tuple(p for p in ["P1", "P2", "P3"] if has_peak(peaks.get(p)))


    
def plot_upset_classic_postproc(results_pp, class_name):
    rows = []

    for r in results_pp:
        if r["class"] != class_name:
            continue

        d = r["peaks_detected"]

        rows.append({
            "P1": not (isinstance(d.get("P1"), float) and math.isnan(d.get("P1"))),
            "P2": not (isinstance(d.get("P2"), float) and math.isnan(d.get("P2"))),
            "P3": not (isinstance(d.get("P3"), float) and math.isnan(d.get("P3"))),
        })

    df = pd.DataFrame(rows)
    
    series = df.value_counts().sort_index()
    total = 250
    
    perc = series / total * 100

    plt.figure(figsize=(6, 4))
    u = UpSet(
        perc,
        show_counts=True,
        sort_by="cardinality",
        sort_categories_by="-input"
    ).plot()

    plt.suptitle(f"Klasa {class_name[-1]} – diagram UpSet", fontsize=12)

    # podpisy osi po polsku
    for ax in plt.gcf().axes:
        if ax.get_ylabel() == "Intersection size":
            ax.set_ylabel("Udział sygnałów [%]")
        if ax.get_xlabel() == "None":
            ax.set_xlabel("Kombinacje wykrytych pików")
    u['totals'].set_xlabel("Wykryte piki [%]", fontsize=10)
    plt.tight_layout()
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
    
    
def plot_signal_pre_post(
    results_combined,
    results_combined_pp,
    file_name
):
    """
    Wykres sygnału z pikami:
    [ PRZED POSTPROCESSINGIEM ] [ PO POSTPROCESSINGU ]
    Styl zgodny z przykładem z prezentacji.
    """

    pre = next(d for d in results_combined if d["file"] == file_name)
    post = next(d for d in results_combined_pp if d["file"] == file_name)

    sig = pre["signal"]
    t = sig.iloc[:, 0].values
    y = sig.iloc[:, 1].values

    # kolory pików
    peak_colors = {'P1': 'red', 'P2': 'green', 'P3': 'blue'}
    peak_colors_d = {'P1': 'orange', 'P2': 'yellow', 'P3': 'cyan'}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    # =========================
    # -------- PRZED ----------
    # =========================
    ax = axes[0]
    ax.plot(t, y, color="black", linewidth=1.5)

    for peak, color in peak_colors.items():
        # referencyjne
        ref = pre["peaks_ref"].get(peak)
        if ref is not None and not (isinstance(ref, float) and math.isnan(ref)):
            ax.plot(t[ref], y[ref], "o", color=color, markersize=8,
                    label=f"{peak} – referencyjny")

        # wykryte (lista)
        det = pre["peaks_detected"].get(peak, [])
        for i in det:
            ax.plot(t[i], y[i], "x", color=peak_colors_d.get(peak, "gray"), markersize=9,
                    markeredgewidth=2,
                    label=f"{peak} – wykryty")

    ax.set_title("Przed postprocessingiem")
    ax.set_xlabel("Numer próbki")
    ax.set_ylabel("Amplituda")
    ax.legend(loc="upper right", fontsize=7)

    # =========================
    # ---------- PO -----------
    # =========================
    ax = axes[1]
    ax.plot(t, y, color="black", linewidth=1.5)

    for peak, color in peak_colors.items():
        # referencyjne
        ref = post["peaks_ref"].get(peak)
        if ref is not None and not (isinstance(ref, float) and math.isnan(ref)):
            ax.plot(t[ref], y[ref], "o", color=color, markersize=8,
                    label=f"{peak} – referencyjny")

        # wykryte (po postproc: int lub NaN)
        det = post["peaks_detected"].get(peak)
        if det is not None and not (isinstance(det, float) and math.isnan(det)):
            ax.plot(t[det], y[det], "x", color=peak_colors_d.get(peak, "gray"), markersize=9,
                    markeredgewidth=2,
                    label=f"{peak} – wykryty")

    ax.set_title("Po postprocessingu")
    ax.set_xlabel("Numer próbki")
    ax.legend(loc="upper right", fontsize=7)

    fig.suptitle(file_name, fontsize=14)
    plt.tight_layout()
    plt.show()
    
    

def plot_signal_with_concave_areas(dataset, filename):
    # znajdź element w dataset o podanej nazwie pliku
    item = next((d for d in dataset if d["file"] == filename), None)
    if item is None:
        raise ValueError(f"Nie znaleziono pliku {filename} w dataset")

    sig_df = item["signal"]
    y = sig_df.iloc[:, 1].values
    t = sig_df.iloc[:, 0].values

    # oblicz pochodne
    dy = np.gradient(y, edge_order=2)
    d2y = np.gradient(dy, edge_order=2)

    # maski dla obszarów wklęsłych i wypukłych
    concave_mask = d2y < 0  # wklęsłe
    convex_mask = d2y >= 0  # wypukłe

    # funkcja pomocnicza do znajdowania granic regionów
    def regions(mask):
        mask_diff = np.diff(mask.astype(int))
        starts = np.where(mask_diff == 1)[0] + 1
        ends = np.where(mask_diff == -1)[0]
        if mask[0]:
            starts = np.insert(starts, 0, 0)
        if mask[-1]:
            ends = np.append(ends, len(mask)-1)
        return list(zip(starts, ends))

    concave_regions = regions(concave_mask)
    convex_regions = regions(convex_mask)

    # rysowanie
    plt.figure(figsize=(12, 5))
    plt.plot(t, y, color='black', label='Sygnał')

    for start, end in concave_regions:
        plt.axvspan(t[start], t[end], color='teal', alpha=0.3)
        plt.text((t[start]+t[end])/2, max(y), 
                 f"d1: {np.mean(dy[start:end]):.3f}\nd2: {np.mean(d2y[start:end]):.3f}", 
                 ha='center', va='bottom', fontsize=8, color='teal')

    for start, end in convex_regions:
        plt.axvspan(t[start], t[end], color='purple', alpha=0.3)
        plt.text((t[start]+t[end])/2, min(y), 
                 f"d1: {np.mean(dy[start:end]):.3f}\nd2: {np.mean(d2y[start:end]):.3f}", 
                 ha='center', va='top', fontsize=8, color='purple')

    plt.xlabel("Numer próbki")
    plt.ylabel("Amplituda")
    plt.title(f"Sygnał {filename} z obszarami wklęsłymi (teal) i wypukłymi (fiolet)")
    plt.show()
