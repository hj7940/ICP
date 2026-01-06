# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 00:37:04 2025

Wszystkie wykresy 

plot_upset_classic_postproc - upsetplot pokazujacy ile razy byla wykryta 
                            konkretna kombinacja pikow

plot_files_in_class         - wszystkie sygnaly danej klasy, 
                            zaznaczone piki referencyjne i detected

plot_signal_pre_post        - pokazuje wyniki detekcji  przed i po zrobieniu 
                            "postprocessingu" w final_algorithm
                        
plot_signal_with_concave_areas - self-explainatorty, ostatecznie jako 
                            rysunek do pracy - ilustracja metod
                
plot_all_signals_with_peaks_final - wszystkie sygnaly klasy na jednym wykresie
plot_all_signals_with_peaks_by_peak_type - jw, ale osobne histogramy dla p1, p2, p3

plot_concave_signals        - po 20 przykladow z kazdej klasy

plot_peak_features_boxplots - tworzenie boxplotow do wynikow z peaks_morphology

plot_peak_detection_pie     - pie chart (ile % sygnalow ma wykryte wszystkie piki),
                            w zamysle dla Klasy 4 zamiast UpSet plot

@author: Hanna Jaworska
"""

from upsetplot import UpSet
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from itertools import groupby
import ast
from ranges import compute_crossings, compute_ranges_avg
import glob


# def has_peak(v):
#     """Zwraca True, jeśli pik jest wykryty (nie NaN, nie pusta lista)."""
#     if v is None:
#         return False
#     if isinstance(v, list):
#         return len(v) > 0
#     try:
#         return not math.isnan(v)
#     except TypeError:
#         return True  # np. int

# def peak_signature(peaks):
#     """Zwraca tuple wykrytych pików spośród P1, P2, P3"""
#     return tuple(p for p in ["P1", "P2", "P3"] if has_peak(peaks.get(p)))

    
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
    peak_colors_d = {'P1': 'orange', 'P2': 'chartreuse', 'P3': 'cornflowerblue'}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    
    
    # =========================
    # -------- PRZED ----------
    # =========================
    ax = axes[0]
    ax.plot(t, y, color="black", linewidth=1)

    for peak, color in peak_colors.items():
        # referencyjne
        ref = pre["peaks_ref"].get(peak)
        if ref is not None and not (isinstance(ref, float) and math.isnan(ref)):
            ax.plot(t[ref], y[ref], "o", color=color, markersize=10,
                    label=f"{peak} – referencyjny", zorder=9)

        # wykryte (lista)
        det = pre["peaks_detected"].get(peak, [])
        for i in det:
            ax.plot(t[i], y[i], "x", color=peak_colors_d.get(peak, "gray"), 
                    markersize=12,
                    markeredgewidth=3,
                    alpha=0.95,
                    label=f"{peak} – wykryty", zorder=10)

    ax.set_title("Przed", fontsize=18)
    ax.set_xlabel("Numer próbki", fontsize=14, labelpad=3)
    ax.set_ylabel("Znormalizowana amplituda", fontsize=14, labelpad=3)
    # handles, labels = ax.get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=10)
    ax.set_xlim(0, 180)
    ax.set_ylim(bottom=0)

    # =========================
    # ---------- PO -----------
    # =========================
    ax = axes[1]
    ax.plot(t, y, color="black", linewidth=1)

    for peak, color in peak_colors.items():
        # referencyjne
        ref = post["peaks_ref"].get(peak)
        if ref is not None and not (isinstance(ref, float) and math.isnan(ref)):
            ax.plot(t[ref], y[ref], "o", color=color, markersize=10,
                    label=f"{peak} – referencyjny", zorder=9)

        # wykryte (po postproc: int lub NaN)
        det = post["peaks_detected"].get(peak)
        if det is not None and not (isinstance(det, float) and math.isnan(det)):
            ax.plot(t[det], y[det], "x", color=peak_colors_d.get(peak, "gray"),
                    markersize=12,
                    markeredgewidth=3,
                    alpha=0.95,
                    label=f"{peak} – wykryty", zorder=10)

    ax.set_title("Po", fontsize=18, pad=3)
    ax.set_xlabel("Numer próbki", fontsize=14, labelpad=3)
    ax.legend(loc="upper right", fontsize=9.2)
    ax.set_xlim(0, 180)
    ax.set_ylim(bottom=0)

    fig.suptitle("")
    plt.tight_layout()
    # plt.show()
    
    
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

    plt.xlabel("Numer próbki", fontsize=16)
    plt.ylabel("Znormalizowana amplituda", fontsize=16)
    plt.title(f"Sygnał {filename} z obszarami wklęsłymi (teal) i wypukłymi (fiolet)")
    plt.show()


def plot_all_signals_with_peaks_final(
    detection_results,
    method_name,
    ranges_name,
    classes=("Class1", "Class2", "Class3", "Class4")
):
    """
    4 subplotsy: Class1..Class4
    - wszystkie sygnały
    - średni sygnał
    - histogram gęstości pików referencyjnych (green)
    - histogram gęstości pików automatycznych (red)

    detection_results : lista słowników z peak_detection
    method_name       : np. "concave", "concave_tuned", "curvature"
    ranges_name       : np. "full", "pm3", "avg", "none"
    """

    titles_dict = {
        "concave": "Maksima w odcinkach wklęsłych",
        "concave_tuned": "Maksima w odcinkach wklęsłych (parametry stroone)",
        "modified_scholkmann0-5": "Zmodyfikowana metoda Scholkmanna (0.5)",
        "modified_scholkmann1": "Zmodyfikowana metoda Scholkmanna (1.0)",
        "curvature": "Maksymalna krzywizna",
        "line_distance_8": "Odległość od linii bazowej",
        "line_perpendicular_8": "Odległość prostopadła do linii",
        "hilbert": "Transformata Hilberta",
        "wavelet": "Ciągła transformata falkowa"
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, class_name in enumerate(classes):
        ax = axes[i]

        # --- tylko dane tej klasy, metody i zakresu ---
        class_items = [
            d for d in detection_results
            if d["class"] == class_name
            and d["method"] == method_name
            and d["ranges"] == ranges_name
        ]

        if not class_items:
            ax.set_title(f"{class_name} – brak danych")
            continue

        # --- wszystkie sygnały ---
        all_x = []
        t = class_items[0]["signal"].iloc[:, 0].values

        for item in class_items:
            sig = item["signal"]
            x = sig.iloc[:, 1].values
            ax.plot(t, x, color="black", alpha=0.07, linewidth=0.7)
            all_x.append(x)

        # --- średni sygnał ---
        mean_signal = np.mean(all_x, axis=0)
        ax.plot(t, mean_signal, color="blue", linewidth=2, label="Średni sygnał")

        # --- zbieranie pików ---
        manual_t = []
        auto_t = []

        for item in class_items:
            sig = item["signal"]
            t = sig.iloc[:, 0].values

            # referencyjne
            for idx in item["peaks_ref"].values():
                if idx is not None and not np.isnan(idx):
                    manual_t.append(t[int(idx)])

            # wykryte
            for plist in item["peaks_detected"].values():
                for idx in plist:
                    auto_t.append(t[int(idx)])

        ax_hist = ax.twinx()

        if manual_t:
            ax_hist.hist(
                manual_t,
                bins=40,
                density=True,
                alpha=0.35,
                color="green",
                label="Piki referencyjne"
            )

        if auto_t:
            ax_hist.hist(
                auto_t,
                bins=40,
                density=True,
                alpha=0.35,
                color="red",
                label="Piki wykryte"
            )

        ax.set_title(f"Klasa {class_name[-1]}", fontsize=14)
        ax.set_xlabel("Numer próbki")
        ax.set_ylabel("Amplituda")
        ax.grid(alpha=0.3)

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax_hist.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right")

    fig.suptitle(
        f"{titles_dict.get(method_name, method_name)} — zakres: {ranges_name}",
        fontsize=18
    )
    plt.tight_layout()
    plt.show()

    
def plot_all_signals_with_peaks_by_peak_type(
    detection_results,
    method_name,
    ranges_name,
    classes=("Class1", "Class2", "Class3", "Class4"),
    peak_types=("P1", "P2", "P3")
):
    """
    Grid: wiersze = klasy, kolumny = typy pików
    - wszystkie sygnały
    - średni sygnał
    - histogram gęstości pików referencyjnych (green)
    - histogram gęstości pików automatycznych (red)
    """

    n_classes = len(classes)
    n_peaks = len(peak_types)

    fig, axes = plt.subplots(n_classes, n_peaks, figsize=(20, 16), squeeze=False, 
                             gridspec_kw={"wspace":0.4, "hspace":0.35})

    titles_dict = {
        "concave": "Maksima w odcinkach wklęsłych",
        "concave_tuned": "Maksima w odcinkach wklęsłych (+parametry find_peaks)",
        "concave_d2x=-0-002": "Maksima w odcinkach wklęsłych (próg 2. pochodnej: -0,002)",
        "concave_d2x=0-002": "Maksima w odcinkach wklęsłych (próg 2. pochodnej: 0,002)",
        "modified_scholkmann0-5": "Zmodyfikowany algorytm Scholkmanna (limit=0.5)",
        "modified_scholkmann1": "Zmodyfikowany algorytm Scholkmanna (limit=1.0)",
        "curvature": "Maksymalna krzywizna",
        "line_distance_8": "Odległość od linii",
        "line_perpendicular_8": "Odległość prostopadła do linii",
        "hilbert": "Transformata Hilberta",
        "wavelet": "Ciągła transformata falkowa"
    }
    
    ranges_dict = {
        "full": "min-max",
        "avg": "na podstawie średniej ruchomej",
        "whiskers": "wąsy"}

    for i, class_name in enumerate(classes):
        # wybieramy tylko dane tej klasy
        class_items = [
            d for d in detection_results
            if d["class"] == class_name
            and d["method"] == method_name
            and (ranges_name is None or d["ranges"] == ranges_name)
        ]

        if not class_items:
            for j, peak_type in enumerate(peak_types):
                axes[i, j].set_title(f"Klasa {class_name[-1]} – brak danych")
            continue

        t = class_items[0]["signal"].iloc[:, 0].values
        all_x = [item["signal"].iloc[:, 1].values for item in class_items]
        mean_signal = np.mean(all_x, axis=0)

        for j, peak_type in enumerate(peak_types):
            ax = axes[i, j]
            
            has_peaks = any(item["peaks_ref"].get(peak_type) is not None for item in class_items) or \
                any(item["peaks_detected"].get(peak_type) for item in class_items)
    
            if not has_peaks:
                ax.set_visible(False)
                continue

            # --- rysujemy wszystkie sygnały ---
            for item in class_items:
                sig = item["signal"].iloc[:, 1].values
                ax.plot(t, sig, color="black", alpha=0.05, linewidth=0.5)

            # --- średni sygnał ---
            ax.plot(t, mean_signal, color="blue", linewidth=1.8, label="Średni sygnał")

            # --- zbieramy piki ---
            manual_t = []
            auto_t = []

            for item in class_items:
                # referencyjne
                idx_ref = item["peaks_ref"].get(peak_type)
                if idx_ref is not None and not np.isnan(idx_ref):
                    manual_t.append(t[int(idx_ref)])

                # wykryte
                idx_auto_list = item["peaks_detected"].get(peak_type, [])
                for idx in idx_auto_list:
                    auto_t.append(t[int(idx)])

            ax_hist = ax.twinx()
            if manual_t:
                ax_hist.hist(manual_t, bins=40, density=False, alpha=0.4, color="green", label="Piki referencyjne")
            if auto_t:
                ax_hist.hist(auto_t, bins=40, density=False, alpha=0.4, color="red", label="Piki wykryte")
                
            ax.set_xlim(0, 180)
            ax.set_ylim(bottom=0)
            
            ax_hist.set_ylim(bottom=0)
            
            ax.margins(x=0, y=0)
            ax_hist.margins(y=0)
            
            ax.set_title(f"Klasa {class_name[-1]} {peak_type}", fontsize=16)
            ax.set_xlabel("Numer próbki", fontsize=12)
            ax.set_ylabel("Amplituda", fontsize=12)
            ax_hist.set_ylabel("Liczba pików", fontsize=12)
            ax.grid(alpha=0.3)

            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax_hist.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="upper right")

    fig.suptitle(
        f"{titles_dict.get(method_name, method_name)} — zakres: {ranges_dict.get(ranges_name, ranges_name if ranges_name is not None else 'none')}",
        fontsize=20,
        y=0.94
    )
    plt.tight_layout(rect=[0,0,1,0.97])



# --------------------------------------------------
# detekcja obszarów wklęsłych (d2x < 0)
# --------------------------------------------------
def concave_regions(y):
    d2 = np.gradient(np.gradient(y))
    mask = d2 < 0

    regions = []
    for k, g in groupby(enumerate(mask), key=lambda x: x[1]):
        if k:  # tylko True
            idx = [i for i, _ in g]
            regions.append((idx[0], idx[-1]))
    return regions


# --------------------------------------------------
# rysowanie
# --------------------------------------------------
def plot_concave_signals(dataset, max_per_class=20):
    colors = plt.cm.tab20.colors  # zamiennik pypalettes

    for class_name in ["Class1", "Class2", "Class3", "Class4"]:
        class_items = [
            d for d in dataset
            if d["class"] == class_name
            and "example_" in d["file"]
        ][:max_per_class]

        n = len(class_items)
        fig, axes = plt.subplots(
            n, 1, figsize=(11, 2.0 * n), sharex=True
        )

        if n == 1:
            axes = [axes]

        for i, (ax, item) in enumerate(zip(axes, class_items)):
            sig = item["signal"]
            y = sig.iloc[:, 1].values
            x = np.arange(len(y))  # numer próbki

            color = colors[i % len(colors)]

            # sygnał
            ax.plot(x, y, color=color, linewidth=1)

            # obszary wklęsłe
            ymin, ymax = y.min(), y.max()
            for start, end in concave_regions(y):
                ax.add_patch(
                    Rectangle(
                        (start, ymin),
                        end - start,
                        ymax - ymin,
                        color=color,
                        alpha=0.25,
                        linewidth=0
                    )
                )

            # piki referencyjne
            for peak_name, idx in item["peaks_ref"].items():
                if idx is not None and not np.isnan(idx):
                    idx = int(idx)
                    ax.axvline(
                        idx,
                        color="black",
                        linestyle="--",
                        linewidth=1
                    )
                    ax.text(
                        idx,
                        ymax,
                        peak_name,
                        fontsize=8,
                        ha="center",
                        va="bottom"
                    )

            ax.set_ylabel("Amplituda", fontsize=8)
            ax.set_title(item["file"], fontsize=9)

            # grid + ticki
            ax.grid(True, which="both", linestyle=":", linewidth=0.6)
            ax.set_xticks(np.arange(0, len(y), 5))

        axes[-1].set_xlabel("Numer próbki")

        fig.suptitle(
            f"{class_name} – obszary wklęsłe (d²x < 0) + piki referencyjne",
            fontsize=14
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


def plot_peak_features_boxplots(df, features=None, classes=("Class1","Class2","Class3","Class4")):
    """
    df - dataframe z kolumnami: Class, Peak, Feature, idx_*, h_*, prom_*, w50_*, ...
    features - lista kolumn do rysowania (np. ["Index","Height","Prominence","Width_50"])
    """


    if features is None:
        features = ["Index","Height","Prominence","Width_50"]

    for feat in features:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
        axes = axes.flatten()

        for i, class_id in enumerate(classes):
            ax = axes[i]
            df_class_id = df[df["Class"] == class_id]

            peaks_to_plot = ["P1","P2","P3"] if class_id != "Class4" else ["P2"]
            data = [df_class_id[df_class_id["Peak"]==p][feat].dropna() for p in peaks_to_plot]

            if not data:
                continue

            bplot = ax.boxplot(
                data,
                patch_artist=True,
                labels=peaks_to_plot,
                medianprops=dict(color="orange", linewidth=2),
                flierprops=dict(marker='o', markerfacecolor="none", markersize=5,
                                linestyle='none', markeredgecolor="teal")
            )

            # opcjonalnie kolorujemy boxy na biało z czarnymi krawędziami
            for patch in bplot['boxes']:
                patch.set(facecolor='white', edgecolor='black')

            ax.set_title(f"{class_id} — {feat}")
            ax.grid(alpha=0.3)
            ax.set_ylabel(feat)

        plt.suptitle(f"Boxplots for feature: {feat}", fontsize=16)
        plt.tight_layout(rect=[0,0,1,0.96])
        plt.show()


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
        

def plot_peak_features_boxplots_it1_it2(
    df_it1,
    df_it2,
    features=("Index", "Height", "Prominence"),
    classes=("Class1", "Class2", "Class3", "Class4"),
    colors=("lightseagreen", "lightblue"),
    fontsize_labels=15,
    fontsize_ticks=14
):
    label_map = {
        "Index": "Położenie piku [nr próbki]",
        "Height": "Amplituda",
        "Prominence": "Wyrazistość"
    }

    # umiarkowane odstępy
    base_pos = {"P1": 1.0, "P2": 2.2, "P3": 3.4}
    offset = 0.18
    width = 0.28

    for feat in features:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes = axes.flatten()

        for ax, class_id in zip(axes, classes):

            df1 = df_it1[df_it1["Class"] == class_id]
            df2 = df_it2[df_it2["Class"] == class_id]

            peaks = ["P1", "P2", "P3"] if class_id != "Class4" else ["P2"]

            data1, data2 = [], []
            pos1, pos2 = [], []

            for p in peaks:
                v1 = df1[df1["Peak"] == p][feat].dropna()
                v2 = df2[df2["Peak"] == p][feat].dropna()

                if len(v1) == 0 or len(v2) == 0:
                    continue

                data1.append(v1)
                data2.append(v2)
                pos1.append(base_pos[p] - offset)
                pos2.append(base_pos[p] + offset)

            if not data1:
                ax.set_visible(False)
                continue

            b1 = ax.boxplot(
                data1,
                positions=pos1,
                widths=width,
                patch_artist=True,
                medianprops=dict(color="black", linewidth=2),
                flierprops=dict(
                    marker='o',
                    markersize=4,
                    markerfacecolor='none',
                    markeredgewidth=1,
                    markeredgecolor=colors[0]
                )
            )

            b2 = ax.boxplot(
                data2,
                positions=pos2,
                widths=width,
                patch_artist=True,
                medianprops=dict(color="black", linewidth=2),
                flierprops=dict(
                    marker='o',
                    markersize=4,
                    markerfacecolor='none',
                    markeredgewidth=1,
                    markeredgecolor=colors[1]
                )
            )

            for box in b1["boxes"]:
                box.set_facecolor(colors[0])
                box.set_edgecolor("black")

            for box in b2["boxes"]:
                box.set_facecolor(colors[1])
                box.set_edgecolor("black")

            ax.set_xticks([base_pos[p] for p in peaks])
            ax.set_xticklabels(peaks, fontsize=fontsize_labels)

            ax.set_title(f"Klasa {class_id[-1]}", fontsize=18)
            ax.set_ylabel(label_map[feat], fontsize=fontsize_labels)
            ax.tick_params(axis='y', labelsize=fontsize_ticks)
            ax.grid(alpha=0.3)

        legend_handles = [
            Patch(facecolor=colors[0], edgecolor="black", label="Zestaw 1"),
            Patch(facecolor=colors[1], edgecolor="black", label="Zestaw 2")
        ]

        fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(1.02, 0.97),  # prawa strona figury
            ncol=1,                      # dwie linie (po jednym wpisie)
            frameon=False,
            handlelength=1.2,            # krótszy → bardziej kwadratowy
            handleheight=1.2,
            fontsize=12
        )

        fig.suptitle(
            f"{label_map[feat]} — piki referencyjne (zestaw 1 vs zestaw 2)",
            fontsize=16,
            y=0.94
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"rysunki/{feat}.pdf", format="pdf", bbox_inches='tight')
        plt.show()
        
def add_significance_stars(ax, pos1, pos2, v1, v2, df_stats, feat, cls, pk, leg_len=0.04, star_offset=0.01):
    """Dodaje prostokątny bracket o stałej wysokości nóżek nad danymi."""
    stat_row = df_stats[
        (df_stats["Feature"] == feat) & 
        (df_stats["Class"] == cls) & 
        (df_stats["Peak"] == pk)
    ]
    if stat_row.empty:
        return

    p = stat_row["p-value"].values[0]
    if p < 0.001:
        stars = "***"
    elif p < 0.01:
        stars = "**"
    elif p < 0.05:
        stars = "*"
    else:
        stars = "ns"

    # Max z uwzględnieniem outlierów
    y_max_data = max(v1.max(), v2.max())
    
    # Podniesienie bracketu nad dane
    y_bracket_top = y_max_data + (leg_len * 3) 
    y_bracket_bottom = y_bracket_top - leg_len

    # Rysowanie bracketu "square"
    ax.plot([pos1, pos1, pos2, pos2], 
            [y_bracket_bottom, y_bracket_top, y_bracket_top, y_bracket_bottom],
            color='black', linewidth=1.1)

    # Gwiazdki (niepogrubione)
    ax.text((pos1 + pos2) / 2, y_bracket_top, stars,
            ha='center', va='bottom', fontsize=13)

def plot_peak_features_boxplots_it1_it2_with_significance(
    df_it1, df_it2, df_stats,
    features=("Index", "Height", "Prominence"),
    classes=("Class1", "Class2", "Class3", "Class4"),
    colors=("lightseagreen", "lightblue")
):
    label_map = {"Index": "Położenie piku [numer próbki]", "Height": "Amplituda", "Prominence": "Wyrazistość"}
    base_pos = {"P1": 1.0, "P2": 2.2, "P3": 3.4}
    offset, width = 0.18, 0.28

    for feat in features:
        # Zwiększona szerokość figury, aby pomieścić legendę na prawo od osi
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for ax, class_id in zip(axes, classes):
            df1 = df_it1[df_it1["Class"] == class_id]
            df2 = df_it2[df_it2["Class"] == class_id]
            peaks = ["P1", "P2", "P3"] if class_id != "Class4" else ["P2"]

            data1, data2, pos1, pos2 = [], [], [], []
            all_vals = []

            for p in peaks:
                v1 = df1[df1["Peak"] == p][feat].dropna()
                v2 = df2[df2[ "Peak"] == p][feat].dropna()
                if len(v1) == 0 or len(v2) == 0: continue

                data1.append(v1)
                data2.append(v2)
                pos1.append(base_pos[p] - offset)
                pos2.append(base_pos[p] + offset)
                all_vals.extend(v1.tolist() + v2.tolist())

            if not data1:
                ax.set_visible(False)
                continue

            # Boxploty z czarną medianą
            box_props = dict(patch_artist=True, widths=width, 
                             medianprops=dict(color="black", linewidth=2.0))

            bp1 = ax.boxplot(data1, positions=pos1, **box_props,
                             flierprops=dict(marker='o', markersize=4, markeredgecolor=colors[0]))
            bp2 = ax.boxplot(data2, positions=pos2, **box_props,
                             flierprops=dict(marker='o', markersize=4, markeredgecolor=colors[1]))

            for box in bp1["boxes"]: box.set(facecolor=colors[0], edgecolor="black")
            for box in bp2["boxes"]: box.set(facecolor=colors[1], edgecolor="black")

            # Ustawienie limitów Y (+5% extra na górze względem poprzedniej wersji)
            if all_vals:
                y_min, y_max = min(all_vals), max(all_vals)
                y_range = y_max - y_min if y_max != y_min else 1
                # Zwiększony margines górny do 0.4 (40% zakresu) dla swobody bracketów
                ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.18 * y_range)
                
                dynamic_leg = 0.02 * y_range 
                for p, p1, p2, v1_b, v2_b in zip(peaks, pos1, pos2, data1, data2):
                    add_significance_stars(ax, p1, p2, v1_b, v2_b, df_stats, feat, class_id, p, leg_len=dynamic_leg)

            ax.set_xticks([base_pos[p] for p in peaks])
            ax.set_xticklabels(peaks, fontsize=16)
            ax.set_title(f"Klasa {class_id[-1]}", fontsize=20, pad=9) # Niepogrubione
            ax.set_ylabel(label_map[feat], fontsize=16)
            ax.grid(axis='y', alpha=0.3)

        # Definicja legendy
        legend_elements = [
            Patch(facecolor=colors[0], edgecolor="black", label="Zestaw 1"),
            Patch(facecolor=colors[1], edgecolor="black", label="Zestaw 2"),
            # Line2D([0], [0], color='none', label=''), 
            Line2D([0], [0], color='none', label='Test U Manna-Whitneya:', markersize=0),
            Line2D([0], [0], marker='$***$', color='black', label='p < 0.001', 
                   linestyle='none', markersize=18),
            Line2D([0], [0], marker='$**$', color='black', label='p < 0.01', 
                   linestyle='none', markersize=13),
            Line2D([0], [0], marker='$*$', color='black', label='p < 0.05', 
                   linestyle='none', markersize=7),
            Line2D([0], [0], marker='$ns$', color='black', label='p ≥ 0.05', 
                   linestyle='none', markersize=10),
        ]

        # Umieszczenie legendy całkowicie na prawo od subplotów
        leg = fig.legend(handles=legend_elements, 
                   loc='center left', 
                   bbox_to_anchor=(0.82, 0.5), # Przesunięcie na prawo
                   frameon=False, 
                   fontsize=14, 
                   handletextpad=0.8)
        
        for text in leg.get_texts():
            if text.get_text() == 'Test U Manna-Whitneya:':
                # Przesunięcie o -20 punktów w lewo (wartość dobrana do handlelength)
                text.set_position((-32, 0))
                
        
        # rect=[0, 0, 0.82, 0.95] ścieśnia wykresy do 82% szerokości, robiąc miejsce na legendę
        plt.tight_layout(rect=[0, 0, 0.82, 0.95], w_pad=2, h_pad=2) 
        plt.savefig(f"rysunki/{feat}.pdf", format="pdf", bbox_inches='tight')
        plt.show()
        
def plot_signal_with_reference_peaks(dataset, file_name):
    """
    Wykres sygnału z pikami referencyjnymi.
    Styl zgodny z wykresem prezentacyjnym.
    
    Parameters
    ----------
    dataset : list of dict
        Elementy zawierają m.in.:
        - file
        - signal (DataFrame: Sample_no, amplitude)
        - peaks_ref (dict: P1, P2, P3)
    file_name : str
        Nazwa pliku do wizualizacji
    """

    item = next(d for d in dataset if d["file"] == file_name)

    sig = item["signal"]
    t = sig.iloc[:, 0].values
    y = sig.iloc[:, 1].values

    peak_colors = {
        "P1": "red",
        "P2": "green",
        "P3": "blue"
    }

    plt.figure(figsize=(10, 4))
    plt.plot(t, y, color="black", linewidth=1.5)

    for peak, color in peak_colors.items():
        ref = item["peaks_ref"].get(peak)

        if ref is not None and not (isinstance(ref, float) and math.isnan(ref)):
            plt.plot(
                t[ref],
                y[ref],
                "o",
                color=color,
                markersize=8,
                label=f"{peak} – referencyjny"
            )

    plt.xlabel("Numer próbki")
    plt.ylabel("Amplituda")
    plt.title(file_name)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
def plot_filt_comparison(dataset_list, file_names, dataset_labels=None, colors_signal="lightseagreen"):
    """
    Porównanie sygnałów (wiersze: różne pliki, kolumny: różne wersje datasetów).
    Piki referencyjne tylko na pierwszym dataset.
    
    Parameters
    ----------
    dataset_list : list of list of dict
        Lista datasetów (np. [it1, it1_smooth_3Hz, it1_smooth_4Hz])
    file_names : list of str
        Nazwy plików do pokazania w wierszach
    dataset_labels : list of str
        Nazwy kolumn (np. ['Sygnał oryginalny', 'Po filtracji 3Hz', 'Po filtracji 4Hz'])
    colors_signal : str
        Kolor linii sygnału
    """
    n_rows = len(file_names)
    n_cols = len(dataset_list)
    
    if dataset_labels is None:
        dataset_labels = [f"Wersja {i+1}" for i in range(n_cols)]
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 3.5*n_rows), sharey=False)
    
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)
    
    peak_colors = {"P1": "red", "P2": "green", "P3": "blue"}
    
    for i, file_name in enumerate(file_names):
        for j, dataset in enumerate(dataset_list):
            ax = axes[i, j]
            item = next(d for d in dataset if d["file"] == file_name)
            sig = item["signal"]
            t = sig.iloc[:, 0].values
            y = sig.iloc[:, 1].values
            
            ax.plot(t, y, color=colors_signal, linewidth=1.45)
            
            # Piki referencyjne tylko w pierwszej kolumnie
            if j == 0:
                for peak, color in peak_colors.items():
                    ref = item["peaks_ref"].get(peak)
                    if ref is not None and not (isinstance(ref, float) and math.isnan(ref)):
                        ax.plot(
                            t[ref],
                            y[ref],
                            "o",
                            color=color,
                            markersize=7,
                            label=peak,
                            zorder=10
                        )
            
            if i == 0:
                ax.set_title(dataset_labels[j], fontsize=18, pad=14)
            if j == 0:
                ax.set_ylabel("Znormalizowana amplituda", fontsize=14, labelpad=6)
            
            ax.set_xlabel("Numer próbki", fontsize=14, labelpad=6)
            ax.grid(alpha=0.3)
            
            # legenda tylko w pierwszej kolumnie i pierwszym wierszu
            if i == 0 and j == 0:
                ax.legend(fontsize=12, loc="upper right")
            ax.set_xlim(0, 180)
            ax.set_ylim(0,1)
            ax.tick_params(axis='both', which='major', labelsize=11)
    
    plt.tight_layout(w_pad=1.5, h_pad=1.4)
            
    
    
    plt.savefig("rysunki/filtracja2.pdf", format="pdf", bbox_inches=None)
    plt.show()
    
def is_true_positive(item, peak, tolerance=3):
    """
    Sprawdza, czy dany pik jest TRUE POSITIVE:
    - wykryty (nie NaN)
    - istnieje referencja
    - |detected - ref| <= tolerance
    """
    det = item["peaks_detected"].get(peak)
    ref = item["peaks_ref"].get(peak)

    if det is None or ref is None:
        return False

    if isinstance(det, float) and math.isnan(det):
        return False
    if isinstance(ref, float) and math.isnan(ref):
        return False

    return abs(det - ref) <= tolerance

def plot_upset_classic_postproc_new(results_pp, class_name, tolerance=3):
    rows = []

    for r in results_pp:
        if r["class"] != class_name:
            continue

        rows.append({
            "P1": is_true_positive(r, "P1", tolerance),
            "P2": is_true_positive(r, "P2", tolerance),
            "P3": is_true_positive(r, "P3", tolerance),
        })

    df = pd.DataFrame(rows)

    series = df.value_counts().sort_index()

    total = 250 if class_name != "Class4" else 50
    perc = series / total * 100

    plt.figure(figsize=(6, 4))
    u = UpSet(
        perc,
        show_counts=True,
        sort_by="cardinality",
        sort_categories_by="-input"
    ).plot()

    plt.suptitle(f"Klasa {class_name[-1]}", fontsize=12)

    for ax in plt.gcf().axes:
        if ax.get_ylabel() == "Intersection size":
            ax.set_ylabel("Udział sygnałów [%]")

    u['totals'].set_xlabel("Wykryte piki [%]", fontsize=10)
    plt.tight_layout()
    # plt.show()


def plot_peak_detection_pie_new(results_combined, class_id, peak="P2", tolerance=3):
    class_results = [d for d in results_combined if d["class"] == class_id]

    tp_count = sum(
        1 for d in class_results
        if is_true_positive(d, peak, tolerance)
    )

    total = len(class_results)
    fn_count = total - tp_count

    labels = [f"{peak} wykryty", f"{peak} niewykryty"]
    sizes = [tp_count, fn_count]
    colors = ["lightblue", "lightseagreen"]

    plt.figure(figsize=(6,6))
    plt.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.0f%%",
        startangle=90
    )
    plt.title(f"Klasa {class_id[-1]}", fontsize=14)
    # plt.show()


def plot_all_signals_with_peaks(
    detection_results,
    method_name,
    ranges_name,
    classes=("Class1", "Class2", "Class3", "Class4")
):
    """
    Grid: wiersze = klasy, kolumny = typy pików
    - wszystkie sygnały
    - średni sygnał
    - histogram gęstości pików referencyjnych (green)
    - histogram gęstości pików automatycznych (red)
    """

    fig, axes = plt.subplots(2,2, figsize=(19, 12), squeeze=False, 
                             gridspec_kw={"wspace":0.3, "hspace":0.3})
    axes = axes.flatten()
    
    titles_dict = {
        "concave": "Maksima w odcinkach wklęsłych",
        "concave_tuned": "Maksima w odcinkach wklęsłych (+parametry find_peaks)",
        "concave_d2x=-0-002": "Maksima w odcinkach wklęsłych (próg 2. pochodnej: -0,002)",
        "concave_d2x=0-002": "Maksima w odcinkach wklęsłych (próg 2. pochodnej: 0,002)",
        "modified_scholkmann0-5": "Zmodyfikowany algorytm Scholkmanna (limit=0.5)",
        "modified_scholkmann1": "Zmodyfikowany algorytm Scholkmanna (limit=1.0)",
        "curvature": "Maksymalna krzywizna",
        "line_distance_8": "Odległość od linii",
        "line_perpendicular_8": "Odległość prostopadła do linii",
        "hilbert": "Transformata Hilberta",
        "wavelet": "Ciągła transformata falkowa"
    }
    
    ranges_dict = {
        "full": "min-max",
        "avg": "na podstawie średniej ruchomej",
        "whiskers": "wąsy"}

    for i, class_name in enumerate(classes):
        # wybieramy tylko dane tej klasy
        class_items = [
            d for d in detection_results
            if d["class"] == class_name
            and d["method"] == method_name
            and (ranges_name is None or d["ranges"] == ranges_name)
        ]
        
        ax = axes[i]

        if not class_items:
            ax.set_visible(False)
            continue

        t = class_items[0]["signal"].iloc[:, 0].values
        all_x = [item["signal"].iloc[:, 1].values for item in class_items]
        mean_signal = np.mean(all_x, axis=0)
        
        for item in class_items:
            ax.plot(t, item["signal"].iloc[:, 1].values, color="black", alpha=0.07, linewidth=0.5, zorder=12)

        # średni sygnał
        ax.plot(t, mean_signal, color="mediumblue", linewidth=2.6, label="Średni sygnał", zorder=15)

        # zbieramy wszystkie piki (bez podziału na typy)
        manual_t = []
        auto_t = []
        for item in class_items:
            if item.get("peaks_ref") is not None:
                for p in item["peaks_ref"].values():
                    if p is not None and not np.isnan(p):
                        manual_t.append(t[int(p)])
            if item.get("peaks_detected") is not None:
                for plist in item["peaks_detected"].values():
                    auto_t.extend([t[int(p)] for p in plist])

        # histogramy pików
        # ax_hist = ax.twinx()
        # if manual_t:
        #     ax_hist.hist(manual_t, bins=40, alpha=0.35, color="green", label="Piki referencyjne")
        # if auto_t:
        #     ax_hist.hist(auto_t, bins=40, alpha=0.35, color="red", label="Piki wykryte")
        # # ax_hist.get_yaxis().set_visible(False)
        
        # 1. Definiujemy stałe krawędzie binów od 0 do 180 (zgodnie z set_xlim)
        # np. co 4 próbki (180 / 45 = 4)
        bin_edges = np.linspace(0, 180, 92) 

        # 2. Rysujemy histogramy używając tych samych krawędzi
        ax_hist = ax.twinx()
        ax_hist.set_zorder(0)
        ax.set_zorder(1)
        ax.patch.set_visible(False)
        if manual_t:
            ax_hist.hist(manual_t, bins=bin_edges, alpha=0.37, 
                         color="green", label="Piki referencyjne", zorder=11)
        if auto_t:
            ax_hist.hist(auto_t, bins=bin_edges, alpha=0.35, 
                         color="red", label="Piki wykryte", zorder=10)
        
        ax.set_xlim(0, 180)
        ax.set_ylim(bottom=0)
        
        ax_hist.set_ylim(bottom=0)
        
        ax.margins(x=0, y=0)
        ax_hist.margins(y=0)
        
        ax.set_title(f"Klasa {class_name[-1]}", fontsize=23, pad=8)
        ax.set_xlabel("Numer próbki", fontsize=17, labelpad=6)
        ax.set_ylabel("Znormalizowana amplituda", fontsize=17, labelpad=6)
        ax_hist.set_ylabel("Liczba pików", fontsize=17, labelpad=6)
        ax.tick_params(axis='both', labelsize=14)
        ax.grid(alpha=0.3)

        # h1, l1 = ax.get_legend_handles_labels()
        # h2, l2 = ax_hist.get_legend_handles_labels()
        # ax.legend(h1 + h2, l1 + l2, loc="upper right")
        
    #     ax.set_title(f"Klasa: {class_name} (N={len(class_items)})", fontsize=16)
    #     ax.set_xlabel("Numer próbki")
    #     ax.set_ylabel("Amplituda")

    # # globalna legenda
    # handles, labels = [], []
    # for ax in axes:
    #     h, l = ax.get_legend_handles_labels()
    #     handles += h
    #     labels += l
    # fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)

    # fig.suptitle(f"{titles_dict.get(method_name, method_name)}", fontsize=20, y=0.96)
    plt.tight_layout()
    # plt.show()


def plot_two_signals_with_peaks_crossings(
    dataset,
    filenames=("Class2_example_0042", "Class2_example_0046"),
    # colors=("lightblue", "lightseagreen"),
    window_fast=2,
    window_slow=12
):
    # crossingi
    crossings = compute_crossings(dataset)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)

    for ax, fname in zip(axes, filenames):
        item = next(d for d in dataset if d["file"] == fname)
        sig_df = item["signal"]

        t = sig_df.iloc[:, 0].values
        y = sig_df.iloc[:, 1].values
        class_id = item["class"]

        # średnie ruchome
        avg_fast = pd.Series(y).rolling(
            window_fast, min_periods=1, center=True
        ).mean().to_numpy()

        avg_slow = pd.Series(y).rolling(
            window_slow, min_periods=1, center=True
        ).mean().to_numpy()

        # wykres sygnału
        ax.plot(t, y, color="black", linewidth=1, label="Sygnał")

        # średnie
        ax.plot(t, avg_fast, "--", color="lightskyblue", linewidth=1,
                label="Średnia szybka \n (szerokość okna: 2 próbki)")
        ax.plot(t, avg_slow, "--", color="lightseagreen", linewidth=1,
                label="Średnia wolna \n (szerokość okna: 4 próbki)")

        # piki referencyjne
        for p_name, idx in item["peaks_ref"].items():
            ax.scatter(
                t[idx], y[idx],
                color="black", zorder=6
            )
            ax.text(
                t[idx], y[idx]+0.05,
                p_name, fontsize=10,
                ha="center", va="bottom"
            )

        # crossingi
        cross_dict = crossings[class_id][fname]
        first = True
        for p_name, inds in cross_dict.items():
            for i in inds:
                ax.axvline(
                    t[i],
                    color="mediumpurple",
                    # linestyle="--",
                    alpha=0.82,
                    label="Położenie punktów \n przecięcia średnich" if first else None
                )
                first = False

        # ax.set_title(fname)
        ax.set_xlabel("Numer próbki", fontsize=12)
        ax.grid(alpha=0.3)
        y_min, y_max = ax.get_ylim()
        margin = 0.05 * (y_max - y_min)   # ~8% wysokości osi
        ax.set_ylim(0, y_max + margin)
        ax.set_xlim(0, 180)

    axes[0].set_ylabel("Amplituda",  fontsize=12)
    axes[1].legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    # plt.show()
    
def plot_two_signals_with_crossings_binary(
    dataset,
    filenames=("Class2_example_0042", "Class2_example_0046"),
    window_fast=2,
    window_slow=4, # 4!!!!!!!!
    y_scale=0.9  # wartość sygnału prostokątnego
):
    """
    Wykres sygnału ICP z zaznaczonymi punktami przecięcia średnich w formie prostokątnej.
    Zamiast dwóch średnich ruchomych pokazuje sygnał binarny: 0 = fast <= slow, y_scale = fast > slow
    """
    crossings = compute_ranges_avg(dataset)  # korzystamy z Twojej funkcji bazowej

    fig, axes = plt.subplots(1, len(filenames), figsize=(15, 5), sharey=True)

    if len(filenames) == 1:
        axes = [axes]  # ułatwienie dla jednej osi

    for ax, fname in zip(axes, filenames):
        item = next(d for d in dataset if d["file"] == fname)
        sig_df = item["signal"]

        t = sig_df.iloc[:, 0].values
        y = sig_df.iloc[:, 1].values
        class_id = item["class"]

        # średnie ruchome
        avg_fast = pd.Series(y).rolling(window_fast, min_periods=1, center=True).mean().to_numpy()
        avg_slow = pd.Series(y).rolling(window_slow, min_periods=1, center=True).mean().to_numpy()

        # sygnał binarny (fast > slow)
        binary_signal = np.zeros_like(y)
        binary_signal[avg_fast > avg_slow] = y_scale
        binary_signal += 0.2

        # wykres prostokątny
        ax.step(t, binary_signal, "--", where="mid", color="mediumpurple", linewidth=1.1, label="Fast MA > Slow MA")

        # wykres sygnału ICP
        ax.plot(t, y, color="black", linewidth=1.6, label="Sygnał ICP")
        
        # ax.plot(t, avg_fast, "--", color="lightskyblue", linewidth=1,
        #         label="Średnia szybka \n (szerokość okna: 2 próbki)")
        # ax.plot(t, avg_slow, "--", color="lightseagreen", linewidth=1,
        #         label="Średnia wolna \n (szerokość okna: 4 próbki)")


        # piki referencyjne
        for p_name, idx in item["peaks_ref"].items():
            ax.scatter(
                t[idx], y[idx],
                color="black", zorder=6, s=70
            )
            ax.text(
                t[idx], y[idx]+0.05,
                p_name, fontsize=16,
                ha="center", va="bottom"
            )

        # crossingi
        cross_dict = crossings[class_id][fname]
        first = True
        for p_name, inds in cross_dict.items():
            for i in inds:
                ax.axvline(
                    t[i],
                    color="lightseagreen",
                    # linestyle="--",
                    alpha=0.82,
                    label="Wykryte punkty \n przecięcia średnich" if first else None
                )
                first = False
        # crossingi jako prostokąty
        cross_dict = crossings[class_id][fname]
        # tworzymy listę wszystkich crossingów w kolejności
        all_crossings = []
        for inds in cross_dict.values():
            all_crossings.extend(inds)
        all_crossings = sorted(all_crossings)
        
        # kolory i alpha
        rect_color = "lightgreen"
        rect_alpha = 0.3
        
        # rysujemy prostokąty między punktami: 1-2, 3-4, 5-6
        for i in range(0, len(all_crossings), 2):
            if i+1 < len(all_crossings):
                start = t[all_crossings[i]]
                end = t[all_crossings[i+1]]
                ax.axvspan(start, end, color=rect_color, alpha=rect_alpha)

        ax.set_xlabel("Numer próbki", fontsize=18)
        ax.tick_params(axis='both', labelsize=14)
        ax.grid(alpha=0.3)
        y_min, y_max = ax.get_ylim()
        margin = 0.05 * (y_max - y_min)
        ax.set_ylim(0, y_max + margin)
        ax.set_xlim(0, 180)

    axes[0].set_ylabel("Znormalizowana amplituda", fontsize=18)
    # axes[-1].legend(loc="upper right", fontsize=9)
    plt.tight_layout(w_pad=1.8)

def plot_column_crossing():
    import glob
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # =========================
    # WCZYTANIE DANYCH
    # =========================
    all_files = glob.glob("crossings_compare/*.csv")
    df_list = [pd.read_csv(f, delimiter=",") for f in all_files]
    df_all = pd.concat(df_list, ignore_index=True)

    dataset_order = [
        "it1",
        "it1_smooth_4Hz",
        "it1_smooth_3Hz",
    ]

    df_all["Dataset"] = pd.Categorical(
        df_all["Dataset"],
        categories=dataset_order,
        ordered=True
    )

    metric = "%_Signals_with_6_crossings_and_ideal_P1P3"

    datasets = [
        ("it1", "brak filtracji"),
        ("it1_smooth_4Hz", "filtracja\n$f_g = 4\\,\\mathrm{Hz}$"),
        ("it1_smooth_3Hz", "filtracja\n$f_g = 3\\,\\mathrm{Hz}$"),
    ]

    classes = ["Class1", "Class2", "Class3"]

    class_colors = {
        "Class1": "lightblue",
        "Class2": "lightseagreen",
        "Class3": "mediumpurple",
    }

    class_map = {
        "Class1": "Klasa 1",
        "Class2": "Klasa 2",
        "Class3": "Klasa 3"
    }

    titles = [
        "min_distance = 0",
        "min_distance = 3 (Klasy 1 i 3)\nmin_distance = 5 (Klasa 2)",
        "min_distance = 5 (Klasy 1 i 3)\nmin_distance = 10 (Klasa 2)",
        "min_distance = 8 (Klasy 1 i 3)\nmin_distance = 12 (Klasa 2)"
    ]

    min_distances = sorted(df_all["min_distance"].unique())

    # =========================
    # PARAMETRY WYGLĄDU
    # =========================
    FS_TITLE = 16
    FS_AXIS = 16
    FS_TICK = 16
    FS_LEG = 15

    bar_width = 0.22
    inner_gap = 0.06
    group_gap = 0.25
    step = bar_width + inner_gap

    x_groups = np.arange(len(datasets)) * (len(classes) * step + group_gap)

    # =========================
    # FIGURA 2×2
    # =========================
    fig, axes = plt.subplots(
        2, 2,
        figsize=(12.5, 9),
        sharey="row"
    )

    # =========================
    # RYSOWANIE
    # =========================
    for plot_idx, (ax, md) in enumerate(zip(axes.flatten(), min_distances)):
        df_md = df_all[df_all["min_distance"] == md]

        for cls_idx, cls in enumerate(classes):
            values = []
            for ds, _ in datasets:
                row = df_md[
                    (df_md["Dataset"] == ds) &
                    (df_md["Class"] == cls)
                ]
                values.append(
                    row[metric].values[0] if not row.empty else np.nan
                )

            ax.bar(
                x_groups + cls_idx * step,
                values,
                width=bar_width,
                color=class_colors[cls],
                label=class_map[cls]
            )

        ax.set_title(titles[plot_idx], fontsize=FS_TITLE, pad=10)
        ax.set_xticks(x_groups + step)
        ax.set_xticklabels(
            [lbl for _, lbl in datasets],
            fontsize=FS_TICK
        )
        ax.tick_params(axis="y", labelsize=FS_TICK)
        ax.grid(axis="y", alpha=0.3)

    # =========================
    # PODPISY OSI Y – KAŻDY WIERSZ
    # =========================
    for row in range(2):
        axes[row, 0].set_ylabel(
            r"$D_P$ [$\%$]",
            fontsize=FS_AXIS
        )

    # =========================
    # LEGENDA – POZA WYKRESAMI
    # =========================
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="center left",
        bbox_to_anchor=(0.9, 0.5),
        fontsize=FS_LEG,
        frameon=True
    )

    # =========================
    # FINAL
    # =========================
    plt.tight_layout(rect=[0, 0, 0.9, 1], w_pad=1.2, h_pad=1.8)
    plt.savefig("rysunki/crossings_min_len.pdf", format="pdf", bbox_inches="tight")
    plt.show()



def plot_column_crossing_old():
    all_files = glob.glob("crossings_compare/*.csv") 
    df_list = []
    
    for f in all_files:
        df = pd.read_csv(f, delimiter=",")
        df_list.append(df)
    
    # 2. Połącz w jeden DataFrame
    df_all = pd.concat(df_list, ignore_index=True)
    
    dataset_order = [
    "it1",
    "it1_smooth_4Hz",
    "it1_smooth_3Hz",
    ]
    
    df_all["Dataset"] = pd.Categorical(
        df_all["Dataset"],
        categories=dataset_order,
        ordered=True
    )
    
    # 3. Wersja kolumnowa – wykres porównawczy dla Class i min_distance
    metric = "%_Signals_with_6_crossings_and_ideal_P1P3"

    datasets = [
    ("it1", "brak filtracji"),
    ("it1_smooth_4Hz", "filtracja\n$f_g = 4\\,\\mathrm{Hz}$"),
    ("it1_smooth_3Hz", "filtracja\n$f_g = 3\\,\\mathrm{Hz}$"),
    ]
    
    classes = ["Class1", "Class2", "Class3"]
    
    class_colors = {
        "Class1": "lightblue",
        "Class2": "lightseagreen",
        "Class3": "mediumpurple",
    }
        
    class_map = {
    "Class1": "Klasa 1",
    "Class2": "Klasa 2",
    "Class3": "Klasa 3"
    }
    
    # Twoje specyficzne tytuły dla kolejnych wykresów
    titles = [
        "min_distance=0",
        "min_distance=3 (Klasy: 1, 3)\nmin_distance=5 (Klasa 2)",
        "min_distance=5 (Klasy: 1, 3)\nmin_distance=10 (Klasa 2)",
        "min_distance=8 (Klasy: 1, 3)\nmin_distance=12 (Klasa 2)"
    ]
    min_distances = sorted(df_all["min_distance"].unique())
    
    fig, axes = plt.subplots(
        1, len(min_distances),
        figsize=(4 * len(min_distances), 4),
        sharey=True
    )
    
    if len(min_distances) == 1:
        axes = [axes]
    
    group_gap = 0.2      # Odstęp między całymi grupami
    bar_width = 0.2      # Szerokość słupka
    inner_gap = 0.05      # <--- NOWOŚĆ: Odstęp między słupkami w grupie
    
    # Obliczamy całkowity krok wewnątrz grupy (szerokość słupka + odstęp)
    step = bar_width + inner_gap
    
    # x_groups musi teraz uwzględniać szerokość wszystkich słupków i przerw między nimi
    x_groups = np.arange(len(datasets)) * (len(classes) * step + group_gap)
    
    for plot_idx, (ax, md) in enumerate(zip(axes, min_distances)):
        df_md = df_all[df_all["min_distance"] == md]
    
        for idx, cls in enumerate(classes):
            values = []
            for ds, _ in datasets:
                row = df_md[(df_md["Dataset"] == ds) & (df_md["Class"] == cls)]
                values.append(row[metric].values[0] if not row.empty else np.nan)
    
            ax.bar(
                x_groups + idx * step,
                values,
                width=bar_width,
                color=class_colors[cls],
                label=class_map[cls]  # Użycie polskiej nazwy
            )
    
        # Ustawienie tytułu z listy (jeśli mamy zdefiniowany dla tego indeksu)
        if plot_idx < len(titles):
            ax.set_title(titles[plot_idx], fontsize=12, pad=10)
    
        ax.set_xticks(x_groups + step)
        ax.set_xticklabels([lbl for _, lbl in datasets], rotation=0, fontsize=12)
        ax.grid(axis="y", alpha=0.3)
    
    axes[0].set_ylabel("$D_p$ [%]", fontsize=12)
    
    # Legenda po prawej stronie, pionowa
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="center left",
        bbox_to_anchor=(0.90, 0.5), # Pozycja poza obszarem wykresu
        ncol=1,                    # Jeden pod drugim
        frameon=True
    )
    
    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Zostawienie miejsca po prawej na legendę
    plt.savefig("rysunki/crossings_min_len.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    
    

from scipy.signal import find_peaks, peak_prominences


def plot_threshold_prominence(
    dataset,
    idx=0,
    threshold_val=0.07,
    prominence_val=0.08,
    threshold_x_offset=12,
    prominence_x_offset=7
    ):
    """
    Schematyczna ilustracja threshold i prominence
    dla pików referencyjnych P1, P2, P3.
    """

    item = dataset[idx]
    sig = item["signal"]
    peaks_ref = item["peaks_ref"]

    x = sig["Sample_no"].values
    y = sig["ICP"].values

    fig, ax = plt.subplots(figsize=(8.2, 5))
    ax.plot(x, y, color="black", lw=1.5)
    
        # 1. Wyliczamy prominence dla WSZYSTKICH punktów, które są lokalnymi maksimami
    # Musimy podać listę indeksów naszych pików referencyjnych (P1, P2, P3)
    peak_indices = [p for p in peaks_ref.values() if p is not None]
    
    # peak_prominences zwraca: (prominences, left_bases, right_bases)
    all_proms, left_bases, right_bases = peak_prominences(y, peak_indices)

    # Mapujemy wyniki z powrotem do nazw pików
    prom_results = dict(zip(peaks_ref.keys(), zip(all_proms, left_bases, right_bases)))


    colors = {
        "P1": "black",
        "P2": "black",
        "P3": "black"
    }

    for name, p in peaks_ref.items():
        if p is None:
            continue

        # Pobieramy wyniki z SciPy dla konkretnego piku
        prom, left_base, right_base = prom_results[name]
        
        # Poziom bazowy (contour line) to y najniższego punktu w wyznaczonym przedziale
        # SciPy wyznacza go jako wyższy z dwóch lokalnych minimów (left/right base)
        base_level = y[p] - prom
        
        # --- RYSOWANIE PROMINENCE ---
        x_prom = x[p] + prominence_x_offset
        
        # Linia konturowa (od lewej do prawej bazy wyznaczonej przez SciPy)
        ax.hlines(
            base_level,
            x_prom-22,
            x_prom+22,
            linestyles="dashed", colors="lightseagreen",
            alpha=0.5
        )
        
        ax.hlines(
            y[p],
            x[p],
            x_prom,
            linestyles="dashdot", colors="lightseagreen",
            alpha=0.7
        )
        
        ax.annotate(
            "",
            xy=(x_prom, y[p]),
            xytext=(x_prom, base_level),
            arrowprops=dict(arrowstyle="<->", 
                            lw=2, 
                            color="lightseagreen",
                            mutation_scale=12,       # Powiększa sam grot strzałki
                            shrinkA=0,               # Strzałka dotyka dokładnie punktu xy
                            shrinkB=0  ,
                            zorder=10)              # Strzałka dotyka dokładnie punktu xyte)
        )

        # --- THRESHOLD (przesuniety w bok) ---
        
        thr_left = y[p] - y[p - 1]
        thr_right = y[p] - y[p + 1]
        threshold_real = min(thr_left, thr_right) + 0.0525
        
        x_thr = x[p] + threshold_x_offset
        # linie przerywane pokazujące zakres threshold
        ax.hlines(
            [y[p], y[p] - threshold_real],
            x[p],
            x_thr,
            linestyles="dotted",
            colors="mediumpurple",
            alpha=0.8
        )

        ax.annotate(
            "",
            xy=(x_thr, y[p]),
            xytext=(x_thr, y[p] - threshold_real),
            arrowprops=dict(
                arrowstyle="<->",
                lw=2,
                color="purple",
                mutation_scale=12,       # Powiększa sam grot strzałki
                shrinkA=0,               # Strzałka dotyka dokładnie punktu xy
                shrinkB=0  ,
                zorder=10)  
        )
        
        # punkt piku
        ax.scatter(x[p], y[p], color="black", zorder=4)

        # etykieta P1/P2/P3
        ax.text(
            x[p],
            y[p] + 0.02,
            name,
            ha="center",
            va="bottom",
            fontsize=15,
            # fontweight="bold",
            color=colors.get(name, "black"),
            zorder=10
        )
        
        
        height_val = y[p]
        ax.annotate(
            "",
            xy=(x[p], y[p]),
            xytext=(x[p], 0),
            arrowprops=dict(arrowstyle="<->", 
                            lw=2, color="dimgray",
                            mutation_scale=12,       # Powiększa sam grot strzałki
                            shrinkA=0,               # Strzałka dotyka dokładnie punktu xy
                            shrinkB=0  ,
                            ), zorder=20  
        )

        
        
        # 4. ETYKIETY TEKSTOWE (tylko dla P1)
        if name == "P3":
            # ax.text(x_prom + 3, (y[p] + base_level)/2, "prominence", color="lightseagreen", #rotation=90, 
            #         va="center", fontsize=16, zorder=10)
            # ax.text(x_thr + 3, (y[p] + (y[p]-threshold_real))/2, "threshold", color="purple", 
            #         va="center", fontsize=16)
            # ax.text(x[p] - 10, height_val/2+0.12, "height", color="dimgray", #rotation=90, 
            #         ha="right", 
            #         va="center", fontsize=16)
            
            ax.text(x_prom + 14, (y[p] + base_level)/2-0.01, "prominence", color="lightseagreen", #rotation=90, 
                    va="center", fontsize=18, zorder=10)
            
            ax.text(x_thr + 4, (y[p] + (y[p]-threshold_real))/2, "threshold", color="purple", 
                    va="center", fontsize=18)
            
            ax.text(x[p]+22, height_val/2+0.035, "height", color="dimgray", #rotation=90, 
                    ha="right", 
                    va="center", fontsize=18)
    
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.4) # Pogrubienie ramki
    # ax.axis('off')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, 180)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    # plt.show()


def plot_concave_threshold_comparison(dataset, filename):
    # 1. Pobieranie danych
    item = next((d for d in dataset if d["file"] == filename), None)
    if item is None: return
    
    sig_df = item["signal"]
    t = sig_df.iloc[:, 0].values
    y = sig_df.iloc[:, 1].values

    # 2. Obliczanie 2. pochodnej
    dy = np.gradient(y, edge_order=2)
    d2y = np.gradient(dy, edge_order=2)

    # 3. Definicje progów
    thresholds = [0, -0.002, 0.002]
    labels = [
        "Wariant bazowy, $f'' < 0$",
        "$f'' < -0.002$",
        "$f'' < 0.002$"
    ]
    colors = ['lightblue', 'lightblue', 'lightblue']

    fig, axs = plt.subplots(3, 1, figsize=(10, 10.4), sharex=True, sharey=True)

    def get_regions(mask):
        mask_diff = np.diff(mask.astype(int))
        starts = np.where(mask_diff == 1)[0] + 1
        ends = np.where(mask_diff == -1)[0]
        if mask[0]: starts = np.insert(starts, 0, 0)
        if mask[-1]: ends = np.append(ends, len(mask)-1)
        return list(zip(starts, ends))

    for i, th in enumerate(thresholds):
        ax = axs[i]
        mask = d2y < th
        regions = get_regions(mask)
        
        ax.plot(t, y, color='black', lw=1.4, alpha=0.9)
        
        for start, end in regions:
            ax.axvspan(t[start], t[end], color=colors[i], alpha=0.6)
            
        ax.set_title(f"{labels[i]}", fontsize=22, pad=10)
        ax.tick_params(axis='both', labelsize=14)
        if i==1:
            ax.set_ylabel("Znormalizowana amplituda", fontsize=18)
        else:
            ax.set_ylabel("")

    axs[2].set_xlabel("Numer próbki", fontsize=18)
    
    ax.set_xlim(0, 180)
    ax.set_ylim(bottom=0)
    plt.tight_layout(w_pad=1.98)
    # plt.show()

# Wywołanie:
# plot_concave_threshold_comparison(dataset, "twoj_plik.csv")


import string
import matplotlib.gridspec as gridspec


def plot_punkty_charakterystyczne_old2(datasets, filenames, figsize_per_plot=(5, 4)):
    n = len(filenames)
    panel_labels = list(string.ascii_lowercase)
    
    fig = plt.figure(figsize=(figsize_per_plot[0] * min(n, 3), 
                              figsize_per_plot[1] * (2 if n > 3 else 1)))
    
    axes = []
    if n == 3:
        gs = gridspec.GridSpec(1, 3, figure=fig)
        for i in range(3):
            axes.append(fig.add_subplot(gs[0, i]))
    elif n == 4:
        gs = gridspec.GridSpec(2, 2, figure=fig)
        for i in range(4):
            axes.append(fig.add_subplot(gs[i // 2, i % 2]))
    elif n == 5:
        # Siatka 2x6 pozwala na 3 wykresy (szer. 2) i 2 wykresy (szer. 3 lub szer. 2 z przesunięciem)
        gs = gridspec.GridSpec(2, 6, figure=fig)
        # Rząd 1: 3 wykresy, każdy zajmuje 2 kolumny
        for i in range(3):
            axes.append(fig.add_subplot(gs[0, i*2 : i*2+2]))
        # Rząd 2: 2 wykresy wyśrodkowane (każdy szer. 2, zaczynamy od kolumny 1)
        axes.append(fig.add_subplot(gs[1, 1:3]))
        axes.append(fig.add_subplot(gs[1, 3:5]))
    else:
        # Fallback dla innych n
        ncols = min(n, 3)
        nrows = (n + ncols - 1) // ncols
        gs = gridspec.GridSpec(nrows, ncols, figure=fig)
        for i in range(n):
            axes.append(fig.add_subplot(gs[i // ncols, i % ncols]))

    # --- Reszta logiki (identyczna) ---
    def find_file_in_datasets(fname):
        for ds in datasets:
            for item in ds:
                if item["file"] == fname: return item
        raise ValueError(f"Plik '{fname}' nie znaleziony.")

    peak_cfg = {"P1": ("red", "P1"), "P2": ("green", "P2"), "P3": ("blue", "P3")}

    for i, (ax, fname, label) in enumerate(zip(axes, filenames, panel_labels)):
        item = find_file_in_datasets(fname)
        signal = item["signal"]
        peaks_ref = item.get("peaks_ref", {})
        x, y = signal.iloc[:, 0].values, signal.iloc[:, 1].values

        ax.plot(x, y, color="black", linewidth=1.5)

        for key, (color, txt) in peak_cfg.items():
            val = peaks_ref.get(key, None)
            if val is not None:
                idx = np.atleast_1d(val)
                for pt in map(int, idx):
                    ax.plot(x[pt], y[pt], "o", color=color, markersize=9)
                    ax.annotate(txt, (x[pt], y[pt]), xytext=(0, 8), textcoords="offset points",
                                ha="center", va="bottom", fontsize=14)

        ax.set_xlabel("Numer próbki", fontsize=14)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0, 180)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.text(0.04, 1, f"({label})", transform=ax.transAxes, fontweight="bold", va="top",
                fontsize=14)
        
        # Logika etykiety Y dostosowana do układów
        if n == 4:
            if i % 2 == 0: ax.set_ylabel("Znormalizowana amplituda", fontsize=14)
        else: # Dla n=3 i n=5 (lewa krawędź)
            if i == 0 or (n == 5 and i == 3):
                ax.set_ylabel("Znormalizowana amplituda", fontsize=14)

    fig.tight_layout()
    return fig, axes

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import string

def plot_punkty_charakterystyczne(datasets, filenames, figsize_per_plot=(5.2, 4)):
    n = len(filenames)
    panel_labels = list(string.ascii_lowercase)
    
    # Stała szerokość figury odpowiadająca 3 kolumnom.
    # Wysokość: 1 wiersz dla n<=3, 2 wiersze dla n>3.
    fig = plt.figure(figsize=(figsize_per_plot[0] * 3, 
                              figsize_per_plot[1] * (2 if n > 3 else 1)))
    
    axes = []
    if n <= 3:
        # Standardowy układ 1x3
        gs = gridspec.GridSpec(1, 3, figure=fig)
        for i in range(n):
            axes.append(fig.add_subplot(gs[0, i]))
            
    elif n == 4:
        # Układ 2x2, ale osadzony w siatce 2x3, aby zachować szerokość kolumn.
        # Kolumna o indeksie 2 w każdym wierszu pozostaje pusta.
        gs = gridspec.GridSpec(2, 3, figure=fig)
        for i in range(4):
            # i // 2 daje wiersz (0,1), i % 2 daje kolumnę (0,1)
            axes.append(fig.add_subplot(gs[i // 2, i % 2]))
            
    elif n == 5:
        # Wykorzystujemy siatkę 6-kolumnową dla idealnego wyśrodkowania.
        # Szerokość 2 jednostek w siatce 6-kolumnowej == szerokość 1 wykresu w siatce 3-kolumnowej.
        gs = gridspec.GridSpec(2, 6, figure=fig)
        # Rząd 1: 3 wykresy (kolumny: 0-2, 2-4, 4-6)
        for i in range(3):
            axes.append(fig.add_subplot(gs[0, i*2 : i*2+2]))
        # Rząd 2: 2 wykresy wyśrodkowane (kolumny: 1-3, 3-5)
        # Dzięki temu pod (a)(b)(c) mamy wyśrodkowane (d)(e) o TEJ SAMEJ szerokości.
        axes.append(fig.add_subplot(gs[1, 1:3]))
        axes.append(fig.add_subplot(gs[1, 3:5]))

    # --- Logika wyszukiwania danych ---
    def find_file_in_datasets(fname):
        for ds in datasets:
            for item in ds:
                if item["file"] == fname: return item
        raise ValueError(f"Plik '{fname}' nie znaleziony.")

    peak_cfg = {"P1": ("red", "P1"), "P2": ("green", "P2"), "P3": ("blue", "P3")}

    # --- Pętla rysująca ---
    for i, (ax, fname, label) in enumerate(zip(axes, filenames, panel_labels)):
        item = find_file_in_datasets(fname)
        signal = item["signal"]
        peaks_ref = item.get("peaks_ref", {})
        x, y = signal.iloc[:, 0].values, signal.iloc[:, 1].values

        # Rysowanie przebiegu
        ax.plot(x, y, color="black", linewidth=1.5)

        # Nakładanie punktów referencyjnych
        for key, (color, txt) in peak_cfg.items():
            val = peaks_ref.get(key, None)
            if val is not None:
                idx = np.atleast_1d(val)
                for pt in map(int, idx):
                    ax.plot(x[pt], y[pt], "o", color=color, markersize=10)
                    ax.annotate(txt, (x[pt], y[pt]+0.02), xytext=(0, 8), textcoords="offset points",
                                ha="center", va="bottom", fontsize=14)

        # Formatowanie osi i czcionek
        ax.set_xlabel("Numer próbki", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0, 180)
        ax.set_ylim(bottom=0)

        # Ustawienie etykiety panelu (a), (b)...
        # x=-0.05 przesuwa lekko przed oś, y=1.05 lekko nad ramkę
        ax.text(0.032, 1, f"({label})", transform=ax.transAxes, 
                fontweight="bold", va="bottom", ha="left", fontsize=14)
        
        # Logika opisu osi Y - tylko dla wykresów na lewej krawędzi aktualnego wiersza
        if n == 4:
            if i % 2 == 0: ax.set_ylabel("Znormalizowana amplituda", fontsize=14)
        elif n == 5:
            # i=0 to (a), i=3 to (d) - oba są na lewej krawędzi swoich rzędów (dla n=5)
            if i == 0 or i == 3: ax.set_ylabel("Znormalizowana amplituda", fontsize=14)
        else:
            if i % 3 == 0: ax.set_ylabel("Znormalizowana amplituda", fontsize=14)

    # tight_layout z parametrem rect zapobiega ucinaniu przesuniętych liter (a,b,c)
    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
    return fig, axes

def plot_punkty_charakterystyczne_old(datasets, filenames, figsize_per_plot=(5, 4)):
    n = len(filenames)
    panel_labels = list(string.ascii_lowercase)
    
    if n <= 3:
        fig, axes = plt.subplots(1, n, figsize=(figsize_per_plot[0]*n, figsize_per_plot[1]), squeeze=False)
        axes = list(axes.flatten())
    else:
        # Tworzymy siatkę 2 wiersze na 6 kolumn (najmniejsza wspólna wielokrotność dla 2 i 3)
        # Dzięki temu każdy subplot zawsze ma szerokość "2 jednostek" siatki
        nrows = 2
        ncols = 6
        fig = plt.figure(figsize=(figsize_per_plot[0]*3, figsize_per_plot[1]*nrows))
        gs = gridspec.GridSpec(nrows, ncols, figure=fig)
        
        axes = []
        # Pierwszy wiersz: 3 wykresy, każdy zajmuje 2 kolumny (0-1, 2-3, 4-5)
        for i in range(2):
            axes.append(fig.add_subplot(gs[0, i*2 : (i*2+2)]))
            
        # Drugi wiersz: 2 wykresy, każdy zajmuje 2 kolumny, ale przesunięte (1-2, 3-4)
        # To sprawi, że będą idealnie wyśrodkowane pod górnym rzędem
        axes.append(fig.add_subplot(gs[1, 1:3]))
        axes.append(fig.add_subplot(gs[1, 3:5]))

    # --- Reszta logiki (identyczna jak u Ciebie) ---
    def find_file_in_datasets(fname):
        for ds in datasets:
            for item in ds:
                if item["file"] == fname: return item
        raise ValueError(f"Plik '{fname}' nie znaleziony.")

    peak_cfg = {"P1": ("red", "P1"), "P2": ("green", "P2"), "P3": ("blue", "P3")}

    for i, (ax, fname, label) in enumerate(zip(axes, filenames, panel_labels)):
        item = find_file_in_datasets(fname)
        signal = item["signal"]
        peaks_ref = item.get("peaks_ref", {})
        x, y = signal.iloc[:, 0].values, signal.iloc[:, 1].values

        ax.plot(x, y, color="black", linewidth=1.2)

        for key, (color, txt) in peak_cfg.items():
            val = peaks_ref.get(key, None)
            if val is not None:
                idx = np.atleast_1d(val)
                for pt in map(int, idx):
                    ax.plot(x[pt], y[pt], "o", color=color, markersize=8)
                    ax.annotate(txt, (x[pt], y[pt]), xytext=(0, 8), textcoords="offset points",
                                ha="center", va="bottom", fontsize=14)

        ax.set_xlabel("Numer próbki", fontsize=14)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0, 180)
        ax.set_ylim(bottom=0)
        ax.text(0.02, 1, f"({label})", transform=ax.transAxes, fontweight="bold", va="top",
                fontsize=14)
        
        # Etykieta Y tylko dla pierwszych wykresów w wierszu
        if i == 0 or i == 3:
            ax.set_ylabel("Znormalizowana amplituda", fontsize=14)

    fig.tight_layout()
    return fig, axes

import matplotlib.gridspec as gridspec

def plot_selected_files(detection_results, class_name, files_to_plot):
    selected_files = [next(item for item in detection_results if item["file"] == f and item["class"] == class_name)
                      for f in files_to_plot]
    
    n_files = len(selected_files)
    nrows = (n_files + 1) // 2 
    
    # Zawsze 2 kolumny w sensie wizualnym, ale używamy 4 jednostek GridSpec dla wyśrodkowania
    fig = plt.figure(figsize=(10, 3.2 * nrows))
    
    if n_files == 3:
        # Siatka 2 rzędy x 4 kolumny
        gs = gridspec.GridSpec(2, 4)
        # Wykres 1: rząd 0, kolumny 0-2 | Wykres 2: rząd 0, kolumny 2-4
        # Wykres 3: rząd 1, kolumny 1-3 (wyśrodkowany, szerokość 2 jednostki - taka sama jak wyżej!)
        ax_positions = [gs[0, 0:2], gs[0, 2:4], gs[1, 1:3]]
    else:
        # Dla 1, 2 lub 4 plików - standardowy układ 2-kolumnowy
        gs = gridspec.GridSpec(nrows, 2)
        ax_positions = [gs[i // 2, i % 2] for i in range(n_files)]

    # --- Reszta Twojego kodu bez zmian ---
    peak_colors_ref = {'P1': 'red', 'P2': 'green', 'P3': 'blue'}
    peak_colors_det = {'P1': 'gold', 'P2': 'chartreuse', 'P3': 'cornflowerblue'}
    letters = [f"({chr(97+i)})" for i in range(n_files)]
    
    for idx, item in enumerate(selected_files):
        ax = fig.add_subplot(ax_positions[idx])
        df = item["signal"]
        t = df.iloc[:, 0].values
        y = df.iloc[:, 1].values
        
        ax.plot(t, y, color='black', lw=1)
        
        # Piki referencyjne
        for p, ref_idx in item.get("peaks_ref", {}).items(): 
            if ref_idx is None or (isinstance(ref_idx, float) and np.isnan(ref_idx)): continue
            ref_idx = [ref_idx] if not isinstance(ref_idx, (list, tuple, np.ndarray)) else list(ref_idx)
            ax.scatter(t[ref_idx], y[ref_idx], color=peak_colors_ref.get(p, "gray"),
                        marker="o", s=70, alpha=1.0, label=f"{p} ref." if idx == 0 else "", zorder=10)
        
        # Piki wykryte
        for p, det_idx in item.get("peaks_detected", {}).items():
            if det_idx is None or (isinstance(det_idx, float) and np.isnan(det_idx)): continue
            det_idx = [det_idx] if not isinstance(det_idx, (list, tuple, np.ndarray)) else list(det_idx)
            ax.scatter(t[det_idx], y[det_idx], color=peak_colors_det.get(p, "gray"),
                        marker="x", s=90, alpha=0.98,linewidths=2, label=f"{p} wykryty" if idx == 0 else "", zorder=10)
        
        # Oznaczenia osi
        ax.set_ylabel('Znormalizowana amplituda' if idx % 2 == 0 or (n_files==3 and idx==2) else '', fontsize=13)
        ax.set_xlabel('Numer próbki', fontsize=13)
        
        # Litery (a, b, c)
        ax.text(0.02, 0.95, letters[idx], transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')
        
        ax.set_xlim(0, 180)
        ax.set_ylim(bottom=0)

    # Legenda wewnątrz pierwszego wykresu
    ax_first = fig.get_axes()[0]
    ax_first.legend(loc='upper right', fontsize=11, frameon=True)

    plt.tight_layout(h_pad=2.1)
    # plt.show()

def plot_selected_files_mid(detection_results, class_name, files_to_plot):
    """
    Rysuje wybrane pliki z danej klasy.
    
    Parametry:
        detection_results: lista słowników z wynikami detekcji
        class_name: nazwa klasy, którą chcemy filtrować
        files_to_plot: lista nazw plików do wyplotowania
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # --- Filtrujemy wybrane pliki ---
    selected_files = [next(item for item in detection_results if item["file"] == f and item["class"] == class_name)
                      for f in files_to_plot]
    n_files = len(selected_files)

    # --- Układ subplotów ---
    if n_files <= 3:
        nrows = 2 if n_files > 1 else 1
        ncols = 2
    else:
        ncols = 3
        nrows = (n_files + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows), squeeze=False)
    axes = axes.flatten()

    # --- Kolory pików ---
    peak_colors_ref = {'P1': 'red', 'P2': 'green', 'P3': 'blue'}
    peak_colors_det = {'P1': 'orange', 'P2': 'chartreuse', 'P3': 'cornflowerblue'}

    letters = [f"({chr(97+i)})" for i in range(n_files)]  # (a), (b), ...

    FS_TITLE = 16
    FS_AXIS = 15
    FS_TICK = 13
    FS_LEG = 12

    for idx, item in enumerate(selected_files):
        ax = axes[idx]
        df = item["signal"]
        t = df.iloc[:, 0].values
        y = df.iloc[:, 1].values

        # --- Sygnał ---
        ax.plot(t, y, color='black', lw=1.5)

        # --- PIKI REFERENCYJNE ---
        for p, ref_idx in item.get("peaks_ref", {}).items():
            if ref_idx is None or (isinstance(ref_idx, float) and np.isnan(ref_idx)):
                continue
            ref_idx = [ref_idx] if not isinstance(ref_idx, (list, tuple, np.ndarray)) else list(ref_idx)
            ax.scatter(t[ref_idx], y[ref_idx], color=peak_colors_ref.get(p, "gray"),
                       marker="o", s=90, alpha=1.0,
                       label=f"{p} referencyjny" if idx == 0 else "", zorder=10)

        # --- PIKI WYKRYTE ---
        for p, det_idx in item.get("peaks_detected", {}).items():
            if det_idx is None or (isinstance(det_idx, float) and np.isnan(det_idx)):
                continue
            det_idx = [det_idx] if not isinstance(det_idx, (list, tuple, np.ndarray)) else list(det_idx)
            ax.scatter(t[det_idx], y[det_idx], color=peak_colors_det.get(p, "gray"),
                       marker="x", s=140, alpha=0.95,
                       label=f"{p} wykryty" if idx == 0 else "", zorder=10)

        # --- Podpisy osi ---
        if idx % ncols == 0:
            ax.set_ylabel('Znormalizowana amplituda', fontsize=FS_AXIS)
        else:
            ax.set_ylabel('')
        ax.set_xlabel('Numer próbki', fontsize=FS_AXIS)

        ax.tick_params(axis='both', labelsize=FS_TICK)

        # --- Litera (a), (b), ... ---
        ax.text(0.02, 0.95, letters[idx], transform=ax.transAxes,
                fontsize=FS_TITLE, fontweight='bold', va='top', ha='left')

        ax.set_title('')
        ax.set_xlim(0, 180)
        ax.set_ylim(bottom=0)

    # --- Usuń puste subplots ---
    for j in range(idx+1, len(axes)):
        fig.delaxes(axes[j])

    # --- Wyśrodkowanie dla niepełnego dolnego rzędu ---
    if n_files % ncols != 0:
        last_row_idx = n_rows = (n_files + ncols - 1) // ncols - 1
        extra = ncols - (n_files % ncols)
        for i in range(extra):
            axes[n_files + i].set_visible(False)

    # --- Legenda tylko raz ---
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=FS_LEG, frameon=True)

    fig.tight_layout()
    plt.show()


def plot_selected_files_old(detection_results, class_name, files_to_plot, ncols=3):
    """
    Rysuje wybrane pliki z danej klasy.
    
    Parametry:
        detection_results: lista słowników z wynikami detekcji
        class_name: nazwa klasy, którą chcemy filtrować
        files_to_plot: lista nazw plików do wyplotowania
        ncols: liczba kolumn w subplotach (domyślnie 3)
    """
    # Filtrujemy tylko pliki z danej klasy i podanej listy
    # selected_files = [
    #     item for item in detection_results
    #     if item["class"] == class_name and item["file"] in files_to_plot
    # ]
    
    selected_files = [next(item for item in detection_results if item["file"] == f and item["class"] == class_name)
                  for f in files_to_plot]

    
    n_files = len(selected_files)
    nrows = (n_files + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.3*nrows), squeeze=False)
    axes = axes.flatten()
    
    # Kolory dla pików
    peak_colors_ref = {'P1': 'red', 'P2': 'green', 'P3': 'blue'}
    peak_colors_det = {'P1': 'gold', 'P2': 'chartreuse', 'P3': 'cornflowerblue'}
    
    letters = [f"({chr(97+i)})" for i in range(n_files)]  # (a), (b), ...
    
    for idx, item in enumerate(selected_files):
        ax = axes[idx]
        df = item["signal"]
        t = df.iloc[:, 0].values
        y = df.iloc[:, 1].values
        
        # Rysujemy sygnał
        ax.plot(t, y, color='black', lw=1.2)
        
        # ===== PIKI REFERENCYJNE =====
        for p, ref_idx in item.get("peaks_ref", {}).items(): 
            if ref_idx is None or (isinstance(ref_idx, float) and np.isnan(ref_idx)):
                continue
            ref_idx = [ref_idx] if not isinstance(ref_idx, (list, tuple, np.ndarray)) else list(ref_idx)
            ax.scatter(t[ref_idx], y[ref_idx], color=peak_colors_ref.get(p, "gray"),
                       marker="o", s=70, alpha=1.0, label=f"{p} referencyjny" if idx == 0 else "", zorder=10)
        
        # ===== PIKI WYKRYTE =====
        for p, det_idx in item.get("peaks_detected", {}).items():
            if det_idx is None or (isinstance(det_idx, float) and np.isnan(det_idx)):
                continue
            det_idx = [det_idx] if not isinstance(det_idx, (list, tuple, np.ndarray)) else list(det_idx)
            ax.scatter(t[det_idx], y[det_idx], color=peak_colors_det.get(p, "gray"),
                       marker="x", s=90, alpha=1, linewidths=6, label=f"{p} wykryty" if idx == 0 else "", zorder=10)
        
        if idx % ncols == 0:
            ax.set_ylabel('Znormalizowana amplituda', fontsize=18)
        else:
            ax.set_ylabel('')
            
        ax.set_xlabel('Numer próbki', fontsize=18)
        
        ax.text(
            0.02, 0.95, letters[idx],
            transform=ax.transAxes,
            fontsize=14,
            fontweight='bold',
            va='top',
            ha='left'
        )
        
        
        ax.set_title('')  # brak tytułów
        ax.set_xlim(0, 180)
        ax.set_ylim(bottom=0)
    
    # Usuń puste subplots
    for j in range(idx+1, len(axes)):
        fig.delaxes(axes[j])
    
    # Dodaj legendę tylko raz
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=14)
    fig.tight_layout()
    # plt.show()


if __name__ == "__main__":
    




    print("bajo jajo")