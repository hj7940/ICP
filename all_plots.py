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
        "modified_scholkmann0-5": "Zmodyfikowana metoda Scholkmanna (limit=0.5)",
        "modified_scholkmann1": "Zmodyfikowana metoda Scholkmanna (limit=1.0)",
        "curvature": "Maksymalna krzywizna",
        "line_distance_8": "Odległość od linii bazowej",
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
    fontsize_labels=14,
    fontsize_ticks=10
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

            ax.set_title(f"Klasa {class_id[-1]}", fontsize=fontsize_labels)
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
            fontsize=15,
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
            ha='center', va='bottom', fontsize=11)

def plot_peak_features_boxplots_it1_it2_with_significance(
    df_it1, df_it2, df_stats,
    features=("Index", "Height", "Prominence"),
    classes=("Class1", "Class2", "Class3", "Class4"),
    colors=("lightseagreen", "lightblue")
):
    label_map = {"Index": "Położenie piku", "Height": "Amplituda", "Prominence": "Wyrazistość"}
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
            ax.set_xticklabels(peaks, fontsize=14)
            ax.set_title(f"Klasa {class_id[-1]}", fontsize=14) # Niepogrubione
            ax.set_ylabel(label_map[feat], fontsize=12)
            ax.grid(axis='y', alpha=0.3)

        # Definicja legendy
        legend_elements = [
            Patch(facecolor=colors[0], edgecolor="black", label="Zestaw 1"),
            Patch(facecolor=colors[1], edgecolor="black", label="Zestaw 2"),
            Line2D([0], [0], color='none', label=''), 
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
                   fontsize=10, 
                   handletextpad=0.8)
        
        for text in leg.get_texts():
            if text.get_text() == 'Test U Manna-Whitneya:':
                # Przesunięcie o -20 punktów w lewo (wartość dobrana do handlelength)
                text.set_position((-32, 0))
                
        fig.suptitle(f"{label_map[feat]} — porównanie zestawów", 
                     fontsize=16, y=0.94)
        
        # rect=[0, 0, 0.82, 0.95] ścieśnia wykresy do 82% szerokości, robiąc miejsce na legendę
        plt.tight_layout(rect=[0, 0, 0.82, 0.95]) 
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
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 3*n_rows), sharey=False)
    
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
            
            ax.plot(t, y, color=colors_signal, linewidth=1.5)
            
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
                            markersize=6,
                            label=peak
                        )
            
            if i == 0:
                ax.set_title(dataset_labels[j], fontsize=16, pad=10)
            if j == 0:
                ax.set_ylabel("Amplituda", fontsize=12)
            
            ax.set_xlabel("Numer próbki", fontsize=12)
            ax.grid(alpha=0.3)
            
            # legenda tylko w pierwszej kolumnie i pierwszym wierszu
            if i == 0 and j == 0:
                ax.legend(fontsize=10, loc="upper right")
    
    plt.tight_layout()
    plt.savefig("rysunki/filtracja.pdf", format="pdf", bbox_inches=None)
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
    colors = ["#66b3ff", "#ff9999"]

    plt.figure(figsize=(6,6))
    plt.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90
    )
    plt.title(f"Klasa {class_id[-1]}", fontsize=12)
    # plt.show()

    
if __name__ == "__main__":
    print("bajo jajo")