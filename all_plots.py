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
from matplotlib.patches import Rectangle
from pypalettes import load_cmap
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

    import matplotlib.pyplot as plt
    import numpy as np

    n_classes = len(classes)
    n_peaks = len(peak_types)

    fig, axes = plt.subplots(n_classes, n_peaks, figsize=(5*n_peaks, 4*n_classes), squeeze=False)

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
                axes[i, j].set_title(f"{class_name} – brak danych")
            continue

        t = class_items[0]["signal"].iloc[:, 0].values
        all_x = [item["signal"].iloc[:, 1].values for item in class_items]
        mean_signal = np.mean(all_x, axis=0)

        for j, peak_type in enumerate(peak_types):
            ax = axes[i, j]

            # --- rysujemy wszystkie sygnały ---
            for item in class_items:
                sig = item["signal"].iloc[:, 1].values
                ax.plot(t, sig, color="black", alpha=0.07, linewidth=0.7)

            # --- średni sygnał ---
            ax.plot(t, mean_signal, color="blue", linewidth=2, label="Średni sygnał")

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
                ax_hist.hist(manual_t, bins=40, density=True, alpha=0.35, color="green", label="Piki referencyjne")
            if auto_t:
                ax_hist.hist(auto_t, bins=40, density=True, alpha=0.35, color="red", label="Piki wykryte")

            ax.set_title(f"{class_name} {peak_type}", fontsize=12)
            ax.set_xlabel("Numer próbki")
            ax.set_ylabel("Amplituda")
            ax.grid(alpha=0.3)

            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax_hist.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="upper right")

    fig.suptitle(
        f"{titles_dict.get(method_name, method_name)} — zakres: {ranges_name if ranges_name is not None else 'none'}",
        fontsize=18
    )
    plt.tight_layout(rect=[0,0,1,0.97])
    plt.show()


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
    cmap = load_cmap("Alexandrite")
    median_color = cmap.colors[0]  # ciemny turkus dla mediany
    outlier_color = cmap.colors[6]  # jasny fiolet dla outlierów

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
                medianprops=dict(color=median_color, linewidth=2),
                flierprops=dict(marker='o', markerfacecolor="none", markersize=5,
                                linestyle='none', markeredgecolor=outlier_color)
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
    
if __name__ == "__main__":
    cmap = load_cmap("Alexandrite")
    colors = cmap.colors
    n = len(colors)
    
    fig, ax = plt.subplots(figsize=(n * 0.9, 2))
    
    for i, col in enumerate(colors):
        ax.add_patch(
            Rectangle(
                (i, 0), 1, 1,
                facecolor=col,
                edgecolor="black"
            )
        )
        ax.text(
            i + 0.5, -0.15, str(i),
            ha="center", va="top", fontsize=10
        )
    
    ax.set_xlim(0, n)
    ax.set_ylim(-0.4, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title("Paleta Alexandrite – indeksy kolorów")
    ax.set_aspect("equal")
    
    plt.show()