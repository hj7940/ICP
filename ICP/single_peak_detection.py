# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 19:31:25 2025

@author: User
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.signal import find_peaks
# from main import load_data # moze uda sie lepiej to rozwiazac
# from wersja2 import save_with_increment

# %% ------- DANE -------------
def load_data(base_path):
    dataset = []

    for i in range(1, 5):
        class_name = f"Class{i}"

        # peaks CSV
        peaks_path = os.path.join(base_path, f"{class_name}_peaks.csv")
        peaks_df = pd.read_csv(peaks_path)

        # sygnały
        folder = os.path.join(base_path, class_name)
        csv_files = sorted(glob.glob(os.path.join(folder, f"{class_name}_example_*.csv")))

        for f in csv_files:
            signal_df = pd.read_csv(f)
            file_name = os.path.splitext(os.path.basename(f))[0]

            # znajdź wiersz z pikami
            row = peaks_df[peaks_df["File"] == file_name]
            if len(row) == 1:
                row = row.iloc[0]
                peaks_ref = {
                    "P1": int(row["P1"]) if row["P1"] >= 0 else None,
                    "P2": int(row["P2"]) if row["P2"] >= 0 else None,
                    "P3": int(row["P3"]) if row["P3"] >= 0 else None,
                }
            else:
                peaks_ref = {"P1": None, "P2": None, "P3": None}

            dataset.append({
                "class": class_name,
                "file": file_name,
                "signal": signal_df,
                "peaks_ref": peaks_ref
            })

    return dataset


# %% ------- DETEKCJA ----------
def single_peak_detection(peak_name, class_name, dataset, method_name, peak_ranges):
    """
    Wykrywa piki w określonym przedziale czasowym.

    Parameters
    ----------
    peak_name : str
        'P1', 'P2' lub 'P3'
    class_name : str
        'Class1' ... 'Class4'
    dataset : list
        wynik load_data()
    method_name : str
        nazwa metody: 'concave', 'modified_scholkmann', 'curvature', 'line_distance'
    peak_ranges : dict
        słownik: peak_ranges[Class][Pik] = (start, end) w jednostkach czasu

    Returns
    -------
    list[np.ndarray]
        lista tablic: dla każdego sygnału array indeksów wykrytych pików w podanym zakresie
    """
    
    if method_name not in METHODS:
        raise ValueError(f"Nieznana metoda: {method_name}")
        
    detect = METHODS[method_name]
    t_start, t_end = peak_ranges[class_name][peak_name]
    
    detected_all = []

    for item in dataset:
        if item["class"] != class_name:
            continue

        signal_df = item["signal"]
        t = signal_df.iloc[:, 0].values   # czas
        y = signal_df.iloc[:, 1].values   # amplituda

        # wykryj wszystkie piki metodą
        peaks = detect(y)
        peaks = np.array(peaks, dtype=int)

        # filtruj tylko te w zadanym zakresie czasu
        peaks_in_range = peaks[(t[peaks] >= t_start) & (t[peaks] <= t_end)]

        detected_all.append(peaks_in_range)

    return detected_all


# %%----------- METODY -------------
def concave(signal, d2x_threshold=0, prominence=0, threshold=None):
    """
    Wykrywa lokalne maksima sygnału (peaki) w obszarach, gdzie druga pochodna < d2x_threshold.

    t : np.ndarray
        Wektor czasu.
    x : np.ndarray
        Wektor wartości sygnału.
    d2x_threshold : float
        Próg dla drugiej pochodnej (obszary wklęsłe spełniają d2x < threshold).
    prominence : float
        Minimalna wyrazistość piku (parametr dla `find_peaks`).
    
    Returns
    -------
    list[int]
        Indeksy wykrytych pików.
    """
    # pochodne
    dx = np.gradient(signal, edge_order=2)
    d2x = np.gradient(dx, edge_order=2)

    # maska obszarów wklęsłych
    concave_mask = d2x < d2x_threshold

    # znajdź lokalne maksima w całym sygnale
    peaks, _ = find_peaks(signal, threshold=threshold, prominence=prominence)

    # wybierz tylko te, które leżą w wklęsłych fragmentach
    concave_peaks = [p for p in peaks if concave_mask[p]]

    return np.array(concave_peaks)

def modified_scholkmann(signal, scale=1):
    """
    Modified-Scholkmann peak detection.
    Zwraca indeksy stabilnych maksimów w wielu skalach.
    """
    
    x = signal
    N = len(x)
    
    # if max_scale is None:
    max_scale = N // scale  # ok. 1/4 długości sygnału, jak w tekście

    # Tworzymy macierz lokalnych maksimów
    LMS = np.zeros((max_scale, N), dtype=int)

    # Iterujemy po skalach
    for s in range(1, max_scale + 1):
        for t in range(s, N - s):
            if x[t] > x[t - s] and x[t] > x[t + s]:
                LMS[s - 1, t] = 1

    # Zliczamy, ile razy dany punkt był lokalnym maksimum
    maxima_strength = LMS.sum(axis=0)

    # Piki = punkty, które były maksimum w wielu skalach
    threshold = np.percentile(maxima_strength, 95)  # tylko te najstabilniejsze
    peaks = np.where(maxima_strength >= threshold)[0]

    return np.array(peaks)

def curvature(signal, d2x_threshold=0, prominence=0, threshold=None):
    """
    Detekcja pików na podstawie lokalnych maksimów krzywizny.
    """
    dx = np.gradient(signal, edge_order=2)
    d2x = np.gradient(dx, edge_order=2)

    # oblicz krzywiznę z zabezpieczeniem przed dzieleniem przez zero
    denom = (1 + dx**2)**1.5 # mianownik
    denom[denom == 0] = 1e-8
    curvature = np.abs(d2x) / denom

    # obszary wklęsłe
    concave_mask = d2x < d2x_threshold

    # wykryj piki w krzywiźnie
    # threshold to jak bardzo "wybija się" pik nad pozostałe punkty
    peaks, _ = find_peaks(curvature, threshold=threshold, prominence=prominence)

    # wybierz tylko piki w obszarach wklęsłych
    concave_peaks = [p for p in peaks if concave_mask[p]]

    return np.array(concave_peaks)

def line_distance(signal, d2x_threshold=0, mode="perpendicular", min_len=3):
    """
    Detekcja pików na podstawie odległości od linii łączącej końce regionu wklęsłego.
    mode = "perpendicular"  → metoda 3 (prostopadła odległość)
    mode = "vertical"       → metoda 4 (pionowa odległość)
    """

    dx = np.gradient(signal, edge_order=2)
    d2x = np.gradient(dx, edge_order=2)

    concave_mask = d2x < d2x_threshold
    peaks = []

    # Znajdź granice regionów wklęsłych
    mask_diff = np.diff(concave_mask.astype(int))
    region_starts = np.where(mask_diff == 1)[0]
    region_ends = np.where(mask_diff == -1)[0]

    # Korekta: jeśli maska zaczyna się/kończy w środku regionu
    if concave_mask[0]:
        region_starts = np.insert(region_starts, 0, 0)
    if concave_mask[-1]:
        region_ends = np.append(region_ends, len(signal) - 1)
        
    t = np.arange(len(signal))
    x = signal
    # Iteracja po regionach wklęsłych
    for start, end in zip(region_starts, region_ends):
        if end - start < min_len:
            continue  # pomiń zbyt krótkie regiony

        x_seg = x[start:end + 1]
        t_seg = t[start:end + 1]

        # Linia bazowa między końcami
        slope = (x_seg[-1] - x_seg[0]) / (t_seg[-1] - t_seg[0])
        intercept = x_seg[0] - slope * t_seg[0]
        x_line = slope * t_seg + intercept

        if mode == "vertical":
           # różnica pionowa
           distance = x_seg - x_line
        elif mode == "perpendicular":
           # różnica prostopadła do linii
           distance = (x_seg - x_line) / np.sqrt(1 + slope**2)
        else:
           raise ValueError("mode must be 'vertical' or 'perpendicular'")

       # Maksymalna odległość to pozycja piku
        peak_rel_idx = np.argmax(distance)
        peak_idx = start + peak_rel_idx
        peaks.append(peak_idx)

    return np.array(peaks)  

    
# %%-------- WYKRESY I WIZUALIZACJE ---------------
def plot_all_signals_with_peaks(data, results_df, method_name):
    """
    Rysuje 4 subplotsy: Class1..Class4
    Każdy subplot:
        - wszystkie sygnały (alpha=0.1)
        - średni sygnał
        - gęstość pików ręcznych (green)
        - gęstość pików automatycznych (red)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i in range(1, 5):
        ax = axes[i-1]
        class_name = f"Class{i}"

        signals = data[f"{class_name}_signals"]
        peaks_df = data[f"{class_name}_peaks"]

        # --- Rysowanie wszystkich sygnałów ---
        common_t = signals[0].iloc[:, 0].values
        all_x = []

        for sig in signals:
            t = sig.iloc[:, 0].values
            x = sig.iloc[:, 1].values

            ax.plot(t, x, color="black", alpha=0.07, linewidth=0.7)
            all_x.append(x)

        # --- średnia krzywa fali ICP ---
        mean_signal = np.mean(all_x, axis=0)
        ax.plot(common_t, mean_signal, color="blue", linewidth=2.0, label="Średni sygnał")
        ax.margins(y=0, x=0)

        # --- gęstość punktów P1,P2,P3 ---
        manual_points_t = []
        auto_points_t   = []

        # automatyczne piki z results_df
        df_method = results_df[
            (results_df["Class"] == class_name) &
            (results_df["Method"] == method_name)
        ]

        for idx, row in peaks_df.iterrows():
            file_name = row["File"]
            sig_idx = next((j for j,f in enumerate(data[f"{class_name}_files"]) if file_name in f), None)
            sig = signals[sig_idx]
            t = sig.iloc[:,0].values

            # ręczne
            for p in ["P1", "P2", "P3"]:
                val = row[p]
                if pd.notna(val) and val != -1:
                    manual_points_t.append(t[int(val)])

            # automatyczne
            row_auto = df_method[df_method["File"] == file_name]
            if len(row_auto) > 0:           
                peaks = row_auto.iloc[0]["Detected_Peaks"]

                # upewnij się, że jest to np. tablica NumPy integerów
                if isinstance(peaks, list):
                    peaks = np.array(peaks, dtype=int)
                elif isinstance(peaks, np.ndarray) and peaks.dtype != int:
                    peaks = peaks.astype(int)
                
                auto_points_t.extend(t[peaks])
  
        ax_hist = ax.twinx()
        
        # narysuj gęstość punktów
        if len(manual_points_t) > 0:
            ax_hist.hist(manual_points_t, bins=50, color="green",
                    alpha=0.3, label="Ręczne piki")

        if len(auto_points_t) > 0:
            ax_hist.hist(auto_points_t, bins=50, color="red",
                    alpha=0.3, label="Automatyczne piki")

        ax.set_title(class_name)
        ax.set_xlabel("Czas [s]")
        ax.set_ylabel("Amplituda")
        ax.grid(True, alpha=0.3)
        
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax_hist.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
        #ax.legend()
   
    fig.suptitle(f'{method_name}')
    plt.tight_layout()
    plt.show()
  
def compute_peak_metrics(dataset, detected_peaks, peak_name, class_name):
    """
    Parameters
    ----------
    dataset : list
        wynik load_data()
    detected_peaks : list of np.ndarray
        wynik single_peak_detection dla danego piku i klasy
    peak_name : str
        'P1', 'P2' lub 'P3'
    class_name : str

    Returns
    -------
    pd.DataFrame
        Metrics dla wszystkich sygnałów danej klasy i piku
    """
    metrics_list = []
    
    class_signals = [item for item in dataset if item["class"] == class_name]

    for item, peaks in zip(class_signals, detected_peaks):
        if item["class"] != class_name:
            continue
        
        file_name = item["file"]
        signal_df = item["signal"]
        t = signal_df.iloc[:, 0].values
        y = signal_df.iloc[:, 1].values

        ref_idx = item["peaks_ref"].get(peak_name)
        if ref_idx is None:
            metrics_list.append({
                "Class": class_name,
                "Peak": peak_name,
                "File": file_name,
                "Mean_X_Error": np.nan,
                "Mean_Y_Error": np.nan,
                "Mean_XY_Error": np.nan,
                "Peak_Count": 0,
                "Detected_Peaks": [],
                "Reference_Peaks": ref_idx 
            })
            continue

        t_ref = t[ref_idx]
        y_ref = y[ref_idx]

        if len(peaks) == 0:
            metrics_list.append({
                "Class": class_name,
                "Peak": peak_name,
                "File": file_name,
                "Mean_X_Error": np.nan,
                "Mean_Y_Error": np.nan,
                "Mean_XY_Error": np.nan,
                "Peak_Count": 0,
                "Detected_Peaks": [],
                "Reference_Peaks": ref_idx 
            })
            continue

        t_detected = t[peaks]
        y_detected = y[peaks]

        dx = abs(t_detected - t_ref)
        dy = abs(y_detected - y_ref)
        dxy = np.sqrt(dx**2 + dy**2)
        
        # if len(peaks) == 0:
        #     metrics_list.append({
        #         "Class": class_name,
        #         "Peak": peak_name,
        #         "Mean_X_Error": np.nan,
        #         "Mean_Y_Error": np.nan,
        #         "Mean_XY_Error": np.nan,
        #         "Peak_Count": 0
        #     })
            
        metrics_list.append({
            "Class": class_name,
            "Peak": peak_name,
            "File": file_name,
            "Mean_X_Error": np.mean(dx),
            "Mean_Y_Error": np.mean(dy),
            "Mean_XY_Error": np.mean(dxy),
            "Peak_Count": len(peaks),
            "Reference_Peaks": ref_idx, 
            "Detected_Peaks": list(peaks)
        })
        
    df_metrics = pd.DataFrame(metrics_list)
    
    if df_metrics.empty:
    # Zwróć pusty DataFrame z odpowiednimi kolumnami, żeby groupby nie padło
        df_metrics = pd.DataFrame(columns=["Class", "Peak", "Mean_X_Error", "Mean_Y_Error", "Mean_XY_Error", "Peak_Count"])
        
    

    # uśrednij po wszystkich sygnałach danej klasy i piku
    # df_avg = df_metrics.groupby(["Class", "Peak"]).mean().reset_index()

    return df_metrics

def plot_class_peaks_grid(dataset, results_df, method_name):

    # lista subplotów (Class, Peak)
    tasks = [
        ("Class1", "P1"), ("Class1", "P2"), ("Class1", "P3"),
        ("Class2", "P1"), ("Class2", "P2"), ("Class2", "P3"),
        ("Class3", "P1"), ("Class3", "P2"), ("Class3", "P3"),
        ("Class4", "P2")   # ostatni, ma być wyśrodkowany
    ]

    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    axes = axes.flatten()

    # indeksy do rysowania
    plot_positions = list(range(9)) + [9 + 1]  # ostani na pozycji 10 → drugi w 4 rzędzie

    # zamieniamy wszystkie axes na niewidoczne — potem włączymy tylko te używane
    for ax in axes:
        ax.set_visible(False)

    for (cls, pk), ax_i in zip(tasks, plot_positions):

        ax = axes[ax_i]
        ax.set_visible(True)

        # sygnały danej klasy
        class_items = [item for item in dataset if item["class"] == cls]
        signals = [item["signal"] for item in class_items]

        if len(signals) == 0:
            ax.set_title(f"{cls} – {pk} (brak danych)")
            continue

        t_common = signals[0].iloc[:, 0].values
        all_y = []

        # wszystkie sygnały
        for sig in signals:
            t = sig.iloc[:, 0].values
            y = sig.iloc[:, 1].values
            all_y.append(y)
            ax.plot(t, y, color='black', alpha=0.07)

        # średni sygnał
        mean_sig = np.mean(all_y, axis=0)
        ax.plot(t_common, mean_sig, color='blue', linewidth=2, label="Średni sygnał")

        # filtr metody
        df_method = results_df[
            (results_df["Class"] == cls) &
            (results_df["Peak"] == pk) &
            (results_df["Method"] == method_name)
        ]

        manual_t = []
        auto_t = []

        for item in class_items:
            fname = item["file"]
            sig = item["signal"]
            t = sig.iloc[:, 0].values

            # referencyjne
            ref = item["peaks_ref"].get(pk)
            if ref is not None:
                if isinstance(ref, (list, np.ndarray)):
                    manual_t.extend(t[np.array(ref, dtype=int)])
                else:
                    manual_t.append(t[int(ref)])

            # automatyczne
            row = df_method[df_method["File"] == fname]
            if len(row) > 0:
                det = row.iloc[0]["Detected_Peaks"]
                if isinstance(det, (list, np.ndarray)):
                    auto_t.extend(t[np.array(det, dtype=int)])
                elif isinstance(det, (np.integer, int)):
                    auto_t.append(t[int(det)])

        # podhistogram
        ax2 = ax.twinx()
        if len(manual_t) > 0:
            ax2.hist(manual_t, bins=40, color='green', alpha=0.3, label="Ręczne piki")
        if len(auto_t) > 0:
            ax2.hist(auto_t, bins=40, color='red', alpha=0.3, label="Automatyczne piki")

        ax.set_title(f"{cls} – {pk}")
        ax.set_xlabel("Czas [s]")
        ax.set_ylabel("Amplituda")
        ax.grid(True, alpha=0.3)


    # GLOBALNA LEGENDA
    handles = [
        plt.Line2D([0], [0], color='blue', lw=2, label='Średni sygnał'),
        plt.Rectangle((0, 0), 1, 1, color='green', alpha=0.3, label='Ręczne piki'),
        plt.Rectangle((0, 0), 1, 1, color='red', alpha=0.3, label='Automatyczne piki')
    ]
    labels = [h.get_label() for h in handles]
    fig.legend(handles, labels, loc='upper right', fontsize=12)

    fig.suptitle(f"Histogramy i sygnały – {method_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # zostawiamy miejsce na legendę
    plt.show()
    
# %% ------------- EXECUTION --------------
base_path = r"C:\Users\User\OneDrive\Dokumenty\praca inżynierska\ICP_pulses_it1"
data = load_data(base_path)

pd.options.display.float_format = '{:.4f}'.format


METHODS = {
    "concave": concave,
    "modified_scholkmann": modified_scholkmann,
    "curvature": curvature,
    "line_distance": line_distance
}

PEAK_RANGES = {
    "Class1": {"P1": (17, 73), "P2": (34, 107), "P3": (60, 143)},
    "Class2": {"P1": (17, 59), "P2": (36, 98), "P3": (61, 138)},
    "Class3": {"P1": (13, 42), "P2": (32, 83), "P3": (49, 116)},
    "Class4": {"P2": (32, 73)},
}

all_metrics = []

# test_df = single_peak_detection("P1", "Class3", data, "modified_scholkmann", PEAK_RANGES)
for method in METHODS.keys():
    # klasy 1-3
    for cls in ["Class1", "Class2", "Class3"]:
        for pk in ["P1", "P2", "P3"]:
            detected = single_peak_detection(pk, cls, data, method, PEAK_RANGES)
            df = compute_peak_metrics(data, detected, pk, cls)
            df["Method"] = method
            if not df.empty:
                all_metrics.append(df)

    # klasa 4 - tylko P2
    cls = "Class4"
    pk = "P2"
    detected = single_peak_detection(pk, cls, data, method, PEAK_RANGES)
    df = compute_peak_metrics(data, detected, pk, cls)
    df["Method"] = method
    if not df.empty:
        all_metrics.append(df)

# # jeden duży DataFrame
metrics_df = pd.concat(all_metrics, ignore_index=True)

# avg_stats = (
#     metrics_df
#     .groupby(["Class", "Peak", "Method"])
#     .agg({
#         "Mean_X_Error": "mean",
#         "Mean_Y_Error": "mean",
#         "Mean_XY_Error": "mean",
#         "Peak_Count": "mean",
#     })
#     .reset_index()
# )
# print(avg_stats)

for m in METHODS.keys():
    plot_class_peaks_grid(data, metrics_df, m)

