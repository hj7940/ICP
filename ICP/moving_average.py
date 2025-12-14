# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 01:08:05 2025

@author: User
"""
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
# import glob
# import os
# from scipy.signal import find_peaks, hilbert, find_peaks_cwt
# import itertools
import math
from scipy.signal import butter, filtfilt

# %% ------- DANE -------------
def load_dataset(base_path, it_type):
    """
    Ładuje sygnały i piki dla IT1 lub IT2.
    
    Parameters
    ----------
    base_path : str
        sciezka do glownego folderu z plikami danego zestawu
    it_type : str 
        "it1" lub "it2"
        
    Returns
    -------
    dataset : list
        Lista zawierajaca elementy:
            
        - class : str (np. "Class1")
        
        - file : str (np. "Class1_example_0001")
        
        - signal: DataFrame (kolumny: Sample_no, ICP)
        
        - peaks_ref : dict (np. {'P1': 31, 'P2': 53, 'P3': 90})
    """
    dataset = []
    file_pattern = "{class_name}_example_*.csv" if it_type == "it1" else "{class_name}_it2_example_*.csv"
    peaks_suffix = "_peaks.csv" if it_type == "it1" else "_it2_peaks.csv"
    
    for i in range(1, 5):
        class_name = f"Class{i}"
        peaks_path = os.path.join(base_path, f"{class_name}{peaks_suffix}") # sciezka do pliku z pikami
        peaks_df = pd.read_csv(peaks_path) # wczytuje piki dla danej klasy
        
        folder = os.path.join(base_path, class_name) # sygnaly dla danej klasy
        csv_files = sorted(glob.glob(os.path.join(folder, file_pattern.format(class_name=class_name))))
        
        for f in csv_files: # f to poszczegolne sygnaly
            signal_df = pd.read_csv(f)
            file_name = os.path.splitext(os.path.basename(f))[0] # zwraca tylko nazwe pliku bez rozszerzenia
            
            row = peaks_df[peaks_df["File"] == file_name]  # df zawierajacy wiersze dla ktorych File==file_name
            if len(row) == 1: # spawdzenie czy DOKLADNIE JEDEN wiersz
                row = row.iloc[0] # z data frame 2d do series 1d
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

# %% ------- PRZYGOTOWANIE --------
def smooth_butter(x, cutoff=1, fs=100, order=3):
    b, a = butter(order, cutoff / (fs / 2), btype='low')
    return filtfilt(b, a, x)

def smooth_dataset(dataset, inplace=False, classes=None, cutoff=4, **kwargs):
    """
    Wygładza wszystkie sygnały w dataset.

    Parameters
    ----------
    dataset : list
        wynik load_data()
    inplace : bool
        jeśli True — modyfikuje przekazany dataset, else zwraca kopię
    classes : list or None
        lista klas do wygładzenia, np. ['Class1','Class2']; None -> wszystkie
    kwargs : dodatkowe parametry w zależności od metody:
        - savgol: window (int, odd), poly (int)
        - ma: window (int)  (centered moving average)
        - butter: cutoff (Hz), fs (Hz), order (int)
        - median: kernel (int, odd)
    Returns
    -------
    dataset_out : list
        wygładzony dataset (jeśli inplace=False). Jeśli inplace=True zwraca None.
    """
    if inplace:
        ds = dataset
    else:
        # Tworzymy nową listę słowników z kopiami DataFrame
        ds = []
        for item in dataset:
            ds.append({
                "class": item["class"],
                "file": item["file"],
                "signal": item["signal"].copy(),  # kopia DataFrame
                "peaks_ref": item["peaks_ref"].copy() if item.get("peaks_ref") else None
            })

    

    target_classes = set(classes) if classes is not None else None

    for item in ds:
        cls = item["class"]
        if target_classes is not None and cls not in target_classes:
            continue

        sig_df = item["signal"]
        x = sig_df.iloc[:, 1].values


        #cutoff = kwargs.get("cutoff", 4.0)
        fs = kwargs.get("fs", 100.0)
        order = kwargs.get("order", 3)
        # filtfilt zachowuje fazę (zero-phase)
        x_s = smooth_butter(x, cutoff=cutoff, fs=fs, order=order)


        # podstawiamy wygładzony sygnał z powrotem do DataFrame (zachowujemy kolumnę czasu)
        # sig_df_s = sig_df.copy()
        sig_df.iloc[:, 1] = x_s
        # item["signal"] = sig_df_s

    if inplace:
        return None
    else:
        return ds


# %% -------- DETEKCJA --------

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
    min_distance : int
        minimalna odleglosc od poprzedniego punktu przeciecia
    max_crossings :
        liczba po ktorej zatrzymuje sie wykrywanie p. przeciecia (domyslnie 6)
        
    Returns
    ------
    cross : dict
        Klucz - nazwa klasy (np. "Class1"), wartosc - slownik zawierajacy nazwe pliku (
            np. Class1_it2_example_0001) oraz liste p. przeciecia (int)
    
    """
    crossings_by_class = {}

    classes = sorted({item["class"] for item in dataset})
    for cls in classes:
        crossings_by_class[cls] = []
    
    amp_thresholds = {
        "Class1": 0.40,
        "Class2": 0.30,
        "Class3": 0.10,
        "Class4": 0.40,
    }

    for cls in classes:
        
        items = [item for item in dataset if item["class"] == cls]
        cls_threshold = amp_thresholds.get(cls, 0.0)
        
        n_crossings_list = []
        
        for item in items:
            sig = item["signal"].iloc[:,1].values
            xf = pd.Series(sig).rolling(window_fast, min_periods=1, center=True).mean().to_numpy()
            xs = pd.Series(sig).rolling(window_slow, min_periods=1, center=True).mean().to_numpy()
            
            crossings = []
            last_cross = -min_distance
            
            found_first_valid = False
            kept_count = 0
            
            for i in range(1, len(sig)):
                crossed = ((xf[i] >= xs[i] and xf[i-1] < xs[i-1]) or
                           (xf[i] <= xs[i] and xf[i-1] > xs[i-1]))
                
                if crossed and (i - last_cross >= min_distance):
                    amp = sig[i]
                    
                    if not found_first_valid:
                        if amp >= cls_threshold:
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

            crossings_by_class[cls].append({
                "File": item["file"],
                "Crossings": crossings
            })
            n_crossings_list.append(len(crossings))
        
    return crossings_by_class

def compute_crossings_summary(dataset, crossings_by_class, window_fast=2, window_slow=4,
                              min_distance=0):
    """
    Generuje podsumowanie liczby punktów przecięcia dla każdej klasy.

    Parameters
    ----------
    dataset : list
        zestaw danych
    crossings_by_class : dict
        wynik funkcji compute_crossings_by_class
    window_fast : int
        liczba próbek dla szybkiej średniej
    window_slow : int
        liczba próbek dla wolnej średniej
    min_distance : int
        minimalna odległość między punktami przecięcia

    Returns
    -------
    summary_df : pd.DataFrame
        Podsumowanie dla każdej klasy z liczbą sygnałów i statystykami punktów przecięcia
    """
    summary_rows = []
    classes = sorted({item["class"] for item in dataset})
    
    for cls in classes:
        items = [item for item in dataset if item["class"] == cls]
        num_signals = len(items)
        n_crossings_list = [len(cross["Crossings"]) for cross in crossings_by_class.get(cls, [])]
        
        avg_crossings = np.mean(n_crossings_list) if n_crossings_list else 0
        min_crossings = np.min(n_crossings_list) if n_crossings_list else 0
        max_crossings = np.max(n_crossings_list) if n_crossings_list else 0
        
        summary_rows.append({
            "Class": cls,
            "Num_Signals": num_signals,
            "Avg_Crossings": avg_crossings,
            "Min_Crossings": min_crossings,
            "Max_Crossings": max_crossings,
            "window_fast": window_fast,
            "window_slow": window_slow,
            "min_distance": min_distance,
        })
    
    summary_df = pd.DataFrame(summary_rows)
    return summary_df


# %% ------ METRYKI --------
def evaluate_peak_crossing_alignment(dataset, crossings_by_class,
                                     tolerance=0,
                                     assignments={
                                         "P1": (1, 2),
                                         "P2": (3, 4),
                                         "P3": (5, 6)
                                     }):
    """
    Sprawdza, czy P1, P2, P3 mieszczą się w przedziałach crossingów:
      P1 → między crossingiem 1 i 2
      P2 → między crossingiem 3 i 4
      P3 → między crossingiem 5 i 6

    tolerance — dopuszczalne przesunięcie w indeksach.
    """

    classes = sorted(crossings_by_class.keys())
    summary = []

    for cls in classes:
        items = [item for item in dataset if item["class"] == cls]

        total = 0
        correct_P1 = 0
        correct_P2 = 0
        correct_P3 = 0

        for item in items:
            total += 1
            fname = item["file"]

            entry = next(e for e in crossings_by_class[cls] if e["File"] == fname)
            crossings = entry["Crossings"]

            # potrzebujemy co najmniej 6 crossingów
            if len(crossings) < 6:
                continue

            # indeksy end
            c = [end for (end) in crossings]

            peaks = item.get("peaks_ref", {})

            for peak_name, peak_idx in peaks.items():
                if peak_idx is None:
                    continue

                # który zakres należy sprawdzić
                if peak_name in assignments:
                    a, b = assignments[peak_name]
                    # crossing numerowane 1-based → konwersja:
                    c1 = c[a-1]
                    c2 = c[b-1]

                    # zakres z tolerancją
                    lo = min(c1, c2) - tolerance
                    hi = max(c1, c2) + tolerance

                    if lo <= peak_idx <= hi:
                        if peak_name == "P1":
                            correct_P1 += 1
                        elif peak_name == "P2":
                            correct_P2 += 1
                        elif peak_name == "P3":
                            correct_P3 += 1

        summary.append({
            "Class": cls,
            "Total Signals": total,
            "P1 inside (1-2)%": 100 * correct_P1 / total if total else 0,
            "P2 inside (3-4)%": 100 * correct_P2 / total if total else 0,
            "P3 inside (5-6)%": 100 * correct_P3 / total if total else 0,
        })

    return pd.DataFrame(summary)

def percent_signals_with_six_crossings_and_peaks(dataset, crossings_by_class, tolerance=0):
    """
    Zwraca DataFrame z procentem sygnałów dla każdej klasy, które:
      1. Mają dokładnie 6 crossingów
      2. Piki P1, P2, P3 znajdują się w odpowiednich przedziałach crossingów
    """
    result = []

    assignments = {
        "P1": (1, 2),
        "P2": (3, 4),
        "P3": (5, 6)
    }

    for cls, entries in crossings_by_class.items():
        total = len(entries)
        if total == 0:
            result.append({
                "Class": cls,
                "Total Signals": 0,
                "Signals with 6 crossings": 0,
                "Signals with 6 crossings & correct P1-P3": 0,
                "Percent with 6 crossings": 0,
                "Percent with 6 crossings & correct P1-P3": 0
            })
            continue

        six_cross_signals = []
        correct_peaks_count = 0

        items_cls = [item for item in dataset if item["class"] == cls]

        for entry in entries:
            crossings = entry["Crossings"]
            file_name = entry["File"]

            if len(crossings) == 6:
                six_cross_signals.append(entry)

                # znajdź odpowiadający sygnał w dataset
                item = next(item for item in items_cls if item["file"] == file_name)
                peaks = item.get("peaks_ref", {})

                correct = True
                for peak_name, (a,b) in assignments.items():
                    idx = peaks.get(peak_name)
                    if idx is None:
                        correct = False
                        break
                    # crossingi numerowane 1-based w assignments → 0-based w liście crossings
                    lo = min(crossings[a-1], crossings[b-1]) - tolerance
                    hi = max(crossings[a-1], crossings[b-1]) + tolerance
                    if not (lo <= idx <= hi):
                        correct = False
                        break
                if correct:
                    correct_peaks_count += 1

        total_six_cross = len(six_cross_signals)
        percent_six_cross = 100 * total_six_cross / total
        percent_correct_peaks = 100 * correct_peaks_count / total if total else 0

        result.append({
            "Class": cls,
            "Total Signals": total,
            "Signals with 6 crossings": total_six_cross,
            "Signals with 6 crossings & correct P1-P3": correct_peaks_count,
            "Percent with 6 crossings": percent_six_cross,
            "Percent with 6 crossings & correct P1-P3": percent_correct_peaks
        })

    return pd.DataFrame(result)



# %% ----- PLOT --------
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
                for end in entry["Crossings"]:
                    ax.axvline(t[end], color="red", alpha=alpha_line)

        elif mode == "hist":
            times = [
                t[end]
                for entry in entries
                for (end) in entry["Crossings"]
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
                                      window_fast=2, window_slow=3,
                                      show_lines=True, show_points=True):

    """
    Rysuje sygnał z klasy `cls` o indeksie `idx` wraz z:
        - fast i slow średnimi ruchomymi
        - wykrytymi crossingami MA
    """

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
    plt.plot(t, x, color="black", linewidth=0.8, label="Sygnał ICP")
    plt.plot(t, xf, color="blue", linestyle="--", linewidth=1.5, label=f"Średnia szybka (okno: {window_fast} próbki)")
    plt.plot(t, xs, color="orange", linestyle="--", linewidth=1.5, label=f"Średnia wolna (okno: {window_slow} próbki)")

    # crossingi
    if show_lines:
        for end in crossings:
            plt.axvline(t[end], color="red", alpha=0.5)
    if show_points:
        for end in crossings:
            plt.scatter(t[end], x[end], color="red", s=50, zorder=5)

    #plt.title(f"{cls} — {file_name} — {len(crossings)} crossings")
    plt.xlabel("Numer próbki")
    plt.ylabel("Amplituda")
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
                        # start = max(0, i - lookback)
                        crossings.append((i))
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


def plot_signals_with_n_crossings(dataset, crossings_by_class, n_crossings=1, classes_to_plot=[1,2,3],
                                  peak_colors={"P1":"green","P2":"orange","P3":"black"},
                                  crossing_color="red"):
    """
    Rysuje wszystkie sygnały z dokładnie `n_crossings` dla podanych klas.
    
    Parameters
    ----------
    dataset : list
        Lista sygnałów z load_data()
    crossings_by_class : dict
        Wynik compute_crossings_summary
    n_crossings : int
        Dokładna liczba crossingów do wybrania
    classes_to_plot : list
        Lista numerów klas, np. [1,2,3]
    peak_colors : dict
        Kolory dla referencyjnych pików
    crossing_color : str
        Kolor dla crossingów
    """

    import math
    classes_to_plot = [f"Class{i}" for i in classes_to_plot]

    for cls in classes_to_plot:
        items = [item for item in dataset if item["class"] == cls]

        # filtrowanie sygnałów z dokładnie n_crossings
        selected_items = []
        for item in items:
            entry = next(e for e in crossings_by_class[cls] if e["File"] == item["file"])
            if len(entry["Crossings"]) == n_crossings:
                selected_items.append((item, entry["Crossings"]))

        n_signals = len(selected_items)
        if n_signals == 0:
            print(f"Klasa {cls}: brak sygnałów z dokładnie {n_crossings} crossingami")
            continue

        # ustawienia subplotów
        n_cols = 5
        n_rows = math.ceil(n_signals / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 2*n_rows), sharex=True, sharey=True)
        axes = axes.flatten()

        for i, (item, crossings) in enumerate(selected_items):
            t = item["signal"].iloc[:,0].values
            x = item["signal"].iloc[:,1].values
            ax = axes[i]

            # sygnał
            ax.plot(t, x, color='blue')

            # piki referencyjne
            peaks = item.get("peaks_ref", {})
            for p_name, idx in peaks.items():
                if idx is not None and 0 <= idx < len(t):
                    ax.plot(t[idx], x[idx], 'o', color=peak_colors.get(p_name, 'black'), alpha=0.8)

            # crossingi
            for end in crossings:
                ax.axvline(t[end], color=crossing_color, alpha=0.6)
                ax.scatter(t[end], x[end], color=crossing_color, s=40)

            ax.set_title(f"{item['file']}")
            ax.grid(True, alpha=0.3)

        # Wyłącz puste subploty
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f"Klasa {cls} — sygnały z {n_crossings} crossingami", fontsize=16)
        plt.tight_layout()
        plt.show()


# def plot_all_signals_class_with_peaks_and_crossings(dataset, crossings_by_class, cls,
#                                                     peak_colors={"P1":"green","P2":"orange","P3":"black"},
#                                                     crossing_color="red",
#                                                     start_idx=None, end_idx=None):
#     """
#     Rysuje wszystkie sygnały danej klasy:
#       • sygnał (lub fragment start_idx:end_idx)
#       • piki referencyjne (P1, P2, P3)
#       • MA-crossingi (normalne)
#     """

#     items = [item for item in dataset if item["class"] == cls]
#     n_signals = len(items)

#     if n_signals == 0:
#         print(f"Brak sygnałów dla klasy {cls}")
#         return

#     # Ustawienia subplotów
#     import math
#     n_cols = 5
#     n_rows = math.ceil(n_signals / n_cols)

#     fig, axes = plt.subplots(n_rows, n_cols,
#                              figsize=(4*n_cols, 2*n_rows),
#                              sharex=True, sharey=True)
#     axes = axes.flatten()

#     for i, item in enumerate(items):
#         t = item["signal"].iloc[:, 0].values
#         x = item["signal"].iloc[:, 1].values

#         # wybrany zakres
#         s_idx = 0 if start_idx is None else start_idx
#         e_idx = len(x) if end_idx is None else end_idx
#         t_seg = t[s_idx:e_idx]
#         x_seg = x[s_idx:e_idx]

#         ax = axes[i]
#         ax.plot(t_seg, x_seg, color='blue', linewidth=1.0)

#         # ===== Piki referencyjne =====
#         peaks = item.get("peaks_ref", {})
#         for pname, idx in peaks.items():
#             if idx is not None and s_idx <= idx < e_idx:
#                 ax.plot(t[idx], x[idx], 'o',
#                         color=peak_colors.get(pname, "black"),
#                         markersize=6)

#         # ===== Crossing =====
#         file_name = item["file"]
#         entry = next(e for e in crossings_by_class[cls] if e["File"] == file_name)
#         crossings = entry["Crossings"]

#         for start, end in crossings:
#             if s_idx <= end < e_idx:
#                 ax.axvline(t[end], color=crossing_color, alpha=0.4)
#                 ax.scatter(t[end], x[end], color=crossing_color, s=25)

#         ax.set_title(file_name, fontsize=8)
#         ax.grid(True, alpha=0.3)

#     # Ukryj puste subploty
#     for j in range(i + 1, len(axes)):
#         axes[j].set_visible(False)

#     fig.suptitle(f"Klasa {cls} — sygnały z pikami i crossingami (zakres indeksów)",
#                  fontsize=16)
#     plt.tight_layout()
#     plt.show()

def plot_all_signals_class_with_peaks_and_crossings(
        dataset, crossings_by_class, cls,
        peak_colors={"P1":"green","P2":"orange","P3":"black"},
        crossing_color="red",
        start_idx=None, end_idx=None):
    """
    Rysuje SYGNAŁY z danej klasy, od start_idx do end_idx (numer sygnału w klasie!).
    Dla każdego:
        • pełny sygnał
        • piki referencyjne
        • crossingi
    """

    import math

    # Pobranie sygnałów klasy
    items_all = [item for item in dataset if item["class"] == cls]

    if len(items_all) == 0:
        print(f"Brak sygnałów dla klasy {cls}")
        return

    # --- tu zmiana! --- WYBIERAMY SYGNAŁY, nie zakres próbek
    s = 0 if start_idx is None else start_idx
    e = len(items_all) if end_idx is None else end_idx
    items = items_all[s:e]

    n_signals = len(items)

    # siatka subplotów
    n_cols = 5
    n_rows = math.ceil(n_signals / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4*n_cols, 2*n_rows),
        sharex=False, sharey=False
    )
    axes = axes.flatten()

    for i, item in enumerate(items):
        t = item["signal"].iloc[:, 0].values
        x = item["signal"].iloc[:, 1].values
        ax = axes[i]

        # sygnał
        ax.plot(t, x, color='blue', linewidth=1.0)

        # ----- piki referencyjne -----
        peaks = item.get("peaks_ref", {})
        for pname, idx in peaks.items():
            if idx is not None:
                ax.plot(t[idx], x[idx], 'o',
                        color=peak_colors.get(pname, "black"),
                        markersize=6)

        # ----- crossingi -----
        entry = next(e for e in crossings_by_class[cls] 
                     if e["File"] == item["file"])

        for end in entry["Crossings"]:
            ax.axvline(t[end], color=crossing_color, alpha=0.4)
            ax.scatter(t[end], x[end], color=crossing_color, s=25)

        ax.set_title(item["file"], fontsize=8)
        ax.grid(True, alpha=0.3)

    # Ukrycie pustych subplotów
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Klasa {cls} — sygnały {s} do {e} z pikami i crossingami",
        fontsize=16
    )
    plt.tight_layout()
    plt.show()



# %% ------------ EXECUTION --------------

if __name__ == "__main__":
    
    base_path = r"C:\Users\User\OneDrive\Dokumenty\praca inżynierska\ICP_pulses_it1"
    data_crude = load_dataset(base_path, "it1")
    data = smooth_dataset(data_crude)
    
    base_path_2 = r"C:\Users\User\OneDrive\Dokumenty\praca inżynierska\ICP_pulses_it2"
    data2_crude = load_dataset(base_path_2, "it2")
    data2 = smooth_dataset(data2_crude)
    
    pd.options.display.float_format = '{:.4f}'.format
    
    
    
    cross = compute_crossings(
        data2,
        window_fast=2,
        window_slow=4,
        min_distance=0,
        max_crossings=6
    )
    
    summary_df = compute_crossings_summary(data2, cross, window_fast=2, window_slow=4)
    
    # Użycie:
    df_six_cross = percent_signals_with_six_crossings_and_peaks(data2, cross)
    print(df_six_cross)
    # print(summary)
    
    # plot_crossings_overview(data, cross, mode="lines")
    # plot_crossings_overview(data, cross, mode="hist")
    
    # plot_crossings_single_signal(data, cross, "Class2", 122, 
    #                             window_fast=2, window_slow=4, show_lines=True, show_points=True)
    
    
    # min_distances = [20, 25, 30, 40, 50]
    # summary_df = sweep_min_distance_all_classes(data, min_distances)
    # print(summary_df)
    
    # plot_all_signals_class_with_peaks(data, "Class1")
    
    #plot_signals_with_n_crossings(data, cross, n_crossings=5, classes_to_plot=[4])
    
    # rysowanie tylko fragmentu sygnału od indeksu 50 do 150
    # plot_all_signals_class_with_peaks_and_crossings(data, cross, "Class2",
    #                                                  start_idx=40, end_idx=50)
    
    # metrics = evaluate_peak_crossing_alignment(data, cross, tolerance=0)
    # print(metrics)
    
    
    # print("Kolejny sukces")