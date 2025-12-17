# %% ------ IMPORTS -------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.signal import find_peaks, hilbert, find_peaks_cwt
import itertools
import time
from moving_average import compute_crossings, smooth_dataset
from data_handling import load_dataset
from functools import reduce

pd.options.display.float_format = '{:.4f}'.format



def wyniki(
    dataset, 
    dataset_name,      # "it1" lub "it2"
    range_type=None,
    peak_ranges_file=None,
    peak_amps_file=None,
    range_name=None,
    methods=None       # lista metod do liczenia
):
    """
    Liczy lub wczytuje metryki dla danej konfiguracji danych i smoothingu.
    Każda metoda zapisuje swoje wyniki osobno w folderze dedykowanym dla konfiguracji.
    
    Parameters:
    ----------
    dataset : list
        dane (sygnaly+piki)
    dataset_name : str
        nazwa zestawu danych
    range_type : 
    
   
        
    """
    if methods is None:
        methods = METHODS.keys()
    

    # unikalny folder dla konfiguracji
    # range_str = range_type if range_type else "no_range"

    folder_wyniki = os.path.join("wyniki", f"{dataset_name}")
    os.makedirs(folder_wyniki, exist_ok=True)

    results = {}

    for method_name in methods:
        all_metrics_file = os.path.join(folder_wyniki, f"{method_name}_all_metrics.csv")
        avg_metrics_file = os.path.join(folder_wyniki, f"{method_name}_avg_metrics.csv")

        # jeśli pliki istnieją, wczytaj je
        if os.path.exists(all_metrics_file) and os.path.exists(avg_metrics_file):
            df_all_metrics = pd.read_csv(all_metrics_file)
            df_avg_metrics = pd.read_csv(avg_metrics_file)
            print(f"Wczytano zapisane wyniki: {method_name}")
        else:
            if range_type == "crossing":
                crossings_by_class = compute_crossings(dataset)
            # liczenie metryk
            all_metrics = []
            for cls in ["Class1", "Class2", "Class3", "Class4"]:
                for peak_name in ["P1","P2","P3"]:
                    detected_all = []
                    for item in dataset:
                        if item["class"] != cls:
                            continue
                        detected = single_peak_detection(
                            peak_name=peak_name,
                            class_name=cls,
                            file_name=item["file"],
                            dataset=[item],
                            method_name=method_name,
                            range_type=range_type,
                            peak_ranges_file=peak_ranges_file,
                            peak_amps_file=peak_amps_file
                        )
                        detected_all.extend(detected)

                    df_metrics = compute_peak_metrics(
                        dataset=[item for item in dataset if item["class"] == cls],
                        detected_peaks=detected_all,
                        peak_name=peak_name,
                        class_name=cls
                    )
                    df_metrics["Method"] = method_name
                    all_metrics.append(df_metrics)

            df_all_metrics = pd.concat(all_metrics, ignore_index=True)
            df_avg_metrics = df_all_metrics.groupby(["Class", "Peak", "Method"]).mean(numeric_only=True).reset_index()

            # zapis do plików
            df_all_metrics.to_csv(all_metrics_file, index=False)
            df_avg_metrics.to_csv(avg_metrics_file, index=False)
            print(f"Obliczono i zapisano wyniki: {method_name}")

        results[method_name] = (df_all_metrics, df_avg_metrics)

    return results

def wyniki_crossings(
    dataset,
    dataset_name,
    crossing_ranges,
    methods
):
    folder_wyniki = os.path.join("wyniki", f"{dataset_name}")
    os.makedirs(folder_wyniki, exist_ok=True)

    results = {}

    for method_name in methods:
        all_metrics_file = os.path.join(folder_wyniki, f"{method_name}_all_metrics.csv")
        avg_metrics_file = os.path.join(folder_wyniki, f"{method_name}_avg_metrics.csv")

        if os.path.exists(all_metrics_file) and os.path.exists(avg_metrics_file):
            df_all = pd.read_csv(all_metrics_file)
            df_avg = pd.read_csv(avg_metrics_file)
            results[method_name] = (df_all, df_avg)
            continue

        all_metrics = []

        for cls in ["Class1", "Class2", "Class3"]:
            for peak_name in ["P1", "P2", "P3"]:

                detected_all = []

                for item in dataset:
                    if item["class"] != cls:
                        continue

                    file_name = item["file"]
                    if file_name not in crossing_ranges:
                        continue
                    if peak_name not in crossing_ranges[file_name]:
                        continue

                    detected = single_peak_detection(
                        peak_name=peak_name,
                        class_name=cls,
                        file_name=file_name,
                        dataset=[item],
                        method_name=method_name,
                        range_type="crossing",
                        peak_ranges_file=crossing_ranges[file_name],
                        peak_amps_file=None
                    )
                    detected_all.extend(detected)

                if not detected_all:
                    continue

                df_metrics = compute_peak_metrics(
                    dataset=[item for item in dataset if item["class"] == cls],
                    detected_peaks=detected_all,
                    peak_name=peak_name,
                    class_name=cls
                )
                df_metrics["Method"] = method_name
                all_metrics.append(df_metrics)

        if not all_metrics:
            df_all = pd.DataFrame(columns=["Class", "Peak", "File", "Mean_X_Error", "Mean_Y_Error",
                                       "Mean_XY_Error", "Min_XY_Error", "Peak_Count",
                                       "Reference_Peaks", "Detected_Peaks",
                                       "Num_Signals_in_Class", "Num_Signals_with_Peak"])
            df_avg = df_all.copy()
            results[method_name] = (df_all, df_avg)
            continue

        df_all = pd.concat(all_metrics, ignore_index=True)
        df_avg = (
            df_all
            .groupby(["Class", "Peak", "Method"])
            .mean(numeric_only=True)
            .reset_index()
        )

        df_all.to_csv(all_metrics_file, index=False)
        df_avg.to_csv(avg_metrics_file, index=False)

        results[method_name] = (df_all, df_avg)

    return results


def consolidate_results(results_dict, avg_index=1):
    """
    results_dict: słownik metod -> krotka(DataFrame_all, DataFrame_avg)
    avg_index: który element krotki wybrać (domyślnie 1 -> df_avg)
    """
    dfs = []
    for method_name, dfs_tuple in results_dict.items():
        df = dfs_tuple[avg_index].copy()
        df["Method"] = method_name  # upewniamy się, że jest kolumna Method
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def top5_configs(df, peak_name, class_name, metric="XY_Error"):
    """
    Zwraca top 5 konfiguracji dla danego piku i klasy,
    dla każdego pliku osobno.
    """
    df_filtered = df[(df["Peak"] == peak_name) & (df["Class"] == class_name)]
    
    # Grupujemy po pliku i sortujemy według metryki rosnąco (bo chcemy najmniejszy błąd)
    top5_per_file = df_filtered.groupby("File").apply(
        lambda x: x.nsmallest(5, metric)
    ).reset_index(drop=True)
    
    return top5_per_file


# %% ----------- RANGES FROM CROSSINGS -------
def get_peak_range_from_crossings(crossings_for_file, peak_name):
    """
    crossings_for_file: lista 6 crossingów dla jednego pliku
    peak_name: "P1", "P2", "P3"
    window_before/after: opcjonalna tolerancja wokół zakresu
    """
    
    if len(crossings_for_file) < 6:
        return None, None
    else:
        if peak_name == "P1":
            start, end = crossings_for_file[0], crossings_for_file[1]
        elif peak_name == "P2":
            start, end = crossings_for_file[2], crossings_for_file[3]
        elif peak_name == "P3":
            start, end = crossings_for_file[4], crossings_for_file[5]
        else:
            raise ValueError(f"Nieznany peak_name: {peak_name}")
    
    return start, end
    
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

def modified_scholkmann(signal, scale=1, threshold=99):
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
    threshold = np.percentile(maxima_strength, threshold)  # tylko te najstabilniejsze
    peaks = np.where(maxima_strength >= threshold)[0]

    return np.array(peaks)

def curvature(signal, d2x_threshold=-0.0015, prominence=0.005, threshold=None):
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

def line_distance(signal, d2x_threshold=0.02, mode="perpendicular", min_len=3):
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

def hilbert_envelope(signal, prominence=0):
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)  # amplituda "obwiedni"

    # detekcja pików na envelope
    peaks, _ = find_peaks(envelope, prominence)
    
    return np.array(peaks)

def wavelet(signal, prominence=0, w_range=(1,10), step=1):
    # Continuous Wavelet Transform
    widths = np.arange(w_range[0], w_range[1] + 1, step)  # zakres szerokości pików
    peaks = find_peaks_cwt(signal, widths)
    
    return np.array(peaks)

# %% -------- DETECTION -------------

def single_peak_detection_old2(peak_name, class_name, file_name, dataset, method_name, range_type="static",
                          peak_ranges_file=None, peak_amps_file=None):
    """
    Wykrywa piki w określonym przedziale czasowym dla jednego pliku.
    """
    if method_name not in METHODS:
        raise ValueError(f"Nieznana metoda: {method_name}")

    detect = METHODS[method_name]

    # --- obsługa zakresów czasowych i amplitud ---
    if peak_ranges_file==None:
        t_start, t_end = -np.inf, np.inf
    elif range_type=="crossing":
        t_start, t_end = peak_ranges_file[peak_name]
    else:
        if class_name in peak_ranges_file and peak_name in peak_ranges_file[class_name]:
            t_start, t_end = peak_ranges_file[class_name][peak_name]
        else: 
            t_start, t_end = None, None
    # if t_range is None or any(v is None for v in t_range):
    #     return []

    if peak_amps_file==None:
        a_start, a_end = -np.inf, np.inf
    elif range_type=="crossing":
        a_start, a_end = peak_amps_file[peak_name]
    else:
        if class_name in peak_amps_file and peak_name in peak_amps_file[class_name]:
            a_start, a_end = peak_amps_file[class_name][peak_name]  # brak ograniczeń amplitudy
        else: 
            t_start, t_end = None, None
    # --- wykrywanie piku ---
    detected_all = []

    for item in dataset:
        if item["class"] != class_name or item["file"] != file_name:
            continue

        signal_df = item["signal"]
        t = signal_df.iloc[:, 0].values
        y = signal_df.iloc[:, 1].values

        peaks = detect(y)
        peaks = np.array(peaks, dtype=int)

        # filtruj tylko te w zadanym zakresie czasu i amplitudy
        peaks = peaks[(t[peaks] >= t_start) & (t[peaks] <= t_end)]
        peaks = peaks[(y[peaks] >= a_start) & (y[peaks] <= a_end)]

        detected_all.append(peaks)

    return detected_all

def single_peak_detection(peak_name, class_name, file_name, dataset, method_name,
                          range_type="static", peak_ranges_file=None, peak_amps_file=None):
    """
    Wykrywa piki w określonym przedziale czasowym dla jednego pliku.
    Zwraca NaN jeśli brak klasy lub piku.
    """
    if method_name not in METHODS:
        raise ValueError(f"Nieznana metoda: {method_name}")

    detect = METHODS[method_name]

    # --- obsługa zakresów czasowych i amplitud ---
    # domyślnie brak ograniczeń
    t_start, t_end = -np.inf, np.inf
    a_start, a_end = -np.inf, np.inf

    if peak_ranges_file is not None:
        if range_type == "crossing":
            # peak_ranges_file może być dla pojedynczego pliku: {"P1":(start,end), ...}
            t_start, t_end = peak_ranges_file.get(peak_name, (np.nan, np.nan))
        else:
            t_start, t_end = peak_ranges_file.get(class_name, {}).get(peak_name, (np.nan, np.nan))

    if peak_amps_file is not None:
        if range_type == "crossing":
            a_start, a_end = peak_amps_file.get(peak_name, (np.nan, np.nan))
        else:
            a_start, a_end = peak_amps_file.get(class_name, {}).get(peak_name, (np.nan, np.nan))

    detected_all = []

    # Filtrujemy sygnały dla danej klasy i pliku
    class_items = [item for item in dataset if item["class"] == class_name and item["file"] == file_name]
    if len(class_items) == 0:
        # brak klasy → zwracamy NaN
        detected_all.append(np.array([]))
        return detected_all

    item = class_items[0]
    signal_df = item["signal"]
    t = signal_df.iloc[:, 0].values
    y = signal_df.iloc[:, 1].values

    # wykryj piki
    peaks = detect(y)
    peaks = np.array(peaks, dtype=int)

    # jeśli zakresy są NaN → pomin filtrowanie
    if not np.isnan(t_start) and not np.isnan(t_end):
        peaks = peaks[(t[peaks] >= t_start) & (t[peaks] <= t_end)]
    if not np.isnan(a_start) and not np.isnan(a_end):
        peaks = peaks[(y[peaks] >= a_start) & (y[peaks] <= a_end)]

    detected_all.append(peaks)
    return detected_all


# %% --------- METRICS, PLOTS ------
def compute_peak_metrics(dataset, detected_peaks, peak_name, class_name):
    metrics_list = []
    class_signals = [item for item in dataset if item["class"] == class_name]
    num_signals_in_class = len(class_signals)
    
    for item, peaks in zip(class_signals, detected_peaks):
        file_name = item["file"]
        signal_df = item["signal"]
        t = signal_df.iloc[:, 0].values
        y = signal_df.iloc[:, 1].values

        ref_idx = item["peaks_ref"].get(peak_name)
        if ref_idx is None or len(peaks) == 0:
            metrics_list.append({
                "Class": class_name,
                "Peak": peak_name,
                "File": file_name,
                "Mean_X_Error": np.nan,
                "Mean_Y_Error": np.nan,
                "Mean_XY_Error": np.nan,
                "Min_XY_Error": np.nan,
                "Peak_Count": len(peaks),
                "Reference_Peaks": ref_idx, 
                "Detected_Peaks": list(peaks)
            })
            continue

        t_detected = t[peaks]
        y_detected = y[peaks]

        dx = abs(t_detected - t[ref_idx])
        dy = abs(y_detected - y[ref_idx])
        dxy = np.sqrt(dx**2 + dy**2)

        metrics_list.append({
            "Class": class_name,
            "Peak": peak_name,
            "File": file_name,
            "Mean_X_Error": np.mean(dx),
            "Mean_Y_Error": np.mean(dy),
            "Mean_XY_Error": np.mean(dxy),
            "Min_XY_Error": np.min(dxy),
            "Peak_Count": len(peaks),
            "Reference_Peaks": ref_idx, 
            "Detected_Peaks": list(peaks)
        })
    df_metrics = pd.DataFrame(metrics_list)
    # liczba sygnałów z wykrytym pikem
    num_detected = df_metrics["Peak_Count"].gt(0).sum()
    df_metrics["Num_Signals_in_Class"] = num_signals_in_class
    df_metrics["Num_Signals_with_Peak"] = num_detected

    return df_metrics


def plot_class_peaks_grid(dataset, results_df, method_name, mode="static"):
    """
    Grid 4x3: każdy subplot = (Class, Peak)
    Ostatni subplot (Class4) jest wyśrodkowany, jeśli istnieje.
    mode = "static" lub "crossing"
    """
    tasks = [
        ("Class1", "P1"), ("Class1", "P2"), ("Class1", "P3"),
        ("Class2", "P1"), ("Class2", "P2"), ("Class2", "P3"),
        ("Class3", "P1"), ("Class3", "P2"), ("Class3", "P3"),
        ("Class4", "P2")  # ostatni, wyśrodkowany
    ]
    
    method_titles_pl = {
        "max_in_concave": "Maksima w odcinkach wklęsłych",
        "scholkmann_mod": "Zmodyfikowana metoda Scholkmanna",
        "curvature": "Maksymalna krzywizna",
        "max_dist_line": "Max, odległość od linii łączącej końce odc. wklęsłego",
        "hilbert": "Transformata Hilberta",
        "cwt": "Ciągła transformata falkowa"
    }

    method_title = method_titles_pl.get(method_name, method_name)
    
    
    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    axes = axes.flatten()

    # pozycje do rysowania
    plot_positions = list(range(9)) + [10]  # ostatni subplot na środku ostatniego rzędu

    # domyślnie wszystkie ax niewidoczne
    for ax in axes:
        ax.set_visible(False)

    for (cls, pk), ax_i in zip(tasks, plot_positions):
        # sprawdź, czy w dataset są dane dla tej klasy
        class_items = [item for item in dataset if item["class"] == cls]
        if len(class_items) == 0:
            continue  # np. Class4 w crossingach
        
        cls_pl = {
            "Class1": "Klasa 1",
            "Class2": "Klasa 2",
            "Class3": "Klasa 3",
            "Class4": "Klasa 4"
        }[cls]
        
        ax = axes[ax_i]
        ax.set_visible(True)

        signals = [item["signal"] for item in class_items]
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

        # filtr dla metody i piku
        df_method = results_df[
            (results_df["Class"] == cls) &
            (results_df["Peak"] == pk) &
            (results_df["Method"] == method_name)
        ]

        manual_t, auto_t = [], []

        for item in class_items:
            fname = item["file"]
            sig = item["signal"]
            t = sig.iloc[:, 0].values

            # referencyjne piki
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

        # histogram piki
        ax2 = ax.twinx()
        if len(manual_t) > 0:
            ax2.hist(manual_t, bins=40, color='green', alpha=0.3, label='Piki referencyjne')
        if len(auto_t) > 0:
            ax2.hist(auto_t, bins=40, color='red', alpha=0.3, label='Piki wykryte')

        ax.set_title(f"{cls_pl} – {pk}", fontsize=17)
        ax.set_xlabel("Numer próbki", fontsize=13)
        ax.set_ylabel("Amplituda", fontsize=13)
        ax.grid(True, alpha=0.3)

    # globalna legenda
    
    legend_ax = axes[11]   # ostatnia wolna komórka siatki
    legend_ax.set_visible(True)
    legend_ax.axis("off")
    
    handles = [
        plt.Line2D([0], [0], color='blue', lw=2, label='Średni sygnał'),
        plt.Rectangle((0, 0), 1, 1, color='green', alpha=0.3, label="Piki referencyjne"),
        plt.Rectangle((0, 0), 1, 1, color='red', alpha=0.3, label="Piki wykryte")
    ]
    # fig.legend(handles, [h.get_label() for h in handles], loc='upper right', fontsize=12)
    legend_ax.legend(handles, [h.get_label() for h in handles],
                     fontsize=14, loc="center")
    # title_mode = "Zakresy stałe" if mode == "static" else "Crossingi"
    fig.suptitle(f"{method_title}\nTryb: {mode}", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_examples(data, dataset_name, example_indices=range(0,3)):
    """
    Rysuje sygnały dla wybranych przykładów w klasie.
    - data: lista słowników po load_data / load_data_it2
    - dataset_name: 'IT1' lub 'IT2' (tylko dla tytułu)
    - example_indices: iterable z numerami przykładów w klasie do narysowania
    """
    classes = sorted(list(set([item["class"] for item in data])))
    n_examples_per_class = len(example_indices)
    fig, axes = plt.subplots(len(classes), n_examples_per_class, figsize=(17,12))
    axes = axes.flatten()

    count = 0
    for cls in classes:
        class_items = [item for item in data if item["class"]==cls]
        for idx in example_indices:
            if idx >= len(class_items):
                continue  # zabezpieczenie
            item = class_items[idx]
            ax = axes[count]
            y = item["signal"].iloc[:,1].values
            t = item["signal"].iloc[:,0].values

            ax.plot(t, y, color='black', lw=1.2)

            # piki referencyjne
            peaks_ref = item["peaks_ref"]
            for pk in ["P1","P2","P3"]:
                peak_idx = peaks_ref.get(pk)
                if peak_idx is not None and 0 <= peak_idx < len(y):
                    ax.scatter(t[peak_idx], y[peak_idx], color='red', s=50)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_position('zero')
            ax.spines['left'].set_position('zero')
            
            ax.set_xlabel("Numer próbki", fontsize=10)
            ax.set_ylabel("Amplituda", fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.axis('on')  # włącz osie
            count += 1

    #plt.suptitle(f"Przykłady sygnałów {dataset_name}", fontsize=16)
    plt.tight_layout(pad=1.5)
    plt.show()

 
def plot_single_signal(dataset, class_name, index, show_peaks=True):
    # Filtrujemy sygnały dla danej klasy
    class_items = [item for item in dataset if item["class"] == class_name]
    if index >= len(class_items):
        print(f"Index {index} wykracza poza liczbę sygnałów w {class_name} ({len(class_items)}).")
        return
    
    item = class_items[index]
    signal_df = item["signal"]
    t = signal_df.iloc[:, 0].values
    y = signal_df.iloc[:, 1].values
    
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(t, y, color='black', lw=1.5, label="Sygnał ICP")
    
    if show_peaks:
        peaks_ref = item["peaks_ref"]
        for pk in ["P1", "P2", "P3"]:
            peak_idx = peaks_ref.get(pk)
            if peak_idx is not None and 0 <= peak_idx < len(y):
                ax.scatter(t[peak_idx], y[peak_idx], s=60, label=f"Pik {pk}", zorder=5)
    
    ax.set_title(f"{class_name} – sygnał nr {index}", fontsize=16)
    ax.set_xlabel("Numer próbki", fontsize=12)
    ax.set_ylabel("Amplituda", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return fig, ax
    
    
# %% ------ PARAMETRY ---------------
METHODS = {
    "concave": lambda sig: concave(sig, 0, 0, None),
    "concave_d2x=-0,002": lambda sig:  concave(sig, -0.002, 0, None),
    "concave_d2x=0,002": lambda sig:  concave(sig, 0.002, 0, None),
    
    "modified_scholkmann_1_99": lambda sig: modified_scholkmann(sig, 1, 99),
    "modified_scholkmann_1_95": lambda sig: modified_scholkmann(sig, 1, 95),
    "modified_scholkmann_1-2_95": lambda sig: modified_scholkmann(sig, 2, 95),
    "modified_scholkmann_1-2_99": lambda sig: modified_scholkmann(sig, 2, 99),
    
    "curvature": lambda sig: curvature(sig, 0, 0, None),
    "line_distance_10": lambda sig: line_distance(sig, 0,"vertical", 10),
    "hilbert": lambda sig: hilbert_envelope(sig, 0),
    "wavelet": lambda sig: wavelet(sig, 0)
}

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

# %% ------- EXECUTION -----------
# %% -------- LOAD DATA ------
base_path = r"C:\Users\User\OneDrive\Dokumenty\praca inżynierska\ICP_pulses_it1"
it1 = load_dataset(base_path, "it1")
# data_raw_it1 = data_it1
it1_sm_4Hz = smooth_dataset(it1, cutoff=4, inplace=False)
it1_sm_3Hz = smooth_dataset(it1, cutoff=3, inplace=False)


base_path_2 = r"C:\Users\User\OneDrive\Dokumenty\praca inżynierska\ICP_pulses_it2"
it2 = load_dataset(base_path_2, "it2")
# data_raw_it2 = data_it2
it2_sm_4Hz = smooth_dataset(it2, cutoff=4, inplace=False)
it2_sm_3Hz = smooth_dataset(it1, cutoff=3, inplace=False)

# %% -------- PARAMETRY -----------
SMOOTHING = {
    "it1": {
        "none": it1,
        "4Hz": it1_sm_4Hz,
        "3Hz": it1_sm_3Hz,
    },
    "it2": {
        "none": it2,
        "4Hz": it2_sm_4Hz,
        "3Hz": it2_sm_3Hz,
    }
}

RANGES = {
    "none": {
        "type": None
    },
    "full": {
        "type": "static",
        "time": time_full,
        "amps": amps_full
    },
    "pm3": {
        "type": "static",
        "time": time_pm3,
        "amps": amps_pm3
    },
    "whiskers": {
        "type": "static",
        "time": time_whiskers,
        "amps": amps_whiskers
    },
    "crossings": {
        "type": "crossing",
    }
}

# %% ------------ WSZYSTKIE WYNIKI ------------
METHODS_LIST = list(METHODS.keys())
ALL_RESULTS = {}

for dataset_name in ["it1", "it2"]:

    for smoothing_name, dataset in SMOOTHING[dataset_name].items(): # for key, value in dict.items():

        # 🔹 crossing liczymy raz na (dataset × smoothing)
        crossing_ranges = None

        for range_name, range_cfg in RANGES.items():

            config_name = f"{dataset_name}_sm-{smoothing_name}_range-{range_name}"
            print(f"\n=== {config_name} ===")

            results_per_method = {}

            # ---------- CROSSINGS ----------
            if range_cfg["type"] == "crossing":

                if crossing_ranges is None:
                    print("Crossing ranges to None")
                    crossing_ranges, _ = compute_crossings(dataset)
                    for method_name in METHODS_LIST:
                        df_all, df_avg = wyniki_crossings(
                            dataset=dataset,
                            dataset_name=config_name,
                            crossing_ranges=crossing_ranges,
                            methods=[method_name]
                        )[method_name]  # wyniki zwracane są w słowniku {method: (df_all, df_avg)}
                        results_per_method[method_name] = (df_all, df_avg)
                print("Nie wchodzi fdo petli wynikow")

            # ---------- STATIC / NONE ----------
            else:
                for method_name in METHODS_LIST:
                    df_all, df_avg = wyniki(
                        dataset=dataset,
                        dataset_name=config_name,
                        range_type=range_cfg["type"],
                        peak_ranges_file=range_cfg.get("time"),
                        peak_amps_file=range_cfg.get("amps"),
                        range_name=range_name if range_cfg["type"] else None,
                        methods=[method_name]  # zwracamy po jednej metodzie
                    )[method_name]  # wyniki zwracane są w słowniku {method: (df_all, df_avg)}
                    results_per_method[method_name] = (df_all, df_avg)

            ALL_RESULTS[config_name] = results_per_method


"""
METHODS_LIST = list(METHODS.keys())


ALL_RESULTS = {}

for dataset_name in ["it1", "it2"]:

    for smoothing_name, dataset in SMOOTHING[dataset_name].items():

        for range_name, range_cfg in RANGES.items():

            config_name = f"{dataset_name}_sm-{smoothing_name}_range-{range_name}"
            print(f"\n=== {config_name} ===")

            # ---------- CROSSINGS ----------
            if range_cfg["type"] == "crossing":
                results_all_files = []

                for item in dataset:  # 🔹 iteracja po plikach
                    file_name = item["file"]

                    # liczymy crossing tylko dla tego pliku
                    crossing_ranges_file, crossing_metrics_file = compute_crossings_summary([item])

                    results_file = wyniki_crossings(
                        dataset=[item],
                        dataset_name=config_name,
                        crossing_ranges=crossing_ranges_file,
                        methods=METHODS_LIST
                    )

                    results_all_files.append(results_file)

                # łączymy wyniki wszystkich plików w jeden słownik/listę
                ALL_RESULTS[config_name] = results_all_files
            # ---------- STATIC / NONE ----------
            else:
                results = wyniki(
                    dataset=dataset,
                    dataset_name=config_name,
                    range_type=range_cfg["type"],
                    peak_ranges_file=range_cfg.get("time"),
                    peak_amps_file=range_cfg.get("amps"),
                    range_name=range_name if range_cfg["type"] else None,
                    methods=METHODS_LIST
                )

            ALL_RESULTS[config_name] = results


# Filtrujemy tylko IT2
it2_results = {k: v for k, v in ALL_RESULTS.items() if k.startswith("it2")}

# Lista DataFrame'ów do połączenia
dfs = []

for config_name, (df_all, df_avg) in it2_results.items():
    df = df_all.copy()
    df["Config"] = config_name  # zapisujemy konfigurację
    dfs.append(df)

df_it2_all = pd.concat(dfs, ignore_index=True)

peaks = ["P1","P2","P3"]
classes = ["Class1","Class2","Class3","Class4"]

best_configs = []

for cls in classes:
    for pk in peaks:
        top_xy = top5_configs(df_it2_all, pk, cls, metric="Mean_XY_Error")
        top_minxy = top5_configs(df_it2_all, pk, cls, metric="Min_XY_Error")
        
        best_configs.append({
            "Class": cls,
            "Peak": pk,
            "Metric": "XY_Error",
            "Top5": top_xy
        })
        best_configs.append({
            "Class": cls,
            "Peak": pk,
            "Metric": "Min_XY_Error",
            "Top5": top_minxy
        })

"""
"""
# ------------------ NORMAL, NO RANGES ------------------
results_normal_no_ranges = wyniki(
    dataset=it1,
    dataset_name="it1",
    range_type=None,   # brak zakresów
    methods=METHODS.keys()
)

# ------------------ NORMAL, RANGES MINMAX ------------------
results_normal_ranges = wyniki(
    dataset=it1,
    dataset_name="it1",
    range_type="static",   # używa peak_ranges_minmax
    peak_ranges_file = time_full,
    peak_amps_file = amps_full,
    range_name="minmax",
    methods=METHODS.keys()
)

# ------------------ SMOOTH, NO RANGES ------------------
results_smooth_no_ranges = wyniki(
    dataset=it1_sm_4Hz,
    dataset_name="it1_sm_4Hz",
    range_type=None,
    methods=METHODS.keys()
)

# ------------------ SMOOTH, RANGES MINMAX ------------------
results_smooth_ranges = wyniki(
    dataset=it1_sm_4Hz,
    dataset_name="it1_sm_4Hz",
    range_type="static",   # używa peak_ranges_minmax
    peak_ranges_file = time_full,
    peak_amps_file = amps_full,
    range_name="minmax",
    methods=METHODS.keys()
)

# # ---------- POROWNANKO ---------
# 1. Nadajemy unikalne nazwy kolumn dla każdego zestawu wyników
# Konsolidacja
df_norm = consolidate_results(results_normal_no_ranges)
df_norm_ranges = consolidate_results(results_normal_ranges)
df_smooth = consolidate_results(results_smooth_no_ranges)
df_smooth_ranges = consolidate_results(results_smooth_ranges)

# Nadanie unikalnych nazw kolumn
cols_needed = ["Class", "Peak", "Method", "Mean_XY_Error", "Min_XY_Error"]

df_norm = df_norm[cols_needed].rename(columns={
    "Mean_XY_Error": "XY_Error",
    "Min_XY_Error": "Min_XY_Error"
})

df_smooth = df_smooth[cols_needed].rename(columns={
    "Mean_XY_Error": "XY_Error_Smooth",
    "Min_XY_Error": "Min_XY_Error_Smooth"
})

df_norm_ranges = df_norm_ranges[cols_needed].rename(columns={
    "Mean_XY_Error": "XY_Error_Ranges",
    "Min_XY_Error": "Min_XY_Error_Ranges"
})

df_smooth_ranges = df_smooth_ranges[cols_needed].rename(columns={
    "Mean_XY_Error": "XY_Error_Smooth_Ranges",
    "Min_XY_Error": "Min_XY_Error_Smooth_Ranges"
})
# Łączenie DataFrame'ów i liczenie różnic
df_compare = pd.merge(df_norm, df_smooth, on=["Class", "Peak", "Method"], how="inner")
df_compare = pd.merge(df_compare, df_norm_ranges, on=["Class", "Peak", "Method"], how="inner")
df_compare = pd.merge(df_compare, df_smooth_ranges, on=["Class", "Peak", "Method"], how="inner")

df_compare["XY_Diff"] = df_compare["XY_Error"] - df_compare["XY_Error_Smooth"]
df_compare["Min_XY_Diff"] = df_compare["Min_XY_Error"] - df_compare["Min_XY_Error_Smooth"]
df_compare["XY_Diff_Range"] = df_compare["XY_Error"] - df_compare["XY_Error_Ranges"]
df_compare["Min_XY_Diff_Range"] = df_compare["Min_XY_Error"] - df_compare["Min_XY_Error_Ranges"]
df_compare["XY_Diff_Range_Smooth"] = df_compare["XY_Error_Smooth"] - df_compare["XY_Error_Smooth_Ranges"]
df_compare["Min_XY_Diff_Range_Smooth"] = df_compare["Min_XY_Error_Smooth"] - df_compare["Min_XY_Error_Smooth_Ranges"]

diff_columns = [col for col in df_compare.columns if "_Diff" in col]
sum_diff = {col: df_compare[col].sum() for col in diff_columns}

print(df_compare)
print(sum_diff)

# df_norm = df_avg_metrics[
#     ["Class", "Peak", "Method", "Mean_XY_Error", "Min_XY_Error"]
# ].rename(columns={
#     "Mean_XY_Error": "XY_Error",
#     "Min_XY_Error": "Min_XY_Error"
# })

# df_smooth = df_avg_metrics_smooth[
#     ["Class", "Peak", "Method", "Mean_XY_Error", "Min_XY_Error"]
# ].rename(columns={
#     "Mean_XY_Error": "XY_Error_Smooth",
#     "Min_XY_Error": "Min_XY_Error_Smooth"
# })

# df_norm_ranges = df_avg_metrics_ranges[
#     ["Class", "Peak", "Method", "Mean_XY_Error", "Min_XY_Error"]
# ].rename(columns={
#     "Mean_XY_Error": "XY_Error_Ranges",
#     "Min_XY_Error": "Min_XY_Error_Ranges"
# })
    
# df_smooth_ranges = df_avg_metrics_ranges_smooth[
#     ["Class", "Peak", "Method", "Mean_XY_Error", "Min_XY_Error"]
# ].rename(columns={
#     "Mean_XY_Error": "XY_Error_Smooth_Ranges",
#     "Min_XY_Error": "Min_XY_Error_Smooth_Ranges"
# })
    
# # df_compare = pd.merge(
# #     df_norm,
# #     df_smooth,
# #     df_norm_ranges,
# #     df_smooth_ranges,
# #     on=["Class", "Peak", "Method"],
# #     how="inner"
# # )

# df_compare = pd.merge(df_norm, df_smooth, on=["Class", "Peak", "Method"], how="inner")
# df_compare = pd.merge(df_compare, df_norm_ranges, on=["Class", "Peak", "Method"], how="inner")
# df_compare = pd.merge(df_compare, df_smooth_ranges, on=["Class", "Peak", "Method"], how="inner")


# df_compare["XY_Diff"] = (
#     df_compare["XY_Error"]
#     - df_compare["XY_Error_Smooth"]
# )

# df_compare["Min_XY_Diff"] = (
#     df_compare["Min_XY_Error"]
#     - df_compare["Min_XY_Error_Smooth"]
# )

# df_compare["XY_Diff_Range"] = (
#     df_compare["XY_Error"]
#     - df_compare["XY_Error_Ranges"]
# )

# df_compare["Min_XY_Diff_Range"] = (
#     df_compare["Min_XY_Error"]
#     - df_compare["Min_XY_Error_Ranges"]
# )
# print(df_compare)

# wybieramy wszystkie kolumny z "_Diff" w nazwie
# diff_columns = [col for col in df_compare.columns if "_Diff" in col]

# # tworzymy słownik sum dla każdej kolumny
# sum_diff = {col: df_compare[col].sum() for col in diff_columns}

# print(sum_diff)

"""
"""


# ---------------- stale zakresy ---------------------
all_metrics = []

for method_name in METHODS.keys():
    for cls in ["Class1", "Class2", "Class3", "Class4"]:
        for peak_name in ["P1","P2","P3"]:
            detected_all = []

            for item in analyzed_data:
                if item["class"] != cls:
                    continue
                file_name = item["file"]

                detected = single_peak_detection(
                    peak_name=peak_name,
                    class_name=cls,
                    file_name=file_name,
                    dataset=[item],
                    method_name=method_name,
                    peak_ranges_file=PEAK_RANGES[cls],
                    peak_amps_file=PEAK_AMPS.get(cls, None)
                )
                detected_all.extend(detected)

            # metryki dla wszystkich sygnałów tej klasy i piku
            df_metrics = compute_peak_metrics(
                dataset=[item for item in analyzed_data if item["class"] == cls],
                detected_peaks=detected_all,
                peak_name=peak_name,
                class_name=cls
            )
            df_metrics["Method"] = method_name
            all_metrics.append(df_metrics)

# scal wszystkie metryki
df_all_metrics = pd.concat(all_metrics, ignore_index=True)
df_avg_metrics = df_all_metrics.groupby(["Class", "Peak", "Method"]).mean(numeric_only=True).reset_index()
print(df_avg_metrics)

# ------------------- crossingi ---------------------
all_metrics_crossings = []

for method_name in METHODS.keys():
    for cls in ["Class1", "Class2", "Class3"]:
        for peak_name in ["P1","P2","P3"]:
            detected_all = []

            for item in analyzed_data:
                if item["class"] != cls:
                    continue
                file_name = item["file"]

                peak_ranges_file_for_item = crossing_ranges.get(file_name, None)
                if peak_ranges_file_for_item is None or peak_name not in peak_ranges_file_for_item:
                    continue
                
                detected = single_peak_detection(
                    peak_name=peak_name,
                    class_name=cls,
                    file_name=file_name,
                    dataset=[item],
                    method_name=method_name,
                    peak_ranges_file=peak_ranges_file_for_item,
                    peak_amps_file=None
                )
                detected_all.extend(detected)

            if len(detected_all) == 0:
                continue

            df_metrics = compute_peak_metrics(
                dataset=[item for item in analyzed_data if item["class"] == cls],
                detected_peaks=detected_all,
                peak_name=peak_name,
                class_name=cls
            )
            df_metrics["Method"] = method_name
            all_metrics_crossings.append(df_metrics)

df_all_metrics_crossings = pd.concat(all_metrics_crossings, ignore_index=True)
df_avg_metrics_crossings = df_all_metrics_crossings.groupby(["Class", "Peak", "Method"]).mean(numeric_only=True).reset_index()
print(df_avg_metrics_crossings)

# %% ----- PLOT - średni sygnał i histogramy -------- 
# zakresy stałe 
for method_name in METHODS.keys(): 
    plot_class_peaks_grid(analyzed_data, df_all_metrics, method_name, mode="static")  # crossingi 

for method_name in METHODS.keys(): 
    plot_class_peaks_grid(analyzed_data, df_all_metrics_crossings, method_name, mode="crossing") 

# %% ------- zapis wyników do CSV ------ 

# --- zapis wyników zakresów stałych --- 
df_avg_metrics.to_csv("avg_metrics_static_neue.csv", index=False)
# df_all_metrics.to_csv("all_metrics_static.csv", index=False)
print("Zapisano: avg_metrics_static.csv oraz all_metrics_static.csv")

# --- zapis wyników crossingów --- 
df_avg_metrics_crossings.to_csv("avg_metrics_crossings_neue.csv", index=False)
# df_all_metrics_crossings.to_csv("all_metrics_crossings.csv", index=False)
print("Zapisano: avg_metrics_crossings.csv oraz all_metrics_crossings.csv")
"""

# %% ------- ILUSTRACJA - METODY ------
"""
file_name = "Class1_example_11" 
class_signals = [item for item in data_it1 if item["class"] == "Class1"]
example_signal_item = class_signals[119]  # np. trzeci sygnał w liście
signal_df = example_signal_item["signal"]
t = signal_df.iloc[:, 0].values
y = signal_df.iloc[:, 1].values

# Wykrywanie pików różnymi metodami
peaks_concave = METHODS["concave"](y)
peaks_scholkmann = METHODS["modified_scholkmann_1_99"](y)
peaks_curvature = METHODS["curvature"](y)
peaks_line_vertical = METHODS["line_distance_10"](y)
peaks_hilbert = METHODS["hilbert"](y)
peaks_wavelet = METHODS["wavelet"](y)

# --- przygotowanie dodatkowych danych dla wizualizacji ---
# fragmenty wklęsłe dla concave / line distance
dx = np.gradient(y, edge_order=2)
d2x = np.gradient(dx, edge_order=2)
concave_mask = d2x < 0  # dla wizualizacji
t_concave = t[concave_mask]
y_concave = y[concave_mask]

# obwiednia Hilberta
analytic_signal = hilbert(y)
envelope = np.abs(analytic_signal)

# krzywizna
denom = (1 + dx**2)**1.5
denom[denom == 0] = 1e-8
curvature_vals = np.abs(d2x) / denom


# line distance – linie między końcami wklęsłych fragmentów
mask_diff = np.diff(concave_mask.astype(int))
region_starts = np.where(mask_diff == 1)[0]
region_ends = np.where(mask_diff == -1)[0]
if concave_mask[0]:
    region_starts = np.insert(region_starts, 0, 0)
if concave_mask[-1]:
    region_ends = np.append(region_ends, len(y)-1)
line_segments = [(start, end, y[start], y[end]) for start, end in zip(region_starts, region_ends) if end-start>=3]

# --- rysowanie 6 subplotów ---
methods = ["concave", "modified_scholkmann_1_99", "curvature", "line_distance_10", "hilbert", "wavelet"]
colors = ["red", "green", "blue", "orange", "purple", "brown"]
titles = ["Maksima w odcinkach wklęsłych", 
          "Zmodyfikowana metoda Scholkmanna", 
          "Maksymalna krzywizna", 
          "Max, odległosć od linii łaczącej końce odc. wklęsłego", 
          "Transformata Hilberta", 
          "Ciągła transformata falkowa"]

plt.figure(figsize=(12,12))

for i, method in enumerate(methods):
    plt.subplot(3,2,i+1)
    plt.plot(t, y, color='black', lw=1, label='Sygnał')

    # dodanie fragmentów wklęsłych dla concave i line distance
    if method in ["concave", "line_distance_10", "curvature"]:
        plt.fill_between(t, y, y.min(), where=concave_mask, color='lightblue', alpha=0.3, label='Fragmenty wklęsłe')

    # naniesienie pików
    peaks = METHODS[method](y)
    plt.scatter(t[peaks], y[peaks], color=colors[i], s=50, zorder=5, label='Piki')

    # dodatkowe elementy
    if method=="hilbert":
        plt.plot(t, envelope, color='purple', linestyle='--', lw=1.5, label='Obwiednia Hilberta')
    if method=="curvature":
        # ZASTĄPIĆ obecną linię krzywizny tym fragmentem:
        curvature_norm = curvature_vals / np.max(curvature_vals)
        plt.plot(t, curvature_norm, color='darkorange', lw=2, label='Krzywizna (norm.)')
        plt.fill_between(t, 0, curvature_norm, color='orange', alpha=0.1)
    if method=="line_distance_10":
        for start, end, y0, y1 in line_segments:
            plt.plot([t[start], t[end]], [y0, y1], color='orange', linestyle='--', lw=1, label="_nolegend_")
    if method == "wavelet":
        colors = ["#ff6666", "#ff9933", "#ffcc00", "#99cc00", "#3399ff"]
        scales=[1,3,5,7,9,10]
        for i, scale in enumerate(scales):
        # przesuń piki lekko w górę w zależności od skali, dla wizualizacji
            y_peaks = y[peaks_wavelet] + 0.05*i
            plt.scatter(t[peaks_wavelet], y_peaks, color=colors[i % len(colors)], s=25, label=f'Maksima dla skali s={scale}')  
    if method == "modified_scholkmann_1_99":

        max_scale = len(y) // 4     # jak w Twojej funkcji
        scales_to_show = [1, 3, 5, 7, 10]   # wizualnie ładne skale
    
        colors_scales = ["#ff6666", "#ff9933", "#ffcc00", "#99cc00", "#3399ff"]
    
        for s, col in zip(scales_to_show, colors_scales):
            lm_points = []
    
            # Szukamy lokalnych maksimów dla skali s
            for t0 in range(s, len(y) - s):
                if y[t0] > y[t0 - s] and y[t0] > y[t0 + s]:
                    lm_points.append(t0)
    
            # Rysujemy punkty lokalnych maksimów
            plt.scatter(t[lm_points], y[lm_points] + 0.025*s,   # lekkie przesunięcie
                        color=col, s=25,
                        label=f'Maksima dla skali s={s}')
    
        # Główny wynik metody
        # peaks = peaks_scholkmann
        # plt.scatter(t[peaks], y[peaks], color='green', s=80,
        #             label='Pik stabilny w wielu skalach', zorder=10)


    plt.title(titles[i], fontsize=16)
    plt.xlabel("Numer próbki", fontsize=12)
    plt.ylabel("Amplituda", fontsize=12)
    plt.grid(True)
    plt.legend(loc='upper right', fontsize=8)

plt.tight_layout(pad=3.0)
plt.show()
"""
# %% ---------- ILUSTRACJA - przykłady sygnalow
'''
# Rysujemy przykłady nr 30–35 w klasach IT1 i IT2
plot_examples(data_it1, "IT1", example_indices=range(80,84))
plot_examples(data_smooth_it1, "IT2", example_indices=range(80,84))

'''
# fig, ax = plot_single_signal(data_it1, "Class2", 9)
# plt.show()

