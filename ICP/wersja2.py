import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.signal import find_peaks


# zapis do wynikow do csv, wyliczenie "cech" poszczegolnych klas: AmpICP, latencji i amplitud p1, p2, p3, pola pod krzywa; okreslenie ile pikow wykrywa dana metoda

# TODO osobna funkcja licząca pochodne 1. i 2. stopnia
# TODO draft pierwsze podejscie do algorytmów
# klasa 1 - p1 najwyzszy, klasa 2 - p2 najwyzszy, p1>p3, klasa 3 - p2 najwyzszy, p1<p3 
# istniejace ilustracje ok, mozna tlumaczyc na polski (grafika zaadaptowane)

# składowe, sposób pomiaru, state of the art, 

def load_data(base_path, n_classes=4):
    """
    Wczytuje pliki CSV i pliki z pikami dla wszystkich klas.
    Zwraca słownik z DataFrame’ami i listami sygnałów.
    """
    data = {}
    for i in range(1, n_classes + 1):
        # plik z pikami
        peaks_path = os.path.join(base_path, f"Class{i}_peaks.csv")
        data[f"Class{i}_peaks"] = pd.read_csv(peaks_path)

        # folder z sygnałami
        folder = os.path.join(base_path, f"Class{i}")
        csv_files = sorted(glob.glob(os.path.join(folder, f"Class{i}_example_*.csv")))
        data[f"Class{i}_files"] = csv_files
        data[f"Class{i}_signals"] = [pd.read_csv(f) for f in csv_files]

    return data

def plot_signal_with_peaks(signal_df, peaks_df, file_list, class_id, example_index, detection_func=None, show_curvature=False):
    """
    Rysuje przykładowy sygnał z nałożonymi pikami (ręcznymi i automatycznymi).
    Opcjonalnie pokazuje też krzywiznę lub drugą pochodną.
    
    Parameters
    ----------
    signal_df : list[pd.DataFrame]
        Lista sygnałów (każdy z kolumnami: czas, amplituda)
    peaks_df : pd.DataFrame
        Dane z rzeczywistymi pikami (kolumna File + P1, P2, P3)
    file_list : list[str]
        Lista ścieżek do plików sygnałów
    class_id : int
        Numer klasy (1–4)
    example_index : int
        Indeks przykładowego sygnału do wizualizacji
    detection_func : callable
        Funkcja detekcji pików (np. detect_peaks_curvature)
    show_curvature : bool
        Czy pokazać krzywiznę zamiast drugiej pochodnej
    """

    # wybór sygnału
    file_name = os.path.basename(file_list[example_index]).replace(".csv", "")
    signal = signal_df[example_index]

    # dane sygnału
    t = signal.iloc[:, 0].values
    x = signal.iloc[:, 1].values

    # pobranie rzeczywistych pików
    peaks_row = peaks_df[peaks_df["File"] == file_name]

    peaks_true = peaks_row.iloc[0, 1:].dropna().astype(int).values if not peaks_row.empty else []
    
    
    # obliczenia pochodnych
    dx = np.gradient(x, t, edge_order=2)
    d2x = np.gradient(dx, t, edge_order=2)
    
    # obliczenie krzywizny
    denom = (1 + dx**2)**1.5
    denom[denom == 0] = 1e-8
    curvature = np.abs(d2x) / denom

    # wykrycie pików automatycznych (jeśli podano funkcję)
    if detection_func is not None:
        try:
            peaks_auto = detection_func(t, x)
        except Exception as e:
            print(f"Błąd w detekcji dla {file_name}: {e}")
            peaks_auto = []
    else:
        peaks_auto = []

    # --- wykresy ---
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(t, x, label="Sygnał ICP", color="black")

    # rzeczywiste piki
    if len(peaks_true) > 0:
        ax1.scatter(t[peaks_true], x[peaks_true], color="green", s=60, label="Piki ręczne")

    # automatyczne piki
    if len(peaks_auto) > 0:
        ax1.scatter(t[peaks_auto], x[peaks_auto], color="red", s=40, marker="x", label="Piki automatyczne")

    ax1.set_title(f"Class {class_id} — {file_name}")
    ax1.set_xlabel("Czas [s]")
    ax1.set_ylabel("Amplituda")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    # dodatkowa oś dla drugiej pochodnej lub krzywizny
    ax2 = ax1.twinx()
    if show_curvature:
        ax2.plot(t, curvature, color="blue", alpha=0.3, label="Krzywizna")
        ax2.set_ylabel("Krzywizna")
    else:
        ax2.plot(t, d2x, color="orange", alpha=0.4, label="Druga pochodna")
        ax2.set_ylabel("x''(t)")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

def run_peak_detection(base_path, detection_methods, tolerance=10):
    """
    Iteruje po wszystkich sygnałach z czterech klas, stosuje metody wykrywania pików
    i zwraca DataFrame z wynikami porównania.
    
    Parameters
    ----------
    base_path : str
        Folder bazowy z danymi.
    detection_methods : dict
        np. {"concave": detect_concave_maxima, "scipy_peaks": detect_peaks_scipy}
    tolerance : int
        Maksymalna różnica indeksowa, uznawana za trafienie.

    Returns
    -------
    results_df : pd.DataFrame
    """
    all_results = []

    for class_id in range(1, 5):  # Class1..Class4
        class_name = f"Class{class_id}"
        print(f" Przetwarzanie {class_name}...")

        # pliki sygnałów
        folder_path = os.path.join(base_path, class_name)
        csv_files = sorted(glob.glob(os.path.join(folder_path, f"{class_name}_example_*.csv")))

        # plik z pikami
        peaks_path = os.path.join(base_path, f"{class_name}_peaks.csv")
        peaks_df = pd.read_csv(peaks_path)

        for file_path in csv_files:
            file_name = os.path.basename(file_path).replace(".csv", "")
            df = pd.read_csv(file_path)
            t = df.iloc[:, 0].values
            x = df.iloc[:, 1].values

            # pobierz ręczne piki
            row = peaks_df[peaks_df["File"] == file_name]
            if row.empty:
                continue
            
            p_values = row.iloc[0, 1:].astype(float).values  # [P1, P2, P3]

            # --- interpretacja ---
            if p_values[0] == -1 and p_values[2] == -1:
                # przypadek: -1, liczba, -1 → wszystkie piki scalone w jeden
                true_peaks = [p_values[1]]
                peak_status = "merged_all"
            else:
                # przypadek: klasyczne trzy piki
                true_peaks = [p for p in p_values if p != -1]
                peak_status = "normal"
            # szukany jest najbliższy ręcznemu wykryty pik, tzn, jesli metoda wykryła
            # 10 pikow, to i tak do liczenia bledu beda brane 3
            # zastosuj wszystkie metody detekcji
            for method_name, method_func in detection_methods.items():
                try:
                    detected_peaks = method_func(t, x)
                except Exception as e:
                    print(f" Błąd w metodzie {method_name} dla {file_name}: {e}")
                    detected_peaks = []

                if len(detected_peaks) == 0 or len(true_peaks) == 0:
                    mean_error = np.nan
                    hit_rate = 0.0
                    mean_y_error = np.nan
                    mean_xy_error = np.nan
                else:
                    # dopasowanie do ręcznych pików
                    distances = [min(abs(d - m) for d in detected_peaks) for m in true_peaks]
                    hits = [d < tolerance for d in distances]
                    # wesja sprzed dodania błędów y, xy
                    # mean_error = np.mean([abs(detected_peaks[min(range(len(detected_peaks)), 
                    #                     key=lambda i: abs(detected_peaks[i] - m))] - m)
                    #                       for m in true_peaks])
                    # hit_rate = np.mean(hits)
                    
                    # indeks najbliższego wykrytego piku dla każdego ręcznego  # indeks najbliższego wykrytego piku dla każdego ręcznego
                    matched_indices = [
                        min(range(len(detected_peaks)), key=lambda i: abs(detected_peaks[i] - m))
                        for m in true_peaks
                    ]

                    # błędy w osi x (indeks), y (amplituda) i euklidesowe
                    x_errors = [abs(detected_peaks[i] - m) for i, m in zip(matched_indices, true_peaks)]
                    y_errors = [abs(x[detected_peaks[i]] - x[int(m)]) for i, m in zip(matched_indices, true_peaks)]
                    xy_errors = [np.sqrt((t[detected_peaks[i]] - t[int(m)])**2 +
                                         (x[detected_peaks[i]] - x[int(m)])**2)
                                 for i, m in zip(matched_indices, true_peaks)]

                    mean_error = np.mean(x_errors)
                    hit_rate = np.mean(hits)
                    mean_y_error = np.mean(y_errors)
                    mean_xy_error = np.mean(xy_errors)

                all_results.append({
                    "Class": class_name,
                    "File": file_name,
                    "Method": method_name,
                    "True_Peaks": true_peaks,
                    "Detected_Peaks": detected_peaks,
                    "Peak_Status": peak_status,
                    "Mean_Error": mean_error,
                    "Hit_Rate": hit_rate,
                    "Mean_Y_Error": mean_y_error,
                    "Mean_XY_Error": mean_xy_error,
                    "Peak_Count": len(detected_peaks)
                    # "Mean_Nearest_Distance": mean_nearest_distance
                })

    results_df = pd.DataFrame(all_results)
    print(" Zakończono analizę wszystkich klas.")
    return results_df

def analyze_results(results_df):
    metrics = ["Mean_Error", "Hit_Rate", "Mean_Y_Error", "Mean_XY_Error", "Peak_Count"]
    classes = sorted(results_df["Class"].unique())

    for class_name in classes:
        df_class = results_df[results_df["Class"] == class_name]
        print(f"\n Klasa: {class_name} — liczba sygnałów: {df_class['File'].nunique()}")
        print(df_class.groupby("Method")[metrics].mean().round(4))

        fig, axes = plt.subplots(1, 4, figsize=(15, 4))
        fig.suptitle(f"Wskaźniki jakości detekcji — {class_name}", fontsize=14, fontweight="bold")

        methods = df_class["Method"].unique()

        for i, metric in enumerate(metrics):
            all_data = df_class[metric].dropna()
            bins = np.linspace(all_data.min(), all_data.max(), 20)
            
            for method in methods:
                data = df_class[df_class["Method"] == method][metric].dropna()
                axes[i].hist(data, bins=bins, alpha=0.6, rwidth=0.7, label=method)
                
            axes[i].set_title(metric.replace("_", " "))
            axes[i].set_xlabel(metric)
            axes[i].set_ylabel("Liczba sygnałów")
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()

        plt.tight_layout()
        plt.show()

def save_with_increment(df, base_name="methods_results", ext=".csv"):
    """
    Zapisuje DataFrame jako base_name.csv,
    a jeśli plik istnieje, to base_name_2.csv, base_name_3.csv, itd.
    """
    filename = base_name + ext
    counter = 2

    # jeśli istnieje, szukaj kolejnych nazw
    while os.path.exists(filename):
        filename = f"{base_name}_{counter}{ext}"
        counter += 1

    df.to_csv(filename, index=False)
    print(f"Zapisano: {filename}")
    
    
def analyze_results_grouped(results_df):
    metrics = ["Mean_Error", "Hit_Rate", "Mean_Y_Error", "Mean_XY_Error", "Peak_Count"]
    classes = sorted(results_df["Class"].unique())
    
    out_df = pd.DataFrame(columns=["Method"] + metrics + ["Class"])
    
    for class_name in classes:
        df_class = results_df[results_df["Class"] == class_name]
        print(f"\n Klasa: {class_name} — liczba sygnałów: {df_class['File'].nunique()}")
        dane_df = df_class.groupby("Method")[metrics].mean().round(3)
        dane_df["Class"] = class_name
        print(dane_df)
        # save_with_increment(dane_df, base_name="methods_results", ext=".csv")
        for idx, row in dane_df.iterrows():
            out_df.loc[len(out_df)] = [idx] + list(row[metrics]) + [row["Class"]]
            
        # wykresy
        fig, axes = plt.subplots(1, 5, figsize=(15, 4))
        fig.suptitle(f"Wskaźniki jakości detekcji — {class_name}", fontsize=14, fontweight="bold")

        methods = list(df_class["Method"].unique())
        n_methods = len(methods)

        for i, metric in enumerate(metrics):
            all_data = df_class[metric].dropna()
            bins = np.linspace(all_data.min(), all_data.max(), 15)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            width = (bins[1] - bins[0]) / (n_methods + 1)  # szerokość pojedynczego słupka

            for j, method in enumerate(methods):
                data = df_class[df_class["Method"] == method][metric].dropna()
                counts, _ = np.histogram(data, bins=bins)

                # przesunięcie słupków dla każdej metody
                offset = (j - n_methods / 2) * width + width / 2
                axes[i].bar(bin_centers + offset, counts, width=width, alpha=0.7, label=method)

            axes[i].set_title(metric.replace("_", " "))
            axes[i].set_xlabel(metric)
            axes[i].set_ylabel("Liczba sygnałów")
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()

        plt.tight_layout()
        plt.show()
        
    save_with_increment(out_df, base_name="methods", ext=".csv")


        
def detect_concave_maxima(t, x, d2x_threshold=0, prominence=0, threshold=None):
    """
    Wykrywa lokalne maksima sygnału (peaki) w obszarach, gdzie druga pochodna < d2x_threshold.
    Pasuje do frameworka `run_peak_detection_experiments`.
    
    Parameters
    ----------
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
    dx = np.gradient(x, t, edge_order=2)
    d2x = np.gradient(dx, t, edge_order=2)

    # maska obszarów wklęsłych
    concave_mask = d2x < d2x_threshold

    # znajdź lokalne maksima w całym sygnale
    peaks, _ = find_peaks(x, threshold=threshold, prominence=prominence)

    # wybierz tylko te, które leżą w wklęsłych fragmentach
    concave_peaks = [p for p in peaks if concave_mask[p]]

    return concave_peaks

def modified_scholkmann(signal, scale=1):
    """
    Modified-Scholkmann peak detection.
    Zwraca indeksy stabilnych maksimów w wielu skalach.
    """
    # x = signal - np.mean(signal)  # detrending (usunięcie trendu)
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

    return peaks, maxima_strength

def detect_peaks_curvature(t, x, d2x_threshold=0, prominence=0, threshold=None):
    """
    Detekcja pików na podstawie lokalnych maksimów krzywizny.
    """
    dx = np.gradient(x, t)
    d2x = np.gradient(dx, t)

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

def detect_peaks_line_distance(t, x, d2x_threshold=0, mode="perpendicular", min_len=3):
    """
    Detekcja pików na podstawie odległości od linii łączącej końce regionu wklęsłego.
    mode = "perpendicular"  → metoda 3 (prostopadła odległość)
    mode = "vertical"       → metoda 4 (pionowa odległość)
    """

    dx = np.gradient(x, t)
    d2x = np.gradient(dx, t)

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
        region_ends = np.append(region_ends, len(x) - 1)

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

def sweep_threshold_for_method(
        base_path,
        method_name,
        detection_func,
        threshold_values,
        tolerance=10):
    """
    Sweeping parametru 'threshold' dla jednej metody detekcji.
    Zwraca DataFrame z wynikami dla każdej wartości threshold.
    """
    all_rows = []

    for thr in threshold_values:
        print(f"Testowanie threshold = {thr:.6f}")

        # opakowanie funkcji z aktualnym threshold
        def wrapped_method(t, x):
            return detection_func(t, x, d2x_threshold=0, threshold=thr)

        # uruchom detekcję
        results = run_peak_detection(
            base_path,
            detection_methods={method_name: wrapped_method},
            tolerance=tolerance
        )

        # dodaj kolumnę ze sweep parametrem
        results["Threshold"] = thr

        all_rows.append(results)

    return pd.concat(all_rows, ignore_index=True)

def sweep_parameter_for_method(
        base_path,
        method_name,
        detection_func,
        sweep_param_name,
        sweep_values,
        fixed_params=None,
        tolerance=10):
    """
    Ogólny sweep parametru dla dowolnej metody detekcji pików.
    
    Parameters
    ----------
    base_path : str
        Folder z danymi.
    method_name : str
        Nazwa metody.
    detection_func : callable
        Funkcja wykrywania pików.
    sweep_param_name : str
        Nazwa parametru, który będziemy zmieniać (np. 'threshold', 'min_len').
    sweep_values : iterable
        Wartości, po których robimy sweep.
    fixed_params : dict, optional
        Pozostałe parametry funkcji, które pozostają stałe.
    tolerance : int
        Maksymalna odległość indeksowa uznawana za trafienie.
    """
    if fixed_params is None:
        fixed_params = {}

    all_rows = []

    for val in sweep_values:
        print(f"Testowanie {sweep_param_name} = {val}")
        
        # Funkcja opakowująca z aktualnym parametrem
        def wrapped_method(t, x):
            params = fixed_params.copy()
            params[sweep_param_name] = val
            return detection_func(t, x, **params)

        # Uruchom detekcję
        results = run_peak_detection(
            base_path,
            detection_methods={method_name: wrapped_method},
            tolerance=tolerance
        )

        results[sweep_param_name] = val
        all_rows.append(results)

    return pd.concat(all_rows, ignore_index=True)

def plot_error_and_peakcount_by_class(results_df, method_name, param_name="Threshold"):
    classes = sorted(results_df["Class"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, cls in zip(axes, classes):
        df = results_df[
            (results_df["Method"] == method_name) &
            (results_df["Class"] == cls)
        ]

        grouped = df.groupby(param_name).agg({
            "Mean_XY_Error": "mean",
            "Peak_Count": "mean"
        }).reset_index()

        # Oś lewa
        ax1 = ax
        ax1.plot(
            grouped[param_name], grouped["Mean_XY_Error"],
            marker="o", linewidth=2, color="blue", label="Mean XY Error"
        )
        ax1.set_xlabel(param_name)
        ax1.set_ylabel("Mean XY Error", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.grid(True, alpha=0.3)

        # Oś prawa
        ax2 = ax1.twinx()
        ax2.plot(
            grouped[param_name], grouped["Peak_Count"],
            marker="s", linewidth=2, color="red", label="Peak Count"
        )
        ax2.set_ylabel("Peak Count", color="red")
        ax2.tick_params(axis="y", labelcolor="red")

        # -------------------------
        # Dodanie prostokąta zakresu poprawnej liczby pików
        # -------------------------
        if cls in ["Class1", "Class2", "Class3"]:
            ymin, ymax = 2.5, 3.5
        else:  # Class4
            ymin, ymax = 0.5, 1.5

        # Prostokąt na osi Peak_Count (prawa oś Y)
        ax2.axhspan(
            ymin, ymax,
            color="green",
            alpha=0.15,
            zorder=0
        )

        ax1.set_title(f"{method_name} — {cls}")

    plt.tight_layout()
    plt.show()

def compute_statistics(data):
    """
    Oblicza statystyki dla klas 1-4:
    - liczba próbek
    - średnia i odchylenie amplitudy
    - czasy (latencje) P1, P2, P3 + odch. standardowe
    - pole pod krzywą (AUC) + odch. standardowe
    Zwraca DataFrame i zapisuje CSV.
    """
    stats_list = []

    for i in range(1, 5):
        class_name = f"Class{i}"
        signals = data[f"{class_name}_signals"]
        peaks_df = data[f"{class_name}_peaks"]

        n_samples = len(signals)

        # amplituda sygnałów
        max_amplitudes = [sig.iloc[:,1].max() for sig in signals]
        mean_amp = np.mean(max_amplitudes)
        std_amp = np.std(max_amplitudes)

        # czasy pików
        p_times = { 'P1': [], 'P2': [], 'P3': [] }
        p_amps = { 'P1': [], 'P2': [], 'P3': [] }
        
        for idx, row in peaks_df.iterrows():
            file_name = row['File']
            sig_idx = next((j for j,sig in enumerate(data[f"{class_name}_files"]) 
                            if file_name in sig), None)
            
            if sig_idx is not None:
                sig = signals[sig_idx]
                t = sig.iloc[:,0].values
                y = sig.iloc[:, 1].values
                
                for j, p_label in enumerate(['P1','P2','P3']):
                    p = row[p_label]
                    if pd.notna(p) and p != -1:
                        # indeks p do czasu
                        p_times[p_label].append(t[int(p)])
                        p_amps[p_label].append(y[int(p)]) 
        

        mean_times = {k: np.mean(v) if v else np.nan for k,v in p_times.items()}
        std_times  = {k: np.std(v) if v else np.nan for k,v in p_times.items()}
        
        # średnia i std amplitudy pików
        mean_amps = {k: np.mean(v) if len(v) > 0 else np.nan for k, v in p_amps.items()}
        std_amps  = {k: np.std(v) if len(v) > 0 else np.nan for k, v in p_amps.items()}


        # pole pod krzywą (AUC)
        aucs = [np.trapezoid(sig.iloc[:,1].values, sig.iloc[:,0].values) for sig in signals]
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        stats_list.append({
            'Class': class_name,
            'N_Samples': n_samples,
            'Mean_Amplitude': mean_amp,
            'Std_Amplitude': std_amp,
            
            'P1_mean_amp': mean_amps['P1'],
            'P1_std_amp':  std_amps['P1'],
            'P2_mean_amp': mean_amps['P2'],
            'P2_std_amp':  std_amps['P2'],
            'P3_mean_amp': mean_amps['P3'],
            'P3_std_amp':  std_amps['P3'],
            
            'P1_mean_time': mean_times['P1'],
            'P1_std_time': std_times['P1'],
            'P2_mean_time': mean_times['P2'],
            'P2_std_time': std_times['P2'],
            'P3_mean_time': mean_times['P3'],
            'P3_std_time': std_times['P3'],
            'Mean_AUC': mean_auc,
            'Std_AUC': std_auc
        })

    stats_df = pd.DataFrame(stats_list)
    print(stats_df)
    save_with_increment(stats_df, "class_statistics", ".csv")
    # stats_df.to_csv("class_statistics.csv", index=False)
    return stats_df


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

# def compare_with_annotated(auto_peaks, manual_peaks, tolerance=10):
#     """
#     Porównuje automatycznie znalezione piki z ręcznymi (P1, P2, P3).
#     tolerance – dopuszczalne odchylenie (np. 10 próbek)
#     """
#     matches = []
#     for mp in manual_peaks:
#         # znajdź automatyczny pik najbliżej ręcznego
#         diffs = np.abs(auto_peaks - mp)
#         if len(diffs) == 0:
#             matches.append((mp, None, None))
#         else:
#             closest_idx = np.argmin(diffs)
#             match = auto_peaks[closest_idx]
#             distance = diffs[closest_idx]
#             matches.append((mp, match, distance))
#     return matches

# --- użycie ---
base_path = r"C:\Users\User\OneDrive\Dokumenty\praca inżynierska\ICP_pulses_it1"
data = load_data(base_path)

pd.options.display.float_format = '{:.4f}'.format

    
methods = {
    "concave": lambda t, x: detect_concave_maxima(t, x, d2x_threshold=0, prominence=0, threshold=0),
    "modified_scholkmann1": lambda t, x: modified_scholkmann(x, 1)[0],
    "curvature": lambda t, x: detect_peaks_curvature(t, x, d2x_threshold=0, threshold=0),
    #"line_perpendicular": lambda t, x: detect_peaks_line_distance(t, x, d2x_threshold=0, mode="perpendicular", min_len=3),
    "line_vertical": lambda t, x: detect_peaks_line_distance(t, x, d2x_threshold=0, mode="vertical", min_len=3),
}

# results = run_peak_detection(base_path, detection_methods=methods)


# analyze_results_grouped(results) # wykresy

# for method in methods:
#     plot_all_signals_with_peaks(data, results, method)



# threshold_values = np.linspace(-0.5, 0.5, 10)

# results_sweep = sweep_threshold_for_method(
#     base_path,
#     method_name="concave",
#     detection_func=detect_concave_maxima,
#     threshold_values=threshold_values
# )

# plot_error_and_peakcount_by_class(results_sweep, "concave")

# threshold_values = range(5, 20, 2)  # min_len od 3 do 15 próbek

# results_sweep = sweep_parameter_for_method(
#     base_path,
#     method_name="concave",
#     detection_func=detect_concave_maxima,
#     sweep_param_name="d2x_threshold",
#     sweep_values=threshold_values,
#     #fixed_params={"d2x_threshold":0}
# )

# plot_error_and_peakcount_by_class(results_sweep, "concave", param_name="d2x_threshold")

# results_sweep_2 = sweep_parameter_for_method(
#     base_path,
#     method_name="curvature",
#     detection_func=detect_peaks_curvature,
#     sweep_param_name="d2x_threshold",
#     sweep_values=threshold_values,
#     #fixed_params={"d2x_threshold":0}
# )

# plot_error_and_peakcount_by_class(results_sweep, "curvature", param_name="d2x_threshold")


# stats_df = compute_statistics(data)


