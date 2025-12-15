import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.signal import find_peaks

# TODO osobna funkcja licząca pochodne 1. i 2. stopnia
# TODO draft pierwsze podejscie do algorytmów
# TODO git repozytorium
# blad w osi x, osi y oraz xy
# klasa 1 - p1 najwyzszy, klasa 2 - p2 najwyzszy, p1>p3, klasa 3 - p2 najwyzszy, p1<p3 
# czujniki srodmiaszowe
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


# def plot_signal_with_peaks(signal_df, peaks_df, file_list, class_id, example_index):
#     """
#     Rysuje przykładowy sygnał z nałożonymi pikami oraz drugą pochodną.
#     """
#     # wybór sygnału
#     file_name = os.path.basename(file_list[example_index]).replace(".csv", "")
#     signal = signal_df[example_index]
    
#     # pobranie pików
#     peaks_row = peaks_df[peaks_df["File"] == file_name]
#     peaks = peaks_row.iloc[0, 1:].dropna().astype(int).values if not peaks_row.empty else []

#     # dane sygnału
#     t = signal.iloc[:, 0].values
#     x = signal.iloc[:, 1].values

#     # druga pochodna
#     dx = np.gradient(x, t, edge_order=2)
#     d2x = np.gradient(dx, t, edge_order=2)

#     # wykres
#     plt.figure(figsize=(10, 6))
    
#     plt.subplot(2, 1, 1)
#     plt.plot(t, x, label="Sygnał ICP")
#     for p in peaks:
#         plt.scatter(t[p], x[p], color="red", s=40, zorder=5)
#     plt.title(f"Class {class_id} — {file_name}")
#     plt.ylabel("Amplituda")
#     plt.legend()
#     plt.grid(True)

#     plt.subplot(2, 1, 2)
#     plt.plot(t, d2x, color='orange', label="Druga pochodna")
#     plt.xlabel("Czas [s]")
#     plt.ylabel("x''(t)")
#     plt.legend()
#     plt.grid(True)

#     plt.tight_layout()
#     plt.show()

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
        print(df_class.groupby("Method")[metrics].mean().round(3))

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

def analyze_results_grouped(results_df):
    metrics = ["Mean_Error", "Hit_Rate", "Mean_Y_Error", "Mean_XY_Error"]
    classes = sorted(results_df["Class"].unique())

    for class_name in classes:
        df_class = results_df[results_df["Class"] == class_name]
        print(f"\n Klasa: {class_name} — liczba sygnałów: {df_class['File'].nunique()}")
        print(df_class.groupby("Method")[metrics].mean().round(3))

        fig, axes = plt.subplots(1, 4, figsize=(15, 4))
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
        
def detect_concave_maxima(t, x, d2x_threshold=0, prominence=0):
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
    peaks, _ = find_peaks(x, prominence=prominence)

    # wybierz tylko te, które leżą w wklęsłych fragmentach
    concave_peaks = [p for p in peaks if concave_mask[p]]

    return concave_peaks

def modified_scholkmann(signal, max_scale=None):
    """
    Modified-Scholkmann peak detection.
    Zwraca indeksy stabilnych maksimów w wielu skalach.
    """
    x = signal - np.mean(signal)  # detrending (usunięcie trendu)
    N = len(x)
    
    if max_scale is None:
        max_scale = N // 4  # ok. 1/4 długości sygnału, jak w tekście

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
    threshold = np.percentile(maxima_strength, 90)  # tylko te najstabilniejsze
    peaks = np.where(maxima_strength >= threshold)[0]

    return peaks, maxima_strength

def detect_peaks_curvature(t, x, d2x_threshold=0, prominence=0):
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
    peaks, _ = find_peaks(curvature, prominence=prominence)

    # wybierz tylko piki w obszarach wklęsłych
    concave_peaks = [p for p in peaks if concave_mask[p]]

    return np.array(concave_peaks)

def detect_peaks_line_distance(t, x, d2x_threshold=0, mode="perpendicular"):
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
        if end - start < 3:
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

# przykład: Class 1, 200-ty sygnał
plot_signal_with_peaks(
    signal_df=data["Class1_signals"],
    peaks_df=data["Class1_peaks"],
    file_list=data["Class1_files"],
    class_id=1,
    example_index=200
)

    
methods = {
    #"concave": lambda t, x: detect_concave_maxima(t, x, d2x_threshold=0.002, prominence=0.02),
    "concave2": lambda t, x: detect_concave_maxima(t, x, d2x_threshold=0, prominence=0),
    #"concave3": lambda t, x: detect_concave_maxima(t, x, d2x_threshold=0, prominence=0.02),
    "modified_scholkmann": lambda t, x: modified_scholkmann(x)[0],
    "curvature": lambda t, x: detect_peaks_curvature(t, x, d2x_threshold=0),
    "line_perpendicular": lambda t, x: detect_peaks_line_distance(t, x, d2x_threshold=0, mode="perpendicular"),
    "line_vertical": lambda t, x: detect_peaks_line_distance(t, x, d2x_threshold=0, mode="vertical"),

}

results = run_peak_detection(base_path, detection_methods=methods)


analyze_results_grouped(results) # wykresy

example_index = 70  # który sygnał z każdej klasy chcesz pokazać (np. 0, 1, 2, 3)

# ---- RYSOWANIE ----
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for class_id in range(1, 5):
    class_name = f"Class{class_id}"

    # Ścieżki do plików
    folder_path = os.path.join(base_path, class_name)
    peaks_path = os.path.join(base_path, f"{class_name}_peaks.csv")

    csv_files = sorted(glob.glob(os.path.join(folder_path, f"{class_name}_example_*.csv")))
    peaks_df = pd.read_csv(peaks_path)

    if len(csv_files) == 0 or peaks_df.empty:
        print(f"⚠️ Brak danych dla {class_name}")
        continue

    # wybór sygnału
    idx = min(example_index, len(csv_files) - 1)
    file_path = csv_files[idx]
    file_name = os.path.basename(file_path).replace(".csv", "")
    df = pd.read_csv(file_path)

    t = df.iloc[:, 0].values
    x = df.iloc[:, 1].values

    # pobierz ręczne piki (po nazwie pliku lub po indeksie)
    peaks_row = peaks_df[peaks_df["File"] == file_name]
    if peaks_row.empty:
        peaks = peaks_df.iloc[idx, 1:].dropna().astype(int).values
    else:
        peaks = peaks_row.iloc[0, 1:].dropna().astype(int).values

    # --- Rysowanie ---
    ax = axes[class_id - 1]
    ax.plot(t, x, color="black", lw=1.2, label="Sygnał ICP")
    if len(peaks) > 0:
        ax.scatter(t[peaks], x[peaks], color="red", s=40, zorder=5, label="Piki ręczne")

    ax.set_title(f"{class_name} — przykład {idx+1}")
    ax.set_xlabel("Czas [s]")
    ax.set_ylabel("Amplituda")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

plt.tight_layout()
plt.show()


# kolory tła (zielony → czerwony)
bg_colors = ["#a8e6a1", "#d7ef9f", "#fff59d", "#f28b82"]

# litery podpisów (a), (b), (c), (d)
labels = ["(a)", "(b)", "(c)", "(d)"]

fig, axes = plt.subplots(1, 4, figsize=(16, 3))  # jedna linia, 4 kolumny
axes = axes.flatten()

example_index = 70  # który przykład z każdej klasy
base_path = r"C:\Users\User\OneDrive\Dokumenty\praca inżynierska\ICP_pulses_it1"

for class_id in range(1, 5):
    class_name = f"Class{class_id}"

    # Ścieżki
    folder_path = os.path.join(base_path, class_name)
    peaks_path = os.path.join(base_path, f"{class_name}_peaks.csv")

    csv_files = sorted(glob.glob(os.path.join(folder_path, f"{class_name}_example_*.csv")))
    peaks_df = pd.read_csv(peaks_path)

    if len(csv_files) == 0 or peaks_df.empty:
        print(f"⚠️ Brak danych dla {class_name}")
        continue

    idx = min(example_index, len(csv_files) - 1)
    file_path = csv_files[idx]
    file_name = os.path.basename(file_path).replace(".csv", "")
    df = pd.read_csv(file_path)

    t = df.iloc[:, 0].values
    x = df.iloc[:, 1].values

    # Ręczne piki (jeśli są)
    peaks_row = peaks_df[peaks_df["File"] == file_name]
    if peaks_row.empty:
        peaks = peaks_df.iloc[idx, 1:].dropna().astype(int).values
    else:
        peaks = peaks_row.iloc[0, 1:].dropna().astype(int).values

    # --- Rysowanie ---
    ax = axes[class_id - 1]
    ax.set_facecolor(bg_colors[class_id - 1])
    ax.plot(t, x, color="black", lw=1.3)

    # Piki tylko dla klas 1–3
    if class_id < 4 and len(peaks) > 0:
        for j, p in enumerate(peaks):
            ax.scatter(t[p], x[p], color="red", s=30, zorder=5)
            ax.text(t[p], x[p] + 0.02 * (x.max() - x.min()), f"P{j+1}",
                    color="black", fontsize=14, ha="center", va="bottom", clip_on=True)

    y_margin = 0.15 * (x.max() - x.min())  # 10% dodatkowego miejsca u góry
    ax.set_ylim(x.min() - 0.5*y_margin, x.max() + y_margin)

            
    ax.set_xticks([])
    ax.set_yticks([])
    
    #if class_id == 1:
    
    #axes[0].set_ylabel("ciśnienie", fontsize=16, labelpad=5)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.text(-0.01, 0.5, "ciśnienie", va='center', ha='left', rotation='vertical', fontsize=16)
    fig.text(0.5, -0.03, "czas", ha='center', fontsize=16)

    
    # Brak siatki, brak legendy, brak tytułu
    ax.grid(False)
    ax.set_title("")

    # Podpis (a), (b), (c), (d)
    ax.text(0.03, 0.95, labels[class_id - 1],
    transform=ax.transAxes, fontsize=14, fontweight="bold", va="top", ha="left")

# Wyrównanie osi, odstępy
plt.tight_layout(w_pad=0.5)
plt.savefig("p1p2p3.pdf", bbox_inches='tight', pad_inches=0.15)
plt.show()
