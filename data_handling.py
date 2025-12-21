import os
import glob
import pandas as pd
from scipy.signal import butter, filtfilt

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 11:12:27 2025

@author: Hanna Jaworska

load_dataset: Wcztytanie danych (it1, it2) do programu
smooth_butter: filtr dolnoprzepustowy buttterwortha
smooth_dataset: wygladzenie wszystkich sygnalow w zbiorze danych
"""

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
    
        dataset = [
        {'class': 'Class1', 'file': 'Class1_example_0001', 'signal': DataFrame, 'peaks_ref': {'P1': 31, 'P2': 53, 'P3': 90}},
        {'class': 'Class1', 'file': 'Class1_example_0002', 'signal': DataFrame, 'peaks_ref': {'P1': 31, 'P2': 58, 'P3': 97}},
         ... i tak dalej
        ]
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


def smooth_butter(x, cutoff=1, fs=1, order=3):
    """
    Wygładza jednowymiarowy sygnał za pomocą cyfrowego filtru
    dolnoprzepustowego Butterwortha z zerowym przesunięciem fazowym.


    Parameters
    ----------
    x : nd,array
        Jednowymiarowy sygnał wejściowy (próbkowany w czasie)
    cutoff : float, optional
        Częstotliwość odcięcia filtru dolnoprzepustowego [Hz].
        Składowe sygnału o częstotliwościach wyższych niż `cutoff`
        są tłumione. Domyślnie 1 Hz.
    fs : float, optional
        Częstotliwość próbkowania sygnału [Hz].
        Domyślnie 180 Hz.
    order : int, optional
        Rząd filtru Butterwortha.
        Wyższy rząd oznacza ostrzejsze przejście pomiędzy pasmem
        przepustowym i zaporowym, kosztem większej podatności
        na oscylacje brzegowe. Domyślnie 3.

    Returns
    -------
    np.ndarray
        Wygładzony sygnał o tej samej długości co sygnał wejściowy.
    
    Notes
    -----
    Funkcja wykorzystuje dwie operacje:

    1. butter
       cyfrowy filtr Butterwortha.
       - filtr jest dolnoprzepustowy (``btype='low'``),
       - częstotliwość odcięcia jest normalizowana względem
         częstotliwości Nyquista (``fs / 2``),
       - współczynniki filtru są zwracane w postaci
         licznika ``b`` i mianownika ``a``.

    2. filtfilt
       filtracja dwukierunkowa (forward–backward).
       Skutki:
       - całkowite zniesienie przesunięcia fazowego (zero-phase),
       - efektywny rząd filtru jest podwojony,
       - zachowana jest dokładna lokalizacja czasowa pików
         i punktów charakterystycznych sygnału.
    """
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
if __name__ == "__main__":
    data = load_dataset(r"ICP_pulses_it1", "it1")
    
    
    # get ref peaks
    refs_df = []
    for i in range (0, 10):
        pulsacja = data[i]
        piki_referencyjne = pulsacja['peaks_ref']
        sygnal_ICP = pulsacja['file']
        row = {'file_name': sygnal_ICP, 'referencja': piki_referencyjne}
        
        refs_df.append(row)
    refs_df = pd.DataFrame(refs_df)
    print(refs_df)