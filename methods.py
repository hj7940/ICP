# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 12:03:29 2025

Metody wykrywania pikow

Zawiera implementacje algorytmów detekcji pików opartych na:
- wkleslosci+find_peaks,
- stabilności maksimów w wielu skalach,
- wkleslosci+krzywiźnie,
- wkleslosci+odległości od linii bazowej,
- obwiedni Hilberta,
- transformacie falkowej CWT.


@author: Hanna Jaworska
"""
import numpy as np
from scipy.signal import find_peaks, hilbert, find_peaks_cwt
from itertools import groupby


def concave(signal, d2x_threshold=0, min_len=8, height=0, prominence=0):
    """
    Detekcja pików na podstawie lokalnych maksimów w obszarach wklęsłych sygnału.

    Wyszukuje lokalne maksima sygnału, a następnie filtruje je,
    pozostawiając tylko te, które leżą w obszarach o ujemnej
    drugiej pochodnej (wklęsłość).
    
    Parameters
    ----------
    signal : np.ndarray
        Jednowymiarowy sygnał wejściowy
    d2x_threshold : float, optional
        Próg dla drugiej pochodnej (obszary wklęsłe spełniają d2x < d2x_threshold).
        Domyślnie 0.
    prominence : float, optional
        Minimalna wyrazistość piku (parametr dla `find_peaks`).
    threshold : float or None, optional
    Minimalna wysokość piku względem sąsiadów
    (parametr ``threshold`` funkcji ``find_peaks``).
    Domyślnie None.
    
    Returns
    -------
    np.ndarray
        Indeksy wykrytych pików.
    """
    # pochodne
    dx = np.gradient(signal, edge_order=2)
    d2x = np.gradient(dx, edge_order=2)

    # maska obszarów wklęsłych
    concave_mask = d2x < d2x_threshold
    
    # --- regiony wklęsłe ---
    regions = []
    for k, g in groupby(enumerate(concave_mask), key=lambda x: x[1]):
        if k:
            idx = [i for i, _ in g]
            if len(idx) >= min_len:
                regions.append((idx[0], idx[-1]))

    # piki globalne
    peaks, _ = find_peaks(signal, height=height, prominence=prominence)

    # tylko piki w długich regionach wklęsłych
    concave_peaks = [
        p for p in peaks
        if any(start <= p <= end for start, end in regions)
    ]
    # znajdź lokalne maksima w całym sygnale
    # peaks, _ = find_peaks(signal, height=height, prominence=prominence)

    # # wybierz tylko te, które leżą w wklęsłych fragmentach
    # concave_peaks = [p for p in peaks if concave_mask[p]]

    return np.array(concave_peaks)


def modified_scholkmann_old(signal, scale=1, threshold=99):
    """
    Zmodyfikowana metoda Scholkmanna do detekcji pików.
    
    Algorytm identyfikuje lokalne maksima w wielu skalach i wybiera
    te punkty, które pozostają maksimami w największej liczbie skal
    (stabilne piki).
    
    Parameters
    ----------
    signal : np.ndarray
        Jednowymiarowy sygnał wejściowy.
    scale : int, optional
        Parametr skali wpływający na maksymalny rozmiar analizowanego
        sąsiedztwa. Domyślnie 1.
    threshold : float, optional
        Percentyl (0–100) liczby wystąpień maksimum w skalach,
        powyżej którego punkt jest uznany za pik.
        Domyślnie 99.
    
    Returns
    -------
    np.ndarray
        Indeksy wykrytych pików
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

def modified_scholkmann(signal, limit=0.5):
    """
    Zmodyfikowana metoda Scholkmanna do detekcji pików.
    
    Algorytm identyfikuje lokalne maksima w wielu skalach i wybiera
    te punkty, które pozostają maksimami w największej liczbie skal
    (stabilne piki).
    
    Parameters
    ----------
    signal : np.ndarray
        Jednowymiarowy sygnał wejściowy.
    scale : int, optional
        Parametr skali wpływający na maksymalny rozmiar analizowanego
        sąsiedztwa. Domyślnie 1.
    threshold : float, optional
        Percentyl (0–100) liczby wystąpień maksimum w skalach,
        powyżej którego punkt jest uznany za pik.
        Domyślnie 99.
    
    Returns
    -------
    np.ndarray
        Indeksy wykrytych pików
    """

    N = len(signal)
    
    # detrending
    t = np.arange(N)
    trend = np.polyval(np.polyfit(t, signal, 1), t)
    x = signal - trend
    
    L = int(np.ceil(N * limit / 2.0)) - 1
    
    gamma = np.zeros(L)
    lms_bool = np.zeros((L, N), dtype=bool)

    #  macierz lokalnych maksimów
    for k in range(1, L + 1):
        # Skrócony zapis wektorowy zamiast pętli po t:
        # Sprawdzamy: x[i-k] < x[i] > x[i+k]
        condition = (x[k:N-k] > x[0:N-2*k]) & (x[k:N-k] > x[2*k:N])
        lms_bool[k-1, k:N-k] = condition
        
        # W klasycznym AMPD w macierzy LMS: 0 = peak, 1 = non-peak
        # Policzmy sumę "nie-pików" w wierszu
        gamma[k-1] = np.sum(~lms_bool[k-1, :])

    # 4. Znalezienie skali 'p'
    # p to indeks wiersza, w którym występuje najwięcej lokalnych maksimów
    p = np.argmin(gamma)
    
    # 5. Selekcja końcowa pików
    # Pikiem jest punkt, który był pikiem we WSZYSTKICH skalach od 1 do p
    # (Suma kolumn w zakresie 0:p musi wynosić 0 w notacji 0/1, 
    #  lub p w naszej notacji boolowskiej True/False)
    
    # Sumujemy kolumny tylko do wiersza p
    column_sum = np.sum(lms_bool[:p, :], axis=0)
    
    # Punkt jest pikiem, jeśli we wszystkich p skalach był maksimum
    peaks = np.where(column_sum == p)[0]

    return np.array(peaks)


def curvature(signal, d2x_threshold=-0.0015, min_len=8):
    """
    Detekcja pików na podstawie lokalnych maksimów krzywizny sygnału.
    
    
    Krzywizna obliczana jest z pierwszej i drugiej pochodnej sygnału.
    Piki krzywizny są następnie filtrowane do obszarów wklęsłych.
     
    Parameters
    ----------
    signal : np.ndarray
        Jednowymiarowy sygnał wejściowy.
    d2x_threshold : float, optional
        Próg drugiej pochodnej definiujący obszary wklęsłe.
        Domyślnie -0.0015.
    prominence : float, optional
        Minimalna wyrazistość piku krzywizny.
        Domyślnie 0.005.
    threshold : float or None, optional
        Minimalna wysokość piku krzywizny (``find_peaks``).
        Domyślnie None.
    
      Returns
      -------
      np.ndarray
          Indeksy wykrytych pików
      """
    dx = np.gradient(signal, edge_order=2)
    d2x = np.gradient(dx, edge_order=2)

    # oblicz krzywiznę z zabezpieczeniem przed dzieleniem przez zero
    denom = (1 + dx**2)**1.5 # mianownik
    denom[denom == 0] = 1e-8
    curvature = np.abs(d2x) / denom

    # obszary wklęsłe
    concave_mask = d2x < d2x_threshold

    # --- regiony wklęsłe ---
    regions = []
    for k, g in groupby(enumerate(concave_mask), key=lambda x: x[1]):
        if k:
            idx = [i for i, _ in g]
            if len(idx) >= min_len:
                regions.append((idx[0], idx[-1]))

    # piki krzywizny
    peaks, _ = find_peaks(curvature)

    concave_peaks = [
        p for p in peaks
        if any(start <= p <= end for start, end in regions)
    ]
    # wykryj piki w krzywiźnie
    # threshold to jak bardzo "wybija się" pik nad pozostałe punkty
    # peaks, _ = find_peaks(curvature, threshold=threshold, prominence=prominence)

    # # wybierz tylko piki w obszarach wklęsłych
    # concave_peaks = [p for p in peaks if concave_mask[p]]

    return np.array(concave_peaks)


def line_distance(signal, d2x_threshold=0.02, mode="perpendicular", min_len=8):
    """
    Detekcja pików na podstawie maksymalnej odległości od linii bazowej.  
    
    Dla każdego wklęsłego regionu sygnału wyznaczana jest linia
    łącząca jego końce, a pik definiowany jest jako punkt
    o maksymalnej odległości od tej linii.
    
    
    Parameters
    ----------
    signal : np.ndarray
        Jednowymiarowy sygnał wejściowy.
    d2x_threshold : float, optional
        Próg drugiej pochodnej definiujący regiony wklęsłe.
        Domyślnie 0.02.
    mode : {"perpendicular", "vertical"}, optional
        Sposób liczenia odległości od linii bazowej:
            - ``"perpendicular"`` – odległość prostopadła,
            - ``"vertical"`` – odległość pionowa.
            Domyślnie "perpendicular".
    min_len : int, optional
        Minimalna długość regionu wklęsłego (liczba próbek),
        aby był analizowany.
        Domyślnie 10.
    
    
    Returns
    -------
    np.ndarray
        Indeksy wykrytych pików
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
    """
    Detekcja pików na podstawie obwiedni sygnału (transformata Hilberta).
    
    
    Parameters
    ----------
    signal : np.ndarray
        Jednowymiarowy sygnał wejściowy.
    prominence : float, optional
        Minimalna wyrazistość piku obwiedni.
        Domyślnie 0.

    Returns
    -------
    np.ndarray
        Indeksy wykrytych pikow
    """
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)  # amplituda "obwiedni"

    # detekcja pików na envelope
    peaks, _ = find_peaks(envelope, prominence)
    
    return np.array(peaks)

def wavelet(signal, w_range=(1,10), step=1):
    """
Detekcja pików z wykorzystaniem ciągłej transformaty falkowej (CWT).
    
    
    Parameters
    ----------
    signal : np.ndarray
        Jednowymiarowy sygnał wejściowy.
    w_range : tuple[int, int], optional
    Zakres szerokości falek (min, max).
    Domyślnie (1, 10).
    step : int, optional
        Krok zmiany szerokości falek.
        Domyślnie 1.
    
    Returns
    -------
    np.ndarray
        Indeksy wykrytych pikow
    """
    # Continuous Wavelet Transform
    widths = np.arange(w_range[0], w_range[1] + 1, step)  # zakres szerokości pików
    peaks = find_peaks_cwt(signal, widths)
    
    return np.array(peaks)

