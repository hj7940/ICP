# -*- coding: utf-8 -*-
"""
Created on Sat Dec 20 20:24:09 2025

Liczenie dla kazdego piku: prominence, width (do funkcji scipy.signal.find_peaks)
oraz index (polozenie na osi x - do zakresow),
sprawdzenie rozkladu normalnego
generowanie boxplotow 
@author: User
"""

from scipy.signal import peak_prominences, peak_widths
import pandas as pd
import numpy as np
from main import it1
from scipy.stats import shapiro, anderson, kstest
from all_plots import plot_peak_features_boxplots


def compute_reference_peak_features(dataset, peaks=("P1","P2","P3")):
    """
    Liczenie cech pikow (referencyjnych):
        - wysokosc height
        - wyrazistosc prominence
        - polozenie w czasie (index)
        - szerokość piku dla różnych poziomów względnej wysokości (width)

    Parameters
    ----------
    dataset : list of ditcs
        Lista slownikow, z ktorych kazdy zawiera infomacje o pojedynczym sygnale
    peaks : tuple of str
        Nazwy pików referencyjnych, które mają być analizowane.
        Domyślnie ``("P1", "P2", "P3")``

    Returns
    -------
    pandas.DataFrame
        Tabela, w której każdy wiersz odpowiada jednemu pikowi referencyjnemu
        w jednym sygnale

    """
    rows = []

    for item in dataset:
        class_id = item["class"]
        file = item["file"]
        sig = item["signal"]

        y = sig.iloc[:, 1].values

        for p in peaks:
            ref_idx = item["peaks_ref"].get(p)

            # brak piku referencyjnego (np. Class4 P1/P3)
            if ref_idx is None:
                continue


            # --- metryki scipy ---
            prom = peak_prominences(y, [ref_idx])[0][0]

            w50 = peak_widths(y, [ref_idx], rel_height=0.5)
            w25 = peak_widths(y, [ref_idx], rel_height=0.25)
            w75 = peak_widths(y, [ref_idx], rel_height=0.75)

            rows.append({
                "Class": class_id,
                "File": file,
                "Peak": p,
                "Index": ref_idx,
                "Height": y[ref_idx],
                "Prominence": prom,
                "Width_50": w50[0][0],
                "Width_25": w25[0][0],
                "Width_75": w75[0][0],
            })

    return pd.DataFrame(rows)


def test_normality_SAK(x, alpha=0.05):
    """
    Testy normalnosci rozkladu:
        - Shapiro-Wilka,
        - Andersona-Darlinga (dla rozkladu normalnego)
        - Kolomogorowa-Smirnowa

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    alpha : float, optional
        Poziom istotnosci - domyslnie 0.05

    Returns
    -------
    dict
        Słownik zawierający:
        - ``N`` – liczebność próby,
        - ``Shapiro_p`` – wartość p testu Shapiro–Wilka,
        - ``Shapiro_normal`` – wynik testu (True/False),
        - ``AD_stat`` – statystyka testu Andersona–Darlinga,
        - ``AD_crit_5`` – wartość krytyczna dla poziomu 5%,
        - ``AD_normal`` – wynik testu Andersona–Darlinga,
        - ``KS_p`` – wartość p testu Kołmogorowa–Smirnowa,
        - ``KS_normal`` – wynik testu KS.

    """
    x = np.asarray(x)
    x = x[~np.isnan(x)]

    res = {
        "N": len(x),
        "Shapiro_p": np.nan,
        "Shapiro_normal": np.nan,
        "AD_stat": np.nan,
        "AD_crit_5": np.nan,
        "AD_normal": np.nan,
        "KS_p": np.nan,
        "KS_normal": np.nan
    }

    if len(x) < 5:
        return res

    # --- Shapiro-Wilk ---
    stat, p = shapiro(x)
    res["Shapiro_p"] = p
    res["Shapiro_normal"] = p >= alpha

    # --- Anderson-Darling ---
    ad = anderson(x, dist="norm")
    crit_5 = ad.critical_values[list(ad.significance_level).index(5.0)]
    res["AD_stat"] = ad.statistic
    res["AD_crit_5"] = crit_5
    res["AD_normal"] = ad.statistic < crit_5

    # --- Kolmogorov–Smirnov (z estymacją μ, σ) ---
    mu, sigma = np.mean(x), np.std(x, ddof=1)
    if sigma > 0:
        ks_stat, ks_p = kstest(x, "norm", args=(mu, sigma))
        res["KS_p"] = ks_p
        res["KS_normal"] = ks_p >= alpha

    return res

def aggregate(df, value_col):
    """
    Funkcja oblicza statystyki oparte na medianie i rozstępie
    międzykwartylowym (IQR), dodatkowo liczona sa wasy (1.5 * IQR)

    Parameters
    ----------
    df : pandas.DataFrame
        DESCRIPTION.
    value_col : str
        Nazwa kolumny w ``df``, dla której obliczane są statystyki.

    Returns
    -------
    pandas.Series
        Zestaw statystyk opisowych:
        - ``median`` – mediana,
        - ``min`` – wartość minimalna,
        - ``max`` – wartość maksymalna,
        - ``q25`` – pierwszy kwartyl,
        - ``q75`` – trzeci kwartyl,
        - ``iqr`` – rozstęp międzykwartylowy,
        - ``lower_whisker`` – dolny wąs boxplotu (Q1 − 1.5·IQR),
        - ``upper_whisker`` – górny wąs boxplotu (Q3 + 1.5·IQR),
        - ``zero_frac`` – udział wartości równych zero w próbie.

    """
    x = df[value_col].dropna().values

    q25 = np.percentile(x, 25)
    q75 = np.percentile(x, 75)
    iqr = q75 - q25
    lower_whisker = q25 - 1.5 * iqr
    upper_whisker = q75 + 1.5 * iqr
    

    return pd.Series({
        "median": np.median(x),
        "min": np.min(x),
        "max": np.max(x),
        "q25": q25,
        "q75": q75,
        "iqr": iqr,
        "lower_whisker": lower_whisker,
        "upper_whisker": upper_whisker,
        "zero_frac": np.mean(x == 0),
    })


df_all = compute_reference_peak_features(dataset=it1)


features = ["Index", "Height", "Prominence", "Width_50", "Width_25", "Width_75"]

rows = []

for (class_id, pk), grp in df_all.groupby(["Class", "Peak"]):
    for feat in features:
        out = test_normality_SAK(grp[feat])

        rows.append({
            "Class": class_id,
            "Peak": pk,
            "Feature": feat,
            **out
        })

df_normality = pd.DataFrame(rows)

df_normality["Normal_Distribution"] = (
    df_normality["Shapiro_normal"] &
    df_normality["AD_normal"] &
    df_normality["KS_normal"]
)

# df_normality.to_csv("normality.csv", index=False)
# odrzucenie rozkladu normalnego ! 

features = ["Index", "Height", "Prominence", "Width_50"]

df_agg = (
    df_all
    .groupby(["Class", "Peak"])
    .apply(lambda g: pd.concat([
        aggregate(g, "Index").add_prefix("idx_"),
        aggregate(g, "Height").add_prefix("h_"),
        aggregate(g, "Prominence").add_prefix("prom_"),
        aggregate(g, "Width_50").add_prefix("w50_"),
    ]))
    .reset_index()
)

# df_agg.to_csv("peaks_morphology.csv", index=False)

# tuned_params = df_agg[["Class", "Peak"]].copy()
# for col_prefix in ["idx_", "h_", "prom_"]:
#     tuned_params[col_prefix + "lower"] = df_agg[col_prefix + "lower_whisker"]
#     tuned_params[col_prefix + "upper"] = df_agg[col_prefix + "upper_whisker"]
    
# tuned_params.to_csv("tuned_params.csv", index=False)


plot_peak_features_boxplots(df_all, features=["Index", "Height","Prominence"])
