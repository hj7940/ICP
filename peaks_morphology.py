# -*- coding: utf-8 -*-
"""
Created on Sat Dec 20 20:24:09 2025

Liczenie dla kazdego piku: prominence, width (do funkcji scipy.signal.find_peaks)
@author: User
"""

from scipy.signal import peak_prominences, peak_widths

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from methods import (concave, curvature, modified_scholkmann_old,
                     modified_scholkmann, 
                     line_distance, hilbert_envelope, wavelet)
from ranges import (ranges_full, ranges_pm3, ranges_whiskers, 
                    generate_ranges_for_all_files, compute_ranges_avg)
from main import peak_detection
from main import (
                  it1, it1_smooth_4Hz, it1_smooth_3Hz,
                  ranges_all_time, ranges_all_amps, compute_peak_metrics,
                  df_ranges_time, df_ranges_amps)
from scipy.stats import shapiro, anderson, kstest, norm


def compute_reference_peak_features(dataset, peaks=("P1","P2","P3")):
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

def robust_aggregate(df, value_col):
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

# odrzucenie rozkladu normalnego ! 

features = ["Index", "Height", "Prominence", "Width_50"]

df_agg = (
    df_all
    .groupby(["Class", "Peak"])
    .apply(lambda g: pd.concat([
        robust_aggregate(g, "Index").add_prefix("idx_"),
        robust_aggregate(g, "Height").add_prefix("h_"),
        robust_aggregate(g, "Prominence").add_prefix("prom_"),
        robust_aggregate(g, "Width_50").add_prefix("w50_"),
    ]))
    .reset_index()
)

df_agg.to_csv("peaks_morphology.csv", index=False)

tuned_params = df_agg[["Class", "Peak"]].copy()
for col_prefix in ["idx_", "h_", "prom_"]:
    tuned_params[col_prefix + "lower"] = df_agg[col_prefix + "lower_whisker"]
    tuned_params[col_prefix + "upper"] = df_agg[col_prefix + "upper_whisker"]
    
tuned_params.to_csv("tuned_params.csv", index=False)
