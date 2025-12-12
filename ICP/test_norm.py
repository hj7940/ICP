# -*- coding: utf-8 -*-
"""
sprawdzenie rozkladu normalnego
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.stats import shapiro, anderson, kstest

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

def compute_statistics_raw(data):
    """
    Zwraca pełne dane (bez uśredniania):
    - amplituda maksymalna sygnału
    - AUC
    - czasy pików P1, P2, P3
    - amplitudy pików P1, P2, P3
    Każdy wiersz to jeden sygnał.
    """

    rows = []

    for i in range(1, 5):
        class_name = f"Class{i}"
        signals = data[f"{class_name}_signals"]
        peaks_df = data[f"{class_name}_peaks"]
        files = data[f"{class_name}_files"]

        for idx, sig in enumerate(signals):

            # dane sygnału
            t = sig.iloc[:, 0].values
            y = sig.iloc[:, 1].values

            max_amp = y.max()
            auc = np.trapezoid(y, t)

            # znajdujemy pikowe informacje z tabeli peaks_df
            file_name_only = os.path.splitext(os.path.basename(files[idx]))[0]
            row_peak = peaks_df[peaks_df["File"] == file_name_only]


            if len(row_peak) == 1:
                row_peak = row_peak.iloc[0]
            else:
                row_peak = None

            # inicjalizacja pików
            p_times = {"P1": np.nan, "P2": np.nan, "P3": np.nan}
            p_amps  = {"P1": np.nan, "P2": np.nan, "P3": np.nan}
            p_difs = {"P1P2": np.nan, "P2P3": np.nan}
            if row_peak is not None:
                for p in ["P1", "P2", "P3"]:
                    pk = row_peak[p]
                    if pd.notna(pk) and pk != -1:
                        pk_idx = int(pk)
                        p_times[p] = t[pk_idx]
                        p_amps[p]  = y[pk_idx]
                if not np.isnan(p_times["P1"]) and not np.isnan(p_times["P2"]):
                    p_difs["P1P2"] = p_times["P2"] - p_times["P1"]

                if not np.isnan(p_times["P2"]) and not np.isnan(p_times["P3"]):
                    p_difs["P2P3"] = p_times["P3"] - p_times["P2"]
           
            # wiersz wynikowy
            rows.append({
                "Class": class_name,
                "File": file_name_only,
                "Max_Amplitude": max_amp,
                "AUC": auc,
                "P1_Time": p_times["P1"],
                "P2_Time": p_times["P2"],
                "P3_Time": p_times["P3"],
                "P1_Amp":  p_amps["P1"],
                "P2_Amp":  p_amps["P2"],
                "P3_Amp":  p_amps["P3"],
                "P1-P2": p_difs["P1P2"],
                "P2-P3": p_difs["P2P3"]
            })

    df = pd.DataFrame(rows)
    return df


def normality_tests(values, alpha=0.05):
    """
    Wykonuje 3 testy normalności:
    1. Shapiro–Wilka
    2. Anderson–Darling (Darlinga)
    3. Kolmogorow–Smirnow (dla N(mu, sigma))
    Zwraca słownik wyników + interpretację, czy dane mają rozkład normalny.
    """
    values = np.array(values)
    values = values[~np.isnan(values)]

    if len(values) < 4:
        return {
            "Shapiro_W": (np.nan, np.nan),
            "Anderson_Darling": (np.nan, np.nan),
            "KS": (np.nan, np.nan)
        }

    # Parametry rozkładu
    mu = np.mean(values)
    sigma = np.std(values)

    # 1. Shapiro-Wilk
    sh_w, sh_p = shapiro(values)
    sh_normal = sh_p > alpha # True = dane zgodne z normalnym

    # 2. Anderson-Darling (Darlinga)
    ad_res = anderson(values, dist='norm')
    ad_stat = ad_res.statistic
    # Szacunkowe p-value: nie ma dokładnego w SciPy, można użyć przybliżenia:
    # przy ad_stat < 0.5 → p ~ 0.25
    # ad_stat ~ 1 → p ~ 0.1
    # lub pozostawić same krytyczne wartości
    ad_critical_values = ad_res.critical_values
    ad_significance_level = ad_res.significance_level
    try:
        idx_5 = list(ad_res.significance_level).index(5.0)
        ad_normal = ad_stat < ad_res.critical_values[idx_5]  # True = normalne
    except ValueError:
        ad_normal = np.nan  # jeśli brak poziomu 5% w danych SciPy
        
    # 3. Kolmogorow–Smirnow (parametry znane)
    ks_stat, ks_p = kstest(values, 'norm', args=(mu, sigma))
    ks_normal = ks_p > alpha
    
    return {
        "Shapiro_W": (sh_w, sh_p, sh_normal),
        "Anderson_Darling": (ad_stat, ad_critical_values, ad_normal),
        "KS": (ks_stat, ks_p, ks_normal)
    }

def run_all_normality_tests(df_raw):
    variables = ["P1_Time", "P2_Time", "P3_Time",
                 "P1_Amp", "P2_Amp", "P3_Amp", "P1-P2", "P2-P3"]
    
    results = []

    for cls in sorted(df_raw["Class"].unique()):
        df_cls = df_raw[df_raw["Class"] == cls]

        for var in variables:
            values = df_cls[var].dropna().values

            res = normality_tests(values)

            results.append({
                "Class": cls,
                "Variable": var,
                "N": len(values),
                "Shapiro_Statistic": res["Shapiro_W"][0],
                "Shapiro_p":         res["Shapiro_W"][1],
                "Anderson_Darling": res["Anderson_Darling"][0],
                "Anderson_Darling_p":         res["Anderson_Darling"][1],
                "KS_Statistic":      res["KS"][0],
                "KS_p":              res["KS"][1]
            })

    return pd.DataFrame(results)

def print_normality_summary(test_results_df, alpha=0.05):
    """
    Dla każdego wiersza DataFrame z wynikami testów normalności
    wypisuje interpretację w czytelny sposób.
    """
    for cls in sorted(test_results_df["Class"].unique()):
        print(f"\n=== {cls} ===")
        df_cls = test_results_df[test_results_df["Class"] == cls]

        for _, row in df_cls.iterrows():
            var = row["Variable"]
            n = row["N"]

            # Shapiro-Wilk
            sh_p = row["Shapiro_p"]
            sh_norm = sh_p > alpha if not np.isnan(sh_p) else False

            # Kolmogorov-Smirnov
            ks_p = row["KS_p"]
            ks_norm = ks_p > alpha if not np.isnan(ks_p) else False

            # Anderson-Darling (sprawdzamy, czy statystyka < krytyczna wartość 5%)
            try:
                ad_crit_5 = row["Anderson_Darling_p"][list(row["Anderson_Darling_p"]).index(5.0)]
                ad_stat = row["Anderson_Darling"]
                ad_norm = ad_stat < ad_crit_5
            except (ValueError, TypeError):
                ad_norm = False

            print(f"\n{var} (N={n}):")
            print(f" Shapiro-Wilk: p={sh_p:.4f} -> {'Normalny' if sh_norm else 'Nie normalny'}")
            print(f" Anderson-Darling: A={row['Anderson_Darling']:.4f} -> {'Normalny' if ad_norm else 'Nie normalny'}")
            print(f" Kolmogorov-Smirnov: p={ks_p:.4f} -> {'Normalny' if ks_norm else 'Nie normalny'}")


def generate_boxplots_percentiles(df, output_folder):
    """
    4 podwykresy (1 na klasę), w każdym boxplot dla P1/P2/P3.
    Zapisuje wykres PNG i CSV z percentylami.
    """
    os.makedirs(output_folder, exist_ok=True)
    all_percentiles = []

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, cls in enumerate(sorted(df["Class"].unique())):
        df_cls = df[df["Class"] == cls]
        box_data = []
        labels = []

        for p in ["P1", "P2", "P3"]:
            col = f"{p}_Time"
            values = df_cls[col].dropna().values
            if len(values) == 0:
                continue
            box_data.append(values)
            labels.append(p)

            # percentyle
            percentiles = np.percentile(values, [0, 25, 50, 75, 100])
            Q1 = np.percentile(values, 25)
            Q3 = np.percentile(values, 75)
            IQR = Q3 - Q1
            lower_whisker = max(values.min(), Q1 - 1.5 * IQR)
            upper_whisker = min(values.max(), Q3 + 1.5 * IQR)
            all_percentiles.append({
                "Class": cls,
                "Variable": col,
                "Min": percentiles[0],
                "Q1": percentiles[1],
                "Median": percentiles[2],
                "Q3": percentiles[3],
                "Max": percentiles[4],
                "Lower_Whisker": lower_whisker,
                "Upper_Whisker": upper_whisker
            })

        if box_data:
            axes[i].boxplot(box_data, labels=labels, whis=1.5)
            axes[i].set_title(f"Klasa {i+1}", fontsize=16)
            axes[i].set_ylabel("Czas występowania", fontsize=14)
            axes[i].tick_params(axis='x', labelsize=14)

    plt.tight_layout(w_pad=4, h_pad=4)
    # plt.savefig(os.path.join(output_folder, "boxplots_all_classes.png"))
    # plt.close()

    # CSV z percentylami
    percentiles_df = pd.DataFrame(all_percentiles)
    percentiles_df.to_csv(os.path.join(output_folder, "boxplot_percentiles.csv"), index=False)

def generate_boxplots_amplitudes(df, output_folder):
    """
    Tworzy boxploty dla amplitud P1, P2, P3 osobno dla każdej klasy.
    Zapisuje PNG oraz CSV z percentylami.
    """
    os.makedirs(output_folder, exist_ok=True)
    all_percentiles = []

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for i, cls in enumerate(sorted(df["Class"].unique())):
        df_cls = df[df["Class"] == cls]
        box_data = []
        labels = []

        for p in ["P1", "P2", "P3"]:
            col = f"{p}_Amp"
            values = df_cls[col].dropna().values
            if len(values) == 0:
                continue

            box_data.append(values)
            labels.append(p)

            # percentyle — całkowicie analogicznie jak w wersji dla czasów
            percentiles = np.percentile(values, [0, 25, 50, 75, 100])
            Q1 = np.percentile(values, 25)
            Q3 = np.percentile(values, 75)
            IQR = Q3 - Q1
            lower_whisker = max(values.min(), Q1 - 1.5 * IQR)
            upper_whisker = min(values.max(), Q3 + 1.5 * IQR)

            all_percentiles.append({
                "Class": cls,
                "Variable": col,
                "Min": percentiles[0],
                "Q1": percentiles[1],
                "Median": percentiles[2],
                "Q3": percentiles[3],
                "Max": percentiles[4],
                "Lower_Whisker": lower_whisker,
                "Upper_Whisker": upper_whisker
            })

        if box_data:
            axes[i].boxplot(box_data, labels=labels, whis=1.5)
            axes[i].set_title(f"Klasa {i+1}", fontsize=16)
            axes[i].set_ylabel("Amplituda", fontsize=14)
            axes[i].tick_params(axis='x', labelsize=14)

    plt.tight_layout(w_pad=4, h_pad=4)
    # plt.savefig(os.path.join(output_folder, "boxplots_amplitudes_all_classes.png"))
    # plt.close()

    # CSV z percentylami amplitud
    percentiles_df = pd.DataFrame(all_percentiles)
    percentiles_df.to_csv(os.path.join(output_folder, "boxplot_percentiles_amplitudes.csv"), index=False)


def generate_violinplots(df, output_folder):
    """
    4 podwykresy (1 na klasę), w każdym violin plot dla P1/P2/P3.
    Zapisuje wykres PNG.
    """
    os.makedirs(output_folder, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, cls in enumerate(sorted(df["Class"].unique())):
        df_cls = df[df["Class"] == cls]
        violin_data = []
        labels = []

        for p in ["P1", "P2", "P3"]:
            col = f"{p}_Time"
            values = df_cls[col].dropna().values
            if len(values) == 0:
                continue
            violin_data.append(values)
            labels.append(p)

        if violin_data:
            axes[i].violinplot(violin_data, showmeans=True, showmedians=True)
            axes[i].set_xticks(range(1, len(labels)+1))
            axes[i].set_xticklabels(labels)
            axes[i].set_title(cls)
            axes[i].set_ylabel("Time [ms]")

    plt.tight_layout(w_pad=4, h_pad=4)
    # plt.savefig(os.path.join(output_folder, "violinplots_all_classes.png"))
    # plt.close()
                  
data = load_data(r"C:\Users\User\OneDrive\Dokumenty\praca inżynierska\ICP_pulses_it1")

df_raw = compute_statistics_raw(data)

# test_results_df =run_all_normality_tests(df_raw)
# df_raw.to_csv("all_raw_values.csv", index=False)
# print(test_results_df)

# test_results_df.to_csv("test_results_w_diff.csv", index=False)


#print_normality_summary(test_results_df)
output_folder = r"C:\Users\User\OneDrive\Dokumenty\praca inżynierska\plots"
generate_boxplots_percentiles(df_raw, output_folder)
# generate_violinplots(df_raw, output_folder)

generate_boxplots_amplitudes(df_raw, output_folder)