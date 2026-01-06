# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 03:30:43 2025

@author: User
"""


# # ------------ NORMALNY, BEZ ZAKRESOW --------------
# all_metrics = []

# for method_name in METHODS.keys():
#     for cls in ["Class1", "Class2", "Class3", "Class4"]:
#         for peak_name in ["P1","P2","P3"]:
#             detected_all = []

#             for item in data_it1:
#                 if item["class"] != cls:
#                     continue
#                 file_name = item["file"]

#                 detected = single_peak_detection(
#                     peak_name=peak_name,
#                     class_name=cls,
#                     file_name=file_name,
#                     dataset=[item],
#                     method_name=method_name,
#                     range_type=None,
#                     peak_ranges_file=None,
#                     peak_amps_file=None
#                 )
#                 detected_all.extend(detected)

#             # metryki dla wszystkich sygnałów tej klasy i piku
#             df_metrics = compute_peak_metrics(
#                 dataset=[item for item in analyzed_data if item["class"] == cls],
#                 detected_peaks=detected_all,
#                 peak_name=peak_name,
#                 class_name=cls
#             )
#             df_metrics["Method"] = method_name
#             all_metrics.append(df_metrics)

# # scal wszystkie metryki
# df_all_metrics = pd.concat(all_metrics, ignore_index=True)
# df_avg_metrics = df_all_metrics.groupby(["Class", "Peak", "Method"]).mean(numeric_only=True).reset_index()
# print(df_avg_metrics)

# # ------------ SMOOTH, BEZ ZAKRESOW -----------
# all_metrics_smooth = []

# for method_name in METHODS.keys():
#     for cls in ["Class1", "Class2", "Class3", "Class4"]:
#         for peak_name in ["P1","P2","P3"]:
#             detected_all = []

#             for item in data_smooth_it1:
#                 if item["class"] != cls:
#                     continue
#                 file_name = item["file"]

#                 detected = single_peak_detection(
#                     peak_name=peak_name,
#                     class_name=cls,
#                     file_name=file_name,
#                     dataset=[item],
#                     method_name=method_name,
#                     range_type=None,
#                     peak_ranges_file=None,
#                     peak_amps_file=None
#                 )
#                 detected_all.extend(detected)

#             # metryki dla wszystkich sygnałów tej klasy i piku
#             df_metrics = compute_peak_metrics(
#                 dataset=[item for item in analyzed_data if item["class"] == cls],
#                 detected_peaks=detected_all,
#                 peak_name=peak_name,
#                 class_name=cls
#             )
#             df_metrics["Method"] = method_name
#             all_metrics_smooth.append(df_metrics)

# # scal wszystkie metryki
# df_all_metrics_smooth = pd.concat(all_metrics_smooth, ignore_index=True)
# df_avg_metrics_smooth = df_all_metrics_smooth.groupby(["Class", "Peak", "Method"]).mean(numeric_only=True).reset_index()
# print(df_avg_metrics_smooth)

# # -------- NORMALNE, ZAKRESY MINMAX-------------
# all_metrics_ranges = []

# for method_name in METHODS.keys():
#     for cls in ["Class1", "Class2", "Class3", "Class4"]:
#         for peak_name in ["P1","P2","P3"]:
#             detected_all = []

#             for item in data_it1:
#                 if item["class"] != cls:
#                     continue
#                 file_name = item["file"]

#                 detected = single_peak_detection(
#                     peak_name=peak_name,
#                     class_name=cls,
#                     file_name=file_name,
#                     dataset=[item],
#                     method_name=method_name,
#                     range_type="static",
#                     peak_ranges_file=peak_ranges_minmax,
#                     peak_amps_file=peak_amps_minmax
#                 )
#                 detected_all.extend(detected)

#             # metryki dla wszystkich sygnałów tej klasy i piku
#             df_metrics = compute_peak_metrics(
#                 dataset=[item for item in analyzed_data if item["class"] == cls],
#                 detected_peaks=detected_all,
#                 peak_name=peak_name,
#                 class_name=cls
#             )
#             df_metrics["Method"] = method_name
#             all_metrics_ranges.append(df_metrics)

# # scal wszystkie metryki
# df_all_metrics_ranges = pd.concat(all_metrics_ranges, ignore_index=True)
# df_avg_metrics_ranges = df_all_metrics_ranges.groupby(["Class", "Peak", "Method"]).mean(numeric_only=True).reset_index()
# print(df_avg_metrics_ranges)

# # -------- SMOOTH, ZAKRESY MINMAX --------------
# all_metrics_ranges_smooth = []

# for method_name in METHODS.keys():
#     for cls in ["Class1", "Class2", "Class3", "Class4"]:
#         for peak_name in ["P1","P2","P3"]:
#             detected_all = []

#             for item in data_smooth_it1:
#                 if item["class"] != cls:
#                     continue
#                 file_name = item["file"]

#                 detected = single_peak_detection(
#                     peak_name=peak_name,
#                     class_name=cls,
#                     file_name=file_name,
#                     dataset=[item],
#                     method_name=method_name,
#                     range_type="static",
#                     peak_ranges_file=peak_ranges_minmax,
#                     peak_amps_file=peak_amps_minmax
#                 )
#                 detected_all.extend(detected)

#             # metryki dla wszystkich sygnałów tej klasy i piku
#             df_metrics = compute_peak_metrics(
#                 dataset=[item for item in analyzed_data if item["class"] == cls],
#                 detected_peaks=detected_all,
#                 peak_name=peak_name,
#                 class_name=cls
#             )
#             df_metrics["Method"] = method_name
#             all_metrics_ranges_smooth.append(df_metrics)

# # scal wszystkie metryki
# df_all_metrics_ranges_smooth = pd.concat(all_metrics_ranges_smooth, ignore_index=True)
# df_avg_metrics_ranges_smooth = df_all_metrics_ranges_smooth.groupby(["Class", "Peak", "Method"]).mean(numeric_only=True).reset_index()
# print(df_avg_metrics_ranges_smooth)


def load_data_it1(base_path):
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

def load_data_it2(base_path):
    dataset = []

    for i in range(1, 5):
        class_name = f"Class{i}"

        # peaks CSV
        peaks_path = os.path.join(base_path, f"{class_name}_it2_peaks.csv")
        peaks_df = pd.read_csv(peaks_path)

        # sygnały
        folder = os.path.join(base_path, class_name)
        csv_files = sorted(glob.glob(os.path.join(folder, f"{class_name}_it2_example_*.csv")))

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


from matplotlib_venn import venn3, venn3_unweighted 
from matplotlib import colors as mcolors
def plot_venn_for_class(results_combined, class_name, ax=None):
    """
    Diagram Venna dla P1, P2, P3 w danej klasie
    """

    # liczniki
    only_P1 = only_P2 = only_P3 = 0
    P1_P2 = P1_P3 = P2_P3 = 0
    P1_P2_P3 = 0

    for item in results_combined:
        if item["class"] != class_name:
            continue

        p = item["peaks_detected"]

        has_P1 = not (p["P1"] is None or (isinstance(p["P1"], float) and math.isnan(p["P1"])))
        has_P2 = not (p["P2"] is None or (isinstance(p["P2"], float) and math.isnan(p["P2"])))
        has_P3 = not (p["P3"] is None or (isinstance(p["P3"], float) and math.isnan(p["P3"])))

        if has_P1 and not has_P2 and not has_P3:
            only_P1 += 1
        elif has_P2 and not has_P1 and not has_P3:
            only_P2 += 1
        elif has_P3 and not has_P1 and not has_P2:
            only_P3 += 1
        elif has_P1 and has_P2 and not has_P3:
            P1_P2 += 1
        elif has_P1 and has_P3 and not has_P2:
            P1_P3 += 1
        elif has_P2 and has_P3 and not has_P1:
            P2_P3 += 1
        elif has_P1 and has_P2 and has_P3:
            P1_P2_P3 += 1
    
    # oryginalne wartości
    original_counts = (only_P1, only_P2, P1_P2,
                       only_P3, P1_P3, P2_P3, P1_P2_P3)
    
    # skalowanie wizualne: pierwiastek z liczby, aby zbliżyć wielkości kół
    scaled_counts = tuple(c**0.001 if c>0 else 0 for c in original_counts)
    # funkcja wyświetlająca ORYGINALNE liczby
    
    def label_formatter(scaled_value, original_values=original_counts, scaled_values=scaled_counts):
    # znajdź indeks scaled_value w scaled_counts i zwróć odpowiadającą wartość z original_counts
        try:
            idx = scaled_values.index(scaled_value)
            return str(original_values[idx])
        except ValueError:
            return str(int(round(scaled_value)))

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
        
    v = venn3(
        subsets=scaled_counts,
        set_labels=("P1", "P2", "P3"),
        ax=ax,
        subset_label_formatter=label_formatter
    )
    
        # tylko P1
    if v.get_patch_by_id("100") is not None:
        v.get_patch_by_id("100").set_color('red')
        v.get_patch_by_id("100").set_alpha(0.55)
        v.get_patch_by_id('100').set_edgecolor('none')
    # tylko P2
    if v.get_patch_by_id("010") is not None:
        v.get_patch_by_id("010").set_color('green')
        v.get_patch_by_id("010").set_alpha(0.55)
        v.get_patch_by_id('010').set_edgecolor('none')
    # tylko P3
    if v.get_patch_by_id("001") is not None:
        v.get_patch_by_id("001").set_color('deepskyblue')
        v.get_patch_by_id("001").set_alpha(0.55)
        v.get_patch_by_id('001').set_edgecolor('none')
    # P1+P2
    if v.get_patch_by_id("110") is not None:
        v.get_patch_by_id("110").set_color('saddlebrown')
        v.get_patch_by_id("110").set_alpha(0.55)
        v.get_patch_by_id('110').set_edgecolor('none')
    # P1+P3
    if v.get_patch_by_id("101") is not None:
        v.get_patch_by_id("101").set_color('darkviolet')
        v.get_patch_by_id("101").set_alpha(0.55)
        v.get_patch_by_id('101').set_edgecolor('none')
    # P2+P3
    if v.get_patch_by_id("011") is not None:
        v.get_patch_by_id("011").set_color('mediumspringgreen')
        v.get_patch_by_id("011").set_alpha(0.55)
        v.get_patch_by_id('011').set_edgecolor('none')
    # P1+P2+P3
    if v.get_patch_by_id("111") is not None:
        v.get_patch_by_id("111").set_color('grey')
        v.get_patch_by_id("111").set_alpha(0.55)
        v.get_patch_by_id('111').set_edgecolor('none')
    

    # podstawowe kolory z matplotlib
    c_red   = mcolors.to_rgb("tomato")   # P1
    c_green = mcolors.to_rgb("limegreen")# P2
    c_blue  = mcolors.to_rgb("dodgerblue") # P3
    
    # mieszanki addytywne
    def mix_colors(*cols):
        r = sum(c[0] for c in cols)
        g = sum(c[1] for c in cols)
        b = sum(c[2] for c in cols)
        # normalizacja do max 1
        m = max(r,g,b)
        if m>1: r,g,b = r/m, g/m, b/m
        return (r,g,b,0.75)  # dodajemy alpha
    
    # ustawienie kolorów ręcznie
    if v.get_patch_by_id("100") is not None:  # P1
        v.get_patch_by_id("100").set_color(c_red + (0.75,))
    if v.get_patch_by_id("010") is not None:  # P2
        v.get_patch_by_id("010").set_color(c_green + (0.75,))
    if v.get_patch_by_id("001") is not None:  # P3
        v.get_patch_by_id("001").set_color(c_blue + (0.75,))
    if v.get_patch_by_id("110") is not None:  # P1+P2
        v.get_patch_by_id("110").set_color(mix_colors(c_red, c_green))
    if v.get_patch_by_id("101") is not None:  # P1+P3
        v.get_patch_by_id("101").set_color(mix_colors(c_red, c_blue))
    if v.get_patch_by_id("011") is not None:  # P2+P3
        v.get_patch_by_id("011").set_color(mix_colors(c_green, c_blue))
    if v.get_patch_by_id("111") is not None:  # P1+P2+P3
        v.get_patch_by_id("111").set_color(mix_colors(c_red, c_green, c_blue))

    ax.set_title(class_name)
    
    
# datasets = [
#     (it2, "it2"),
#     (it2_smooth_4Hz, "it2_smooth_4Hz"),
#     (it2_smooth_3Hz, "it2_smooth_3Hz"),
# ]

#c1_p1=peak_detection_single(it2, "concave", "Class1", "P1", df_ranges_time.loc["it2", "avg"], None, "avg")
 # -------------------- lekka krakasa, moznaby rzec ----------------
# --- wariant a ---
# df_variant_a = pd.DataFrame([
#     # -------- Class1 --------
#     ("Class1", "P1", "it2",            "it2",            "avg",  "concave"),
#     ("Class1", "P2", "it2_smooth_4Hz", "it2_smooth_4Hz", "avg",  "curvature"),
#     ("Class1", "P3", "it2_smooth_4Hz", "it2_smooth_4Hz", "full", "concave"),

#     # -------- Class2 --------
#     ("Class2", "P1", "it2_smooth_3Hz", "it2_smooth_3Hz", "avg",  "hilbert"),
#     ("Class2", "P2", "it2",            "it2",            "whiskers", "wavelet"),
#     ("Class2", "P3", "it2",            "it2",            "avg",      "curvature"),

#     # -------- Class3 --------
#     ("Class3", "P1", "it2_smooth_4Hz", "it2_smooth_4Hz", "full", "wavelet"),
#     ("Class3", "P2", "it2",            "it2",            "whiskers", "concave"),
#     ("Class3", "P3", "it2",            "it2",            "none", "modified_scholkmann_1-2_99"),

#     # -------- Class4 --------
#     ("Class4", "P2", "it2_smooth_4Hz", "it2_smooth_4Hz", "none", "modified_scholkmann_1-2_99"),
# ],
# columns=[
#     "class",
#     "peak",
#     "detect_dataset",
#     "ranges_dataset",
#     "range_type",
#     "method"
# ])

# # --- wariant b ---
# df_variant_b = pd.DataFrame([
#     # -------- Class1 --------
#     ("Class1", "P1", "it2",            "it2",            "full", "concave"),
#     ("Class1", "P2", "it2",            "it2",            "avg",  "curvature"),
#     ("Class1", "P3", "it2_smooth_4Hz", "it2_smooth_4Hz", "full", "concave"),

#     # -------- Class2 --------
#     ("Class2", "P1", "it2_smooth_3Hz", "it2",            "avg",  "wavelet"),
#     ("Class2", "P2", "it2_smooth_4Hz", "it2_smooth_4Hz", "full", "curvature"),
#     ("Class2", "P3", "it2",            "it2",            "whiskers", "hilbert"),

#     # -------- Class3 --------
#     ("Class3", "P1", "it2_smooth_4Hz", "it2_smooth_4Hz", "full", "wavelet"),
#     ("Class3", "P2", "it2_smooth_4Hz", "it2_smooth_4Hz", "full",     "concave"),
#     ("Class3", "P3", "it2",            "it2",            "none", "modified_scholkmann_1-2_99"),

#     # -------- Class4 --------
#     ("Class4", "P2", "it2_smooth_4Hz", "it2_smooth_4Hz", "none", "modified_scholkmann_1-2_99"),
# ],
# columns=[
#     "class",
#     "peak",
#     "detect_dataset",
#     "ranges_dataset",
#     "range_type",
#     "method"
# ])

#%% A IT1
# results_a = run_variant(
#     df_variant_a,
#     datasets_dict=datasets_dict,
#     ranges_all_time=ranges_all_time,
#     ranges_all_amps=ranges_all_amps
# )


# results_combined_a = combine_peaks_by_file(results_a)
# # results_combined_a_pp = postprocess_peaks(results_combined_a)


# df_pogladowe_a_pp = pd.DataFrame([
#     {
#         "file": d["file"],
#         "peaks_ref": d["peaks_ref"],
#         "peaks_detected": d["peaks_detected"],
#     }
#     for d in results_combined_a
# ])

# mask = df_pogladowe_a_pp["peaks_detected"].apply(
#     lambda d: any(len(d[p]) > 1 for p in ["P2"])
# )
# rows_multi_a = df_pogladowe_a_pp[mask]
# print(rows_multi_a)
# mask = df_pogladowe_a_pp["peaks_detected"].apply(
#     lambda d: any(len(d[p]) > 1 for p in ["P1", "P2", "P3"])
# )

# rows_multi_a = df_pogladowe_a_pp[mask]
# print(rows_multi_a)

# count_a = df_pogladowe_a["peaks_detected"].apply(
#     lambda d: len(d["P1"]) == 0 or len(d["P2"]) == 0 or len(d["P3"]) == 0
# ).sum()

# empty_lists_count_a = df_pogladowe_a["peaks_detected"].apply(
#     lambda d: (len(d["P1"]) == 0) + (len(d["P2"]) == 0) + (len(d["P3"]) == 0)
# ).sum()

# print(count_a, empty_lists_count_a)


# plot_files_in_class(results_combined, "Class1")
# plot_files_in_class(results_combined, "Class2")
# plot_files_in_class(results_combined, "Class3")
# plot_files_in_class(results_combined, "Class4")

# %% B
# results_b = run_variant(
#     df_variant_a,
#     datasets_dict=datasets_dict,
#     ranges_all_time=ranges_all_time,
#     ranges_all_amps=ranges_all_amps
# )


# results_combined_b = combine_peaks_by_file(results_b)
# # results_combined_b_pp = postprocess_peaks(results_combined_b)

# df_pogladowe_b_pp = pd.DataFrame([
#     {
#         "file": d["file"],
#         "peaks_ref": d["peaks_ref"],
#         "peaks_detected": d["peaks_detected"],
#     }
#     for d in results_combined_b
# ])

# mask = df_pogladowe_b_pp["peaks_detected"].apply(
#     lambda d: any(len(d[p]) > 1 for p in ["P1", "P2", "P3"])
# )

# rows_multi_b = df_pogladowe_b_pp[mask]
# print(rows_multi_b)

# count_b = df_pogladowe_b["peaks_detected"].apply(
#     lambda d: len(d["P1"]) == 0 or len(d["P2"]) == 0 or len(d["P3"]) == 0
# ).sum()

# empty_lists_count_b = df_pogladowe_b["peaks_detected"].apply(
#     lambda d: (len(d["P1"]) == 0) + (len(d["P2"]) == 0) + (len(d["P3"]) == 0)
# ).sum()

# print(count_b, empty_lists_count_b)

# plot_files_in_class(results_combined_b, "Class1")
# plot_files_in_class(results_combined_b, "Class2")
# plot_files_in_class(results_combined_b, "Class3")
# plot_files_in_class(results_combined_b, "Class4")

# all_metrics = []

# for class_name in ["Class1","Class2","Class3","Class4"]:
#         df_metrics = compute_peak_metrics_all_peaks(results_a, class_name)
#         # dodajemy info o metodzie
#         all_metrics.append(df_metrics) 
        
# # for class_name in ["Class4"]:
# #     for peak_name in ["P2"]:
# #         df_metrics = compute_peak_metrics_fixed(results_a, df_variant_a, peak_name, class_name)
# #         # dodajemy info o metodzie
# #         all_metrics.append(df_metrics) 

# # for class_name in ["Class1","Class2","Class3","Class4"]:
# #     df_metrics = compute_peak_metrics_all_peaks_combined(results_a, class_name)
# #     all_metrics.append(df_metrics)
# df_all_metrics = pd.concat(all_metrics, ignore_index=True)
# df_avg_metrics = df_all_metrics.groupby(["Class"]).mean(numeric_only=True).reset_index()

def compute_peak_metrics_fixed(detection_results, settings, peak_name, class_name):
#     metrics_list = []
#     # bierzemy tylko sygnały dla danej klasy
#     class_signals = [item for item in detection_results if (
#         item["class"] == class_name 
#         and item["peak"] == peak_name
#         and item["method"] in settings["method"].values)]
    
    
    
#     for item in class_signals:
#         file_name = item["file"]
#         signal_df = item["signal"]
#         method_name = item.get("method", None)
#         t = signal_df.iloc[:, 0].values
#         y = signal_df.iloc[:, 1].values

#         ref_idx = item["peaks_ref"].get(peak_name)
#         detected = item["peaks_detected"].get(peak_name, [])

#         if ref_idx is None or len(detected) == 0:
#             metrics_list.append({
#                 "Class": class_name,
#                 "Peak": peak_name,
#                 "File": file_name,
#                 "Method": method_name,
#                 "Mean_X_Error": np.nan,
#                 "Mean_Y_Error": np.nan,
#                 "Mean_XY_Error": np.nan,
#                 "Min_XY_Error": np.nan,
#                 "Peak_Count": 0,
#                 "Reference_Peaks": ref_idx,
#                 "Detected_Peaks": list(detected)
#             })
#             continue

#         t_detected = t[detected]
#         y_detected = y[detected]

#         # jeśli wykryto wiele pików P1, bierzemy średnie błędy względem wszystkich referencji
#         if np.isscalar(ref_idx):
#             dx = abs(t_detected - t[ref_idx])
#             dy = abs(y_detected - y[ref_idx])
#         else:  # ref_idx może być listą wielu indeksów
#             dx = np.min([abs(t_detected - t[r]) for r in ref_idx], axis=0)
#             dy = np.min([abs(y_detected - y[r]) for r in ref_idx], axis=0)

#         dxy = np.sqrt(dx**2 + dy**2)

#         metrics_list.append({
#             "Class": class_name,
#             "Peak": peak_name,
#             "File": file_name,
#             "Method": method_name,
#             "Mean_X_Error": np.mean(dx),
#             "Mean_Y_Error": np.mean(dy),
#             "Mean_XY_Error": np.mean(dxy),
#             "Min_XY_Error": np.min(dxy),
#             "Peak_Count": len(detected),
#             "Reference_Peaks": ref_idx,
#             "Detected_Peaks": list(detected)
#         })

#     df_metrics = pd.DataFrame(metrics_list)

#     # liczba sygnałów w klasie liczy się teraz względem sygnałów, które **mogą mieć dany pik**
#     num_signals_in_class = len(class_signals)
#     num_detected = df_metrics["Peak_Count"].gt(0).sum()
#     df_metrics["Num_Signals_in_Class"] = num_signals_in_class
#     df_metrics["Num_Signals_with_Peak"] = num_detected

#     return df_metrics




# # --- wariant a ---
# df_variant_a = pd.DataFrame([
#     # -------- Class1 --------
#     ("Class1", "P1", "it1",            "it1",            "avg",  "concave"),  
#     ("Class1", "P2", "it1_smooth_4Hz", "it1_smooth_4Hz", "avg",  "curvature"),
#     ("Class1", "P3", "it1_smooth_4Hz", "it1_smooth_4Hz", "full", "concave"),

#     # -------- Class2 --------
#     ("Class2", "P1", "it1_smooth_3Hz", "it1_smooth_3Hz", "avg",  "hilbert"),
#     ("Class2", "P2", "it1_smooth_4Hz", "it1_smooth_4Hz", "whiskers", "line_distance_10"),
#     ("Class2", "P3", "it1",            "it1",            "full",      "hilbert"),

#     # -------- Class3 --------
#     ("Class3", "P1", "it1_smooth_4Hz", "it1_smooth_4Hz", "full", "wavelet"),
#     ("Class3", "P2", "it1_smooth_4Hz", "it1_smooth_4Hz", "avg", "hilbert"),
#     ("Class3", "P3", "it1",            "it1",            "full", "modified_scholkmann_1-2_99"),

#     # -------- Class4 --------
#     ("Class4", "P2", "it1", "it1", "full", "concave_d2x=0-002"),
# ],
# columns=[
#     "class",
#     "peak",
#     "detect_dataset",
#     "ranges_dataset",
#     "range_type",
#     "method"
# ])

# # --- wariant b ---
# df_variant_b = pd.DataFrame([
#     # -------- Class1 --------
#     ("Class1", "P1", "it1",            "it1",            "full", "concave"),
#     ("Class1", "P2", "it1_smooth_4Hz", "it1_smooth_4Hz", "avg",  "curvature"),
#     ("Class1", "P3", "it1_smooth_4Hz", "it1_smooth_4Hz", "full", "concave"),

#     # -------- Class2 --------
#     ("Class2", "P1", "it1_smooth_3Hz", "it1_smooth_3Hz", "avg",  "hilbert"),
#     ("Class2", "P2", "it1_smooth_4Hz", "it1_smooth_4Hz", "full", "line_distance_10"),
#     ("Class2", "P3", "it1",            "it1",            "pm3", "hilbert"),

#     # -------- Class3 --------
#     ("Class3", "P1", "it1_smooth_4Hz", "it1_smooth_4Hz", "full", "wavelet"),
#     ("Class3", "P2", "it1_smooth_4Hz", "it1_smooth_4Hz", "avg",  "wavelet"),
#     ("Class3", "P3", "it1",            "it1",            "none", "modified_scholkmann_1-2_99"),

#     # -------- Class4 --------
#     ("Class4", "P2", "it1", "it1", "pm3", "concave_d2x=0-002"),
# ],
# columns=[
#     "class",
#     "peak",
#     "detect_dataset",
#     "ranges_dataset",
#     "range_type",
#     "method"
# ])




"""
Class1 P1:
    a) avg, concave (min blad) 
    b) full, concave (max. signals w/ peaks)
    
Class1 P2:
    a) smooth 4Hz, avg, curvature (min blad) 
    b) avg, curvature (max signals w/ peaks)   
    ----------- ZMIANA ----------------------
        a,b) smooth 4Hz, avg, curvature 
    
Class1 P3:
    a,b) smooth 4Hz, full, concave   

    
Class2 P1:
    a) smooth 3Hz, avg, hilbert (min blad, 249)
    b) smooth 3Hz, avg, wavelet (podobny, 250) SAME_AVG!!!!!!!!!
    -------- ZMIANA -------------------
        a,b) smooth 3Hz, avg, hilbert
    
Class2 P2:
    a) whiskers, wavelet (min blad)
    b) smooth 4Hz, full, curvature (max sig w/ peaks)
    ----------- ZMIANA -------------
    a) smooth 4Hz, whiskers, line distance (vertical lub perpendicular)
    b) smooth 4Hz, full, line distance (vertical lub perpendicular)
    
Class2 P3:
    a) avg, curvature (min blad)
    b) whiskers, hilbert (doslownie o 2 wiecej sig w/ peaks)
    ----------- ZMIANA -----------
    a) full, hilbert
    b) pm3, hilbert
    

Class3 P1:
    a,b) smooth 4Hz, full, wavelet LUB smooth 4Hz, whiskers, wavelet // (AVG)
    
Class3 P2:
    a) whiskers, concave (min blad)
    b) smooth 4Hz, full, concave (max sig)
    ----------- ZMIANA ------------
    a) smooth 4Hz, avg, hilbert // <----- tylko b?
    b) smooth 4Hz, avg, wavelet

Class3 P3:
    a,b) none, modified scholkmann 1/2 99 LUB none, modified scholkmann 1 99
    ------------ ZMIANA ------------
    a) full modified scholkmann 1/2 99
    b) none modified scholkmann 1/2 99

Class4 P2:
    a,b) smooth 4Hz, none, modified scholkmann 1/2 99 LUB smooth 4Hz, none, modified scholkmann 1 99
    ---------- ZMIANA ------------
    a) full, conncave d2x=0,002
    b) pm3, concave d2x=0,002
    
DALSZE DZIALANIA:
Class1: 
    - jesli sa dwa P3 to wziac tylko pozniejszy i usunac pierwszy (np. [80, 96] -> [96])
    - jesli dwa P2 to pierwszy
    - jesli P2 i P3 sa w odleglosci mniejszej niz 3 probki to usun P2 i P3 (zostaw puste listy)
    - jesli kilka P1 to srednia 
    PO WYKONANIU TYCH OPERCAJI:
    - jesli nie jest spelniony warunek P1<P2<P3 (na osi X!) to usun P1, P2 i P3
    (jesli P1>P2 ale P3 jest najpozniejsze to zostaw P3)
    
Class2:
    - jesli jest kilka P3 to srednia
    - jesli jest kilka P2 to wziac pierwszy
    - jesli sa dwa/kilka P1 to wziac pierwszy
    PO WYKONANIU TYCH OPERCAJI:
    - jesli nie jest spelniony warunek P1<P2<P3 (na osi X!) to usun P1, P2 i P3
    (jesli P1>P2 ale P3 jest najpozniejsze to zostaw P3)
    
Class3:
    - jesli jest kilka P3 to srednia
    - jesli jest kilka P1 to srednia
    - jesli jest kilka P2 to wziac pierwszy
    - jesli P2 i P3 sa w odleglosci mniejszej niz 3 probki to usun P3
    
Class4:
    - jesli jest kilka P2 to srednia
    
"""


# %% ================= DRUGI ZESTAW (IT2) ====================
# results_a_it2 = run_variant(
#     df_variant_a,
#     datasets_dict=datasets_dict,
#     ranges_all_time=ranges_all_time,
#     ranges_all_amps=ranges_all_amps)

# results_combined_a_it2 = combine_peaks_by_file(results_a_it2)
# results_combined_a_it2_pp = postprocess_peaks(results_combined_a_it2)

# df_pogladowe_a_it2_pp = pd.DataFrame([
#     {
#         "file": d["file"],
#         "peaks_ref": d["peaks_ref"],
#         "peaks_detected": d["peaks_detected"],
# }
#     for d in results_combined_a_it2_pp])



# results_b_it2 = run_variant(
#     df_variant_b,
#     datasets_dict=datasets_dict,
#     ranges_all_time=ranges_all_time,
#     ranges_all_amps=ranges_all_amps)

# results_combined_b_it2 = combine_peaks_by_file(results_b_it2)
# results_combined_b_it2_pp = postprocess_peaks(results_combined_b_it2)

# df_pogladowe_b_it2_pp = pd.DataFrame([
#     {
#         "file": d["file"],
#         "peaks_ref": d["peaks_ref"],
#         "peaks_detected": d["peaks_detected"],
#     }
#     for d in results_combined_b_it2_pp])



# # wariant a
# df_metrics_a_pre = pd.concat([compute_metrics_pre_postproc(results_a_it2, cls) for cls in classes], ignore_index=True)
# df_metrics_a_post = pd.concat([compute_metrics_postproc(results_combined_a_it2_pp, cls) for cls in classes], ignore_index=True)

# # wariant b
# df_metrics_b_pre = pd.concat([compute_metrics_pre_postproc(results_b_it2, cls) for cls in classes], ignore_index=True)
# df_metrics_b_post = pd.concat([compute_metrics_postproc(results_combined_b_it2_pp, cls) for cls in classes], ignore_index=True)

# df_metrics_a_pre.to_csv("metrics_a_pre.csv", index=False)
# df_metrics_a_post.to_csv("metrics_a_post.csv", index=False)
# df_metrics_b_pre.to_csv("metrics_b_pre.csv", index=False)
# df_metrics_b_post.to_csv("metrics_b_post.csv", index=False)

# plot_files_in_class(results_combined_a_it2_pp, "Class1")
# plot_files_in_class(results_combined_a_it2_pp, "Class2")
# plot_files_in_class(results_combined_a_it2_pp, "Class3")
# plot_files_in_class(results_combined_a_it2_pp, "Class4")

# plot_files_in_class(results_combined_b_it2_pp, "Class1")
# plot_files_in_class(results_combined_b_it2_pp, "Class2")
# plot_files_in_class(results_combined_b_it2_pp, "Class3")
# plot_files_in_class(results_combined_b_it2_pp, "Class4")

# classes = ["Class1", "Class2", "Class3",]
# for class_id in classes:
#     plot_upset_classic_postproc(results_combined_a_it2_pp, class_id)
#     plot_upset_classic_postproc(results_combined_b_it2_pp, class_id)

# df_pogladowe_a_it2_pp['Class'] = df_pogladowe_a_it2_pp['file'].str.split('_').str[0]




# =========== WILCOXON ====================
# ============================================

expected_peaks = ["P1","P2","P3"]
# wilcoxon_results = run_wilcoxon_pipeline(results_combined_new_pp, results_combined_new_s_pp, expected_peaks)
wilcoxon_table = run_wilcoxon_pipeline(results_combined_new_pp, results_combined_new_s_pp)

wilcoxon_table.to_csv(
    "wilcoxon_results_summary.csv",
    index=False,
)

print(wilcoxon_table)

# 2. Obliczenie wyniku GLOBALNEGO (dla całej pracy)
# Łączymy wszystkie pary dxy z wyników szczegółowych w jeden wektor
df_new = compute_dxy_per_peak(results_combined_new_pp, expected_peaks)
df_simpl = compute_dxy_per_peak(results_combined_new_s_pp, expected_peaks)
merged_global = merge_dxy_for_wilcoxon(df_new, df_simpl)

# Wykonujemy test na całości danych

stat_glob, p_glob = wilcoxon(merged_global["dxy_new"], 
                             merged_global["dxy_new_simpl"], 
                             zero_method="pratt")

print(f"\n--- WYNIK GLOBALNY ---")
print(f"Suma par do testu: {len(merged_global)}")
print(f"p-value globalne: {p_glob:.20f}")

# Test jednostronny: czy błąd 'new' jest ISTOTNIE MNIEJSZY niż 'simpl'?
stat, p_less = wilcoxon(merged_global["dxy_new"], 
                        merged_global["dxy_new_simpl"], 
                        alternative='less', 
                        zero_method="pratt")

if p_less < 0.05:
    print("Algorytm dopasowany ma istotnie mniejsze błędy (ma przewagę).")
else: print("Łojojo") 

tp_total_new = df_metrics_post['TP'].sum()
tp_total_simpl = df_metrics_post_s['TP'].sum()

summary = {
    "Dopasowany": {
        "TP_total": tp_total_new, 
        "Mean_Err": df_new['dxy'].mean()
    },
    "Uproszczony": {
        "TP_total": tp_total_simpl, 
        "Mean_Err": df_simpl['dxy'].mean()
    }
}

print(f"Suma TP (Dopasowany): {tp_total_new}")
print(f"Suma TP (Uproszczony): {tp_total_simpl}")

diff = merged_global["dxy_new"] - merged_global["dxy_new_simpl"]
print(diff.median(), diff.mean())


merged_before = pd.merge(
    df_new,
    df_simpl,
    on=["file","class","peak"],
    suffixes=("_new","_new_simpl")
)

merged_after = merged_before.dropna(
    subset=["dxy_new","dxy_new_simpl"]
)

print(len(merged_before))
print(len(merged_after))


diff_global = merged_global["dxy_new"] - merged_global["dxy_new_simpl"]

# Mediana
mediana = diff_global.median()

# Średnia
srednia = diff_global.mean()

# Kwartyle
q1 = diff_global.quantile(0.25)
q3 = diff_global.quantile(0.75)

print(f"GLOBALNY TEST:")
print(f"Q1: {q1:.3f}")
print(f"Mediana: {mediana:.3f}")
print(f"Q3: {q3:.3f}")
print(f"Średnia: {srednia:.3f}")

# =====================================================




# mask = (
#     (df_pogladowe_a_it2_pp["Class"] == "Class3") &
#     df_pogladowe_a_it2_pp["peaks_detected"].apply(
#         lambda d:
#             not (isinstance(d.get("P1"), float) and math.isnan(d.get("P1"))) and
#             not (isinstance(d.get("P2"), float) and math.isnan(d.get("P2"))) and
#             not (isinstance(d.get("P3"), float) and math.isnan(d.get("P3")))
#     )
# )

# df_all_peaks_class3 = df_pogladowe_a_it2_pp[mask]

# print(df_all_peaks_class3[["file", "peaks_detected"]])
# print(f"\nLiczba sygnałów Class3 z wykrytymi P1, P2 i P3: {len(df_all_peaks_class3)}")

# mask_complement = (
#     (df_pogladowe_a_it2_pp["Class"] == "Class3") &
#     df_pogladowe_a_it2_pp["peaks_detected"].apply(
#         lambda d:
#             (isinstance(d.get("P1"), float) and math.isnan(d.get("P1"))) or
#             (isinstance(d.get("P2"), float) and math.isnan(d.get("P2"))) or
#             (isinstance(d.get("P3"), float) and math.isnan(d.get("P3")))
#     )
# )

# df_not_all_peaks_class3 = df_pogladowe_a_it2_pp[mask_complement]

# print(df_not_all_peaks_class3[["file", "peaks_detected"]])
# print(f"\nLiczba sygnałów Class3 BEZ kompletu P1–P2–P3: {len(df_not_all_peaks_class3)}")
# mask = (df_pogladowe_a_it2_pp["Class"] == "Class3") & df_pogladowe_a_it2_pp["peaks_detected"].apply(
#     lambda d: isinstance(d.get("P1"), float) and math.isnan(d.get("P1"))
# )

# df_missing_p1 = df_pogladowe_a_it2_pp[mask]

# # Wyświetlamy
# print(df_missing_p1)

# plot_peak_detection_pie(results_combined_a_it2_pp, "Class4", peak="P2")
# plot_peak_detection_pie(results_combined_b_it2_pp, "Class4", peak="P2")


# mask = df_pogladowe_a_pp["peaks_detected"].apply(
#     lambda d: any(len(d[p]) > 1 for p in ["P2"])
# )
# rows_multi_a = df_pogladowe_a_pp[mask]
# print(rows_multi_a)
# mask = df_pogladowe_a_pp["peaks_detected"].apply(
#     lambda d: any(len(d[p]) > 1 for p in ["P1", "P2", "P3"])
# )
# rows_multi_a = df_pogladowe_a_pp[mask]
# print(rows_multi_a)

# count_a = df_pogladowe_a["peaks_detected"].apply(
#     lambda d: len(d["P1"]) == 0 or len(d["P2"]) == 0 or len(d["P3"]) == 0
# ).sum()

# empty_lists_count_a = df_pogladowe_a["peaks_detected"].apply(
#     lambda d: (len(d["P1"]) == 0) + (len(d["P2"]) == 0) + (len(d["P3"]) == 0)
# ).sum()

# print(count_a, empty_lists_count_a)




# =========== WILCOXON ====================
# ============================================

expected_peaks = ["P1","P2","P3"]
# wilcoxon_results = run_wilcoxon_pipeline(results_combined_new_pp, results_combined_new_s_pp, expected_peaks)
wilcoxon_table = run_wilcoxon_pipeline(results_combined_new_pp, results_combined_new_s_pp)

wilcoxon_table.to_csv(
    "wilcoxon_results_summary.csv",
    index=False,
)

print(wilcoxon_table)

# 2. Obliczenie wyniku GLOBALNEGO (dla całej pracy)
# Łączymy wszystkie pary dxy z wyników szczegółowych w jeden wektor
df_new = compute_dxy_per_peak(results_combined_new_pp, expected_peaks)
df_simpl = compute_dxy_per_peak(results_combined_new_s_pp, expected_peaks)
merged_global = merge_dxy_for_wilcoxon(df_new, df_simpl)

# Wykonujemy test na całości danych

stat_glob, p_glob = wilcoxon(merged_global["dxy_new"], 
                             merged_global["dxy_new_simpl"], 
                             zero_method="pratt")

print(f"\n--- WYNIK GLOBALNY ---")
print(f"Suma par do testu: {len(merged_global)}")
print(f"p-value globalne: {p_glob:.20f}")

# Test jednostronny: czy błąd 'new' jest ISTOTNIE MNIEJSZY niż 'simpl'?
stat, p_less = wilcoxon(merged_global["dxy_new"], 
                        merged_global["dxy_new_simpl"], 
                        alternative='less', 
                        zero_method="pratt")

if p_less < 0.05:
    print("Algorytm dopasowany ma istotnie mniejsze błędy (ma przewagę).")
else: print("Łojojo") 

tp_total_new = df_metrics_post['TP'].sum()
tp_total_simpl = df_metrics_post_s['TP'].sum()

summary = {
    "Dopasowany": {
        "TP_total": tp_total_new, 
        "Mean_Err": df_new['dxy'].mean()
    },
    "Uproszczony": {
        "TP_total": tp_total_simpl, 
        "Mean_Err": df_simpl['dxy'].mean()
    }
}

print(f"Suma TP (Dopasowany): {tp_total_new}")
print(f"Suma TP (Uproszczony): {tp_total_simpl}")

diff = merged_global["dxy_new"] - merged_global["dxy_new_simpl"]
print(diff.median(), diff.mean())


merged_before = pd.merge(
    df_new,
    df_simpl,
    on=["file","class","peak"],
    suffixes=("_new","_new_simpl")
)

merged_after = merged_before.dropna(
    subset=["dxy_new","dxy_new_simpl"]
)

print(len(merged_before))
print(len(merged_after))


diff_global = merged_global["dxy_new"] - merged_global["dxy_new_simpl"]

# Mediana
mediana = diff_global.median()

# Średnia
srednia = diff_global.mean()

# Kwartyle
q1 = diff_global.quantile(0.25)
q3 = diff_global.quantile(0.75)

print(f"GLOBALNY TEST:")
print(f"Q1: {q1:.3f}")
print(f"Mediana: {mediana:.3f}")
print(f"Q3: {q3:.3f}")
print(f"Średnia: {srednia:.3f}")

# =====================================================
# =========================
# 1. Funkcja do obliczania błędów XY dla pojedynczego piku
# =========================
def compute_dxy_per_peak(results_combined, expected_peaks=("P1","P2","P3")):
    """
    Zwraca DataFrame z błędami XY dla każdego pliku i piku.
    
    results_combined: lista słowników po postprocess
    expected_peaks: tuple oczekiwanych pików
    """
    records = []
    for item in results_combined:
        file_name = item["file"]
        class_id = item.get("class", "Unknown")
        sig = item["signal_raw"]
        t = sig.iloc[:,0].values
        y = sig.iloc[:,1].values
        for peak in expected_peaks:
            ref_idx = item["peaks_ref"].get(peak)
            detected_val = item["peaks_detected"].get(peak)
            
            # obsługa braków
            if ref_idx is None or detected_val is None or (isinstance(detected_val,float) and np.isnan(detected_val)):
                dxy = np.nan
            else:
                dx = abs(t[detected_val] - t[ref_idx])
                dy = abs(y[detected_val] - y[ref_idx])
                dxy = np.sqrt(dx**2 + dy**2)
            
            records.append({
                "file": file_name,
                "class": class_id,
                "peak": peak,
                "dxy": dxy
            })
    return pd.DataFrame(records)

# =========================
# 2. Funkcja do łączenia dwóch wariantów w pary
# =========================
def merge_dxy_for_wilcoxon(df1, df2):
    # Dodajemy "class" do listy kluczy (on=)
    merged = pd.merge(df1, df2, on=["file", "class", "peak"], suffixes=("_new","_new_simpl"))
    merged = merged.dropna(subset=["dxy_new","dxy_new_simpl"])
    return merged

# =========================
# 3. Funkcja do wykonania testu Wilcoxona dla wszystkich pików
# =========================
def wilcoxon_test_all_peaks(merged_df):
    """
    merged_df: DataFrame z kolumnami ['file','peak','dxy_new','dxy_new_simpl']
    Zwraca DataFrame z wynikami testu dla każdego piku.
    """
    results = []
    for peak in merged_df["peak"].unique():
        sub = merged_df[merged_df["peak"]==peak]
        if len(sub) < 5:  # zbyt mało danych
            p_value = np.nan
            stat = np.nan
        else:
            stat, p_value = wilcoxon(sub["dxy_new"], sub["dxy_new_simpl"])
        results.append({
            "peak": peak,
            "n_pairs": len(sub),
            "wilcoxon_stat": stat,
            "p_value": p_value
        })
    return pd.DataFrame(results)
    
def wilcoxon_test_by_class_and_peak(merged_df):
    """
    merged_df: DataFrame z kolumnami ['file', 'class', 'peak', 'dxy_new', 'dxy_new_simpl']
    """
    results = []
    # Grupowanie po dwóch kolumnach: klasie i piku
    for (class_id, peak), sub in merged_df.groupby(["class", "peak"]):
        
        # Obliczamy różnice
        diff = sub["dxy_new"] - sub["dxy_new_simpl"]
        
        # Sprawdzamy, czy wszystkie różnice są zerem (identyczne wyniki)
        if (diff == 0).all():
            p_value = 1.0  # identyczne rozkłady
            stat = 0.0
        if len(sub) < 5:
            p_value = np.nan
            stat = np.nan
        else:
            # zero_method="pratt" pomaga, gdy jest dużo identycznych par
            try:
                stat, p_value = wilcoxon(sub["dxy_new"], sub["dxy_new_simpl"], zero_method="pratt")
            except:
                stat, p_value = np.nan, np.nan

        results.append({
            "class": class_id,
            "peak": peak,
            "n_pairs": len(sub),
            "wilcoxon_stat": stat,
            "p_value": p_value
        })
    return pd.DataFrame(results)

# =========================
# 4. Kompletny pipeline
# =========================
# def run_wilcoxon_pipeline(results_new_pp, results_new_s_pp, expected_peaks=("P1","P2","P3")):
#     df_new = compute_dxy_per_peak(results_new_pp, expected_peaks)
#     df_simpl = compute_dxy_per_peak(results_new_s_pp, expected_peaks)
#     merged = merge_dxy_for_wilcoxon(df_new, df_simpl)
#     wilcoxon_results = wilcoxon_test_by_class_and_peak(merged)
#     return wilcoxon_results
# #, df_new, df_simpl

def run_wilcoxon_pipeline(results_new_pp, results_new_s_pp, expected_peaks=("P1","P2","P3")):
    df_new = compute_dxy_per_peak(results_new_pp, expected_peaks)
    df_simpl = compute_dxy_per_peak(results_new_s_pp, expected_peaks)

    merged = merge_dxy_for_wilcoxon(df_new, df_simpl)

    # 1️⃣ mediany + kwartyle (NOWE)
    summary_stats = summarize_dxy_by_class_peak(merged)

    # 2️⃣ test Wilcoxona
    wilcoxon_results = wilcoxon_test_by_class_and_peak(merged)

    # 3️⃣ scal wszystko w jedną tabelę (KLUCZOWE)
    final_table = pd.merge(
        summary_stats,
        wilcoxon_results,
        on=["class", "peak", "n_pairs"],
        how="left"
    )

    return final_table



def prepare_metrics_df(df, version_name):
    cols = [
        "Class",
        "Peak",
        "Mean_X_Error",
        "Mean_Y_Error",
        "Mean_XY_Error",
        "TP",
        "FP",
        "FN",
    ]
    
    out = df[cols].copy()
    out["Version"] = version_name
    return out


def summarize_dxy_by_class_peak(merged_df):
    summary = []
    for (class_id, peak), sub in merged_df.groupby(["class", "peak"]):
        summary.append({
            "class": class_id,
            "peak": peak,
            "n_pairs": len(sub),
            "median_new": sub["dxy_new"].median(),
            "q1_new": sub["dxy_new"].quantile(0.25),
            "q3_new": sub["dxy_new"].quantile(0.75),
            "median_simpl": sub["dxy_new_simpl"].median(),
            "q1_simpl": sub["dxy_new_simpl"].quantile(0.25),
            "q3_simpl": sub["dxy_new_simpl"].quantile(0.75),
            "median_diff": (sub["dxy_new"] - sub["dxy_new_simpl"]).median()
        })
    return pd.DataFrame(summary)