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