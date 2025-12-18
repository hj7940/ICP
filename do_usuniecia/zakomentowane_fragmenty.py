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