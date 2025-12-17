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