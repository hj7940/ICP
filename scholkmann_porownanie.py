# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 22:05:02 2025

@author: User
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_handling import load_dataset, smooth_dataset
from methods import (concave, curvature, modified_scholkmann_old,
                     modified_scholkmann, 
                     line_distance, hilbert_envelope, wavelet)
from ranges import (ranges_full, ranges_pm3, ranges_whiskers, 
                    generate_ranges_for_all_files, compute_ranges_avg)
from main import peak_detection
from main import (all_methods, it2, it2_smooth_4Hz, it2_smooth_3Hz,
                  it1, it1_smooth_4Hz, it1_smooth_3Hz,
                  ranges_all_time, ranges_all_amps, compute_peak_metrics,
                  df_ranges_time, df_ranges_amps)

