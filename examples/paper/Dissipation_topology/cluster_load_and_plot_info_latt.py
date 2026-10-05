import os
import re
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from matplotlib.colors import LinearSegmentedColormap, Normalize
from marker_functions import *
import tables
import numpy as np
from cluster_plotting_functions import *
from infoLattice_functions import *

################### Parameters ##################
L = 50
diss_strength = 0.05
J = -1.0
mu = -0.3
min_l = 5
max_l= 6
init = 'triv'


################### Data paths ##################
script_dir = os.path.dirname(os.path.abspath(__file__))
data_filepath = os.path.join(script_dir, "results/cluster_results",)

################### Load data ##################

#-------------------LITE------------------------

checkpoint_folder = os.path.join(data_filepath, f"xx_diss={diss_strength}_J={J}_h={mu}_L={L}_lmin={min_l}_lmax={max_l}_init={init}_mixed")

info_latt, loaded_times = load_info_data(
    info_latt_path=os.path.join(checkpoint_folder, "info_lattice.pkl"),
    times_lite_path=os.path.join(checkpoint_folder, "times.pkl"),
)

#print(loaded_times)

# ---- TEMP: export LITE times (delete after one run) ----
#np.save("/Users/yuliyabilinskaya/Library/CloudStorage/OneDrive-KTH/Master_uppsats/code/Yuliya_code_OPEN_Kitaev/loaded_times.npy", np.asarray(loaded_times, dtype=float))
#print(f"Saved {len(loaded_times)} LITE times")
# ---- END TEMP ----

info_latt_bits = []

for lattice_at_time in info_latt:
    converted_lattice = lattice_at_time.deepcopy()

    for key, value in lattice_at_time.items():
        converted_lattice[key] = value / np.log(2) # LITE calculates entropy in natural log

    info_latt_bits.append(converted_lattice)

#-------------------OPDM------------------------

h5_path = os.path.join(
    data_filepath,
    f"opdm_t_corr_diss={diss_strength}_J={J:g}_mu={mu}_L={L}_init=mixed_{init}.h5",
)

with tables.open_file(h5_path, mode="r") as h5:
    #opdm_t_corr = np.asarray(h5.root.opdm_t_corr.read())
    info_latt_t = np.asarray(h5.root.info_latt_t.read(), dtype=float)
    times_corr = np.asarray(h5.root.times_corr.read(), dtype=float)
#print(times_corr)

info_latt_corr = [
    {l + 1: np.array([val]) for l, val in enumerate(row)}
    for row in info_latt_t
]

#info_latt_corr = [
#    calc_info_from_corr_matr(opdm)
#    for opdm in opdm_t_corr
#]


#################### Plot ##################

#plot_info_per_scale_3d(
#    info_latt_corr,
#    times_corr,
#    t_min=0,
#    t_max=None,
#    fix_l=5,
#    elev=40,
#    azim=-60,
#)

#plot_info_per_scale_3d(
#    info_latt_bits,
#    loaded_times,
#    t_min=0,
#    t_max=None,
#    fix_l=min_l-1,
#    elev=40,
#    azim=-60,
#)

#plot_info_per_scale_diff_3d(
#    info_latt_corr, times_corr,
#    info_latt_bits, loaded_times,
#    fix_l_a=5,       # keep all scales available from corr_mat
#    fix_l_b=5,      # cap at LITE's production max_l
#    t_min=0, t_max=None,
#    elev=40, azim=-60,
#)

#plot_info_per_scale_diff_2d(
#    info_latt_corr, times_corr,
#    info_latt_bits, loaded_times,
#    fix_l_a=5,
#    fix_l_b=5,
#    t_min=0, t_max=3,
#    # scales=[0, 1, 2, 3],  # optional: only some ell values
#)

#plot_info_per_scale_diff_2d_common_time(
#    info_latt_corr, times_corr,
#    info_latt_bits, loaded_times,
#    fix_l_a=5,
#    fix_l_b=5,
#    t_min=0, t_max=2,
#    #scales=[5],  # optional: only some ell values
#)

#plot_total_info_vs_time(
#    info_latt_corr, times_corr,
#    info_latt_bits, loaded_times,
#    fix_l_a=5,
#    fix_l_b=5,
#    t_min=0, t_max=None,
#)
