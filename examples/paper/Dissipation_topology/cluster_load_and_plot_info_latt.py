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
L = 51
diss_strength = 0.0
J = -1.0
min_l = 5
max_l= 6
init = 'W'


################### Data paths ##################
script_dir = os.path.dirname(os.path.abspath(__file__))
data_filepath = os.path.join(script_dir, "results/cluster_results",)

################### Load data ##################

checkpoint_folder = os.path.join(data_filepath, f"xx_diss={diss_strength}_J={J}_L={L}_lmin={min_l}_lmax={max_l}_init={init}_mixed")

info_latt, loaded_times = load_info_data(
    info_latt_path=os.path.join(checkpoint_folder, "info_lattice.pkl"),
    times_lite_path=os.path.join(checkpoint_folder, "times.pkl"),
)


#h5_path = os.path.join(
#    data_filepath,
#    f"opdm_t_corr_diss={diss_strength:.1f}_J={J:g}_L={L}_init=mixed_{init}.h5",
#)

#with tables.open_file(h5_path, mode="r") as h5:
#    opdm_t_corr = np.asarray(h5.root.opdm_t_corr.read())
#    times_corr = np.asarray(h5.root.times_corr.read(), dtype=float)
#
#info_latt_corr = [
#    calc_info_from_corr_matr(opdm)
#    for opdm in opdm_t_corr
#]


################### Plot ##################

#plot_info_per_scale_3d(
#    info_latt_corr,
#    times_corr,
#    t_min=None,
#    t_max=None,
#    fix_l=None,
#    elev=40,
#    azim=-60,
#)

plot_info_per_scale_3d(
    info_latt,
    loaded_times,
    t_min=0,
    t_max=4,
    fix_l=None,
    elev=40,
    azim=-60,
)

