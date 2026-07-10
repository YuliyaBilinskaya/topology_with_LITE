import numpy as np
import tables as tb
from marker_functions import calc_marker_state_from_opdm


def save_markers_from_opdm_h5(
    opdm_file,
    marker_file=None,
    flatten=False,
    compression_level=5,
):
    """
    Read an HDF5 file containing:
      - /opdm_t_corr : shape (n_times, 2L, 2L)
      - /times_corr  : shape (n_times,)
    and write a new HDF5 file containing:
      - /markers_t   : shape (n_times, L)
      - /times_corr  : shape (n_times,)

    Parameters
    ----------
    opdm_file : str
        Input HDF5 file, e.g. 'opdm_t_corr_diss=0.2_J=-1_L=50.h5'
    marker_file : str or None
        Output HDF5 file. If None, derives name from opdm_file.
    flatten : bool
        Passed through to calc_marker_state_from_opdm.
    compression_level : int
        Compression level for the output file.
    """
    if marker_file is None:
        if opdm_file.endswith(".h5"):
            marker_file = opdm_file[:-3] + "_markers.h5"
        else:
            marker_file = opdm_file + "_markers.h5"

    filters = tb.Filters(complevel=compression_level, complib="blosc:zstd")

    with tb.open_file(opdm_file, mode="r") as h5_in:
        opdm_store = h5_in.root.opdm_t_corr
        times_corr = h5_in.root.times_corr[:]

        n_times = opdm_store.shape[0]
        two_L = opdm_store.shape[1]
        L = two_L // 2

        with tb.open_file(marker_file, mode="w") as h5_out:
            marker_store = h5_out.create_earray(
                h5_out.root,
                "markers_t",
                atom=tb.Float32Atom(),
                shape=(0, L),
                filters=filters,
                expectedrows=n_times,
            )

            for time_idx in range(n_times):
                opdm = opdm_store[time_idx]

                marker = calc_marker_state_from_opdm(
                    opdm,
                    flatten=flatten,
                )

                # Handle either:
                #   marker = array(...)
                # or
                #   marker, eigvals = ...
                if isinstance(marker, tuple):
                    marker = marker[0]

                marker = np.asarray(marker, dtype=np.float32)
                marker_store.append(marker[None, :])

            h5_out.create_array(h5_out.root, "times_corr", times_corr.astype(np.float32))
