import numpy as np
from copy import deepcopy

swc_dtype = np.dtype([('id', "int32"), ('type', "int16"),
                      ('x', "float32"), ('y', "float32"), ('z', "float32"),
                      ('radius', "float32"), ('parent', "int32")])


def correct_swc(input_swc_file, output_swc_file, smooth_samples=10,
                min_radius_value=0.11, interp=True, smooth=True,
                soma_radius=None, reset_tags=[[4, 3], [7, 4]], save=True):
    """
    Correct an SWC morphology by interpolating missing diameters (below min_diam_value) and smoothing the
    diameters with a moving average (smooth_samples).

    Parameters
    ----------
    input_swc_file: str or Path
        The input swc file
    output_swc_file: str or Path
        The output swc file
    smooth_samples: int
        Number of points for smoothing
    min_radius_value: float
        Minimum diameter in um to consider it "missing" and to interpolate
    interp: bool
        If True (default), missing diameters are interpolated
    smooth: bool
        If True (default), diameters are smoothed
    soma_radius: float or None
        If given, the somatic radius is reset
    reset_tags: list of lists
        Resets tags. E.g. [[4, 3], [7, 4]] means that first all "4" tags are reset to "3", then all "7" tags are
        reset to "4".
    save: bool
        If True, the output swc is saved to the output_swc_file

    """
    swc_data = np.loadtxt(input_swc_file, dtype=swc_dtype)
    initial_branches = np.where(np.diff(swc_data["parent"]) > 1)[0]
    initial_branches[0] += 1

    old_radii = swc_data["radius"]
    new_radii = deepcopy(swc_data["radius"])
    missing_idxs = np.array([], dtype="int")
    corrected_idxs = np.array([], dtype="int")
    n_missing_values = 0

    if not interp:
        min_radius_idxs = np.where(old_radii < min_radius_value)[0]
        new_radii[min_radius_idxs] = min_radius_value

    for i_br, init in enumerate(initial_branches):
        if i_br < len(initial_branches) - 1:
            idxs = range(init, initial_branches[i_br + 1])
        else:
            idxs = range(init, len(swc_data))

        idxs = np.array(idxs)

        if interp:
            # interpolate min values
            min_radius_idxs = np.where(old_radii[idxs] < min_radius_value)
            if len(min_radius_idxs[0]) > 0:
                min_radius_idxs = min_radius_idxs[0]
                n_missing_values += len(min_radius_idxs)
                # print(f"Path {i_br + 1} / {len(initial_branches)} has {len(min_radius_idxs)} missing values")
                missing = idxs[min_radius_idxs]
                missing_idxs = np.concatenate((missing_idxs, missing))

                diff_greater_0 = np.where(np.diff(missing) > 1)[0]

                if len(diff_greater_0) > 0:
                    cont_intervals = []
                    for i, d in enumerate(diff_greater_0 + 1):
                        if i == 0:
                            cont_intervals.append(missing[:d])
                        else:
                            cont_intervals.append(missing[diff_greater_0[i - 1] + 1:d])
                    cont_intervals.append(missing[d:])
                else:
                    cont_intervals = [[missing[0], missing[-1]]]
                corrected = np.array([], dtype="int")
                for cont_int in cont_intervals:
                    # linear interp
                    start_idx = cont_int[0]
                    end_idx = cont_int[-1]

                    if end_idx < len(old_radii) - 1:
                        m = (old_radii[end_idx + 1] - old_radii[start_idx - 1]) / ((end_idx - start_idx) + 2)
                        new_radii_interp = m * np.arange(start_idx, end_idx + 1) - m * (start_idx - 1) + \
                                           old_radii[start_idx - 1]
                        new_radii[start_idx:end_idx + 1] = new_radii_interp
                        corrected = np.concatenate((corrected, list(range(start_idx, end_idx + 1))))
                        corrected_idxs = np.concatenate((corrected_idxs, list(range(start_idx, end_idx + 1))))
                    else:
                        m = (old_radii[end_idx] - old_radii[start_idx - 1]) / ((end_idx - start_idx) + 1)
                        new_radii_interp = m * np.arange(start_idx, end_idx) - m * (start_idx - 1) + \
                                           old_radii[start_idx - 1]
                        new_radii[start_idx:end_idx] = new_radii_interp
                        corrected = np.concatenate((corrected, list(range(start_idx, end_idx))))
                        corrected_idxs = np.concatenate((corrected_idxs, list(range(start_idx, end_idx))))

                # if len(corrected) < len(missing):
                #     print(f"Path {i_br} missing values: {len(missing) - len(corrected)}")


        if smooth:
            # smooth
            if len(idxs) > smooth_samples:
                new_radii[idxs] = np.convolve(new_radii[idxs], np.ones(smooth_samples) / 
                                              smooth_samples, mode='same')
            else:
                print(f"Skipped path {i_br} because it has {len(idxs)} points")

    if interp:
        print(f"Interpolated diameters of {n_missing_values} out of {len(swc_data)} points")

    min_radius_idxs = np.where(new_radii < min_radius_value)[0]
    new_radii[min_radius_idxs] = min_radius_value

    swc_corrected = deepcopy(swc_data)
    swc_corrected["radius"] = new_radii
    if soma_radius is not None:
        soma_idx = np.where(swc_data["type"] == 1)[0]
        swc_corrected["radius"][soma_idx] = soma_radius

    if reset_tags is not None:
        for tags in reset_tags:
            assert len(tags) == 2, "Each entry of 'reset_tags' should have 2 values."
            swc_corrected["type"][swc_corrected["type"] == tags[0]] = tags[1]
            print(f"Reset tag {tags[0]} to {tags[1]}")

    if save:
        print(f"Saving corrected morphology to {output_swc_file}")
        np.savetxt(output_swc_file, swc_corrected)

    return swc_corrected
