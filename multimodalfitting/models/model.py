import json
from pathlib import Path
import numpy as np

import bluepyopt.ephys as ephys

from .morphology_modifiers import replace_axon_with_hillock_ais, replace_axon_with_ais, fix_morphology_exp


def define_mechanisms(cell_model_folder):
    """Defines mechanisms"""
    
    path_mechs = cell_model_folder / "mechanisms.json"

    assert path_mechs.is_file(), "Couldn't find the 'mechanisms.json' file in the folder!"

    mech_definitions = json.load(open(path_mechs))

    mechanisms = []
    for sectionlist, channels in mech_definitions.items():

        seclist_loc = [ephys.locations.NrnSeclistLocation(
            sectionlist, seclist_name=sectionlist
        )]

        for channel in channels:
            mechanisms.append(
                ephys.mechanisms.NrnMODMechanism(
                    name="%s.%s" % (channel, sectionlist),
                    mod_path="",
                    suffix=channel,
                    locations=seclist_loc,
                    preloaded=True,
                )
            )

    return mechanisms


def define_electrode(
        probe_type="linear",
        num_linear=20,
        linear_span=[-500, 1000],
        z_shift=20,
        probe_center=[0, 300, 20],
        mea_dim=[20, 4],
        mea_pitch=[50, 50],
        probe_file=None
):
    """
    Defines LFPy electrode object

    Parameters
    ----------
    cell: LFPy.Cell
        The LFP Cell associated to the electrode
    probe_type: str
        'linear' or  'planar'
    num_linear: int
        Number of linear electrodes (if 'linear')
    linear_span: 2d array-like
        Minimum and maximum y-values for linear probe
    z_shift: float
        The shift in the z-direction (distance from the cell soma)
    probe_center: 3d array
        The center of the probe
    mea_dim: 2d array
        Dimensions of planar probe (nrows, ncols)
    mea_pitch: 3d arraay
        The pitch of the planar probe (row pitch, column pitch)
    probe_file: str
        The path to a json file representing the probe

    Returns
    -------
    electrode: MEAutility.MEA object
        The MEAutility electrode object
    """

    import MEAutility as mu

    if probe_file is None:
        assert probe_type in ['linear', 'planar']

        if probe_type == 'linear':

            mea_positions = np.zeros((num_linear, 3))
            mea_positions[:, 2] = z_shift
            mea_positions[:, 1] = np.linspace(linear_span[0], linear_span[1],
                                              num_linear)

            mea_info = {
                'pos': list([list(p) for p in mea_positions]),
                'center': False,
                'plane': 'xy'
            }
            probe = mu.return_mea(info=mea_info)

        elif probe_type == 'planar':

            mea_info = {
                'dim': mea_dim,
                'electrode_name': 'hd-mea',
                'pitch': mea_pitch,
                'shape': 'square',
                'size': 5,
                'type': 'mea',
                'plane': 'xy'
            }
            probe = mu.return_mea(info=mea_info)

            # Move the MEA out of the neuron plane (yz)
            probe.move(probe_center)

    else:
        probe_file = Path(probe_file)
        with probe_file.open('r') as f:
            info = json.load(f)

        probe = mu.return_mea(info=info)

    return probe


def define_parameters(cell_model_folder, parameter_file=None, release=False, v_init=None):
    """
    Defines parameters

    Parameters
    ----------
    model_name: str
            "hay", "hay_ais", or "hay_ais_hillock"
    release: bool
        If True, the frozen release parameters are returned. Otherwise, the unfrozen parameters with bounds are
        returned (use False - default - for optimizations)

    Returns
    -------
    parameters: list
        List of BPO parameters
    """

    path_params = cell_model_folder

    if parameter_file is None:
        if release:
            param_configs = json.load(open(path_params / "parameters_release.json"))
        else:
            param_configs = json.load(open(path_params / "parameters.json"))
    else:
        parameter_file = Path(parameter_file)
        assert parameter_file.is_file(), "Parameter file doesn't exist"
        param_configs = json.load(open(parameter_file))
    parameters = []

    for param_config in param_configs:

        if "value" in param_config:
            frozen = True
            value = param_config["value"]
            bounds = None
        elif "bounds" in param_config:
            frozen = False
            bounds = param_config["bounds"]
            value = None
        else:
            raise Exception(
                "Parameter has to have bounds or value: %s" % param_config
            )

        if param_config["type"] == "global":
            if param_config["param_name"] == "v_init" and v_init is not None:
                print(f"Setting v_init to {v_init}")
                value = v_init
            parameters.append(
                ephys.parameters.NrnGlobalParameter(
                    name=param_config["param_name"],
                    param_name=param_config["param_name"],
                    frozen=frozen,
                    bounds=bounds,
                    value=value,
                )
            )

        elif param_config["type"] in ["section", "range", "meta"]:

            if param_config["dist_type"] == "uniform":
                scaler = ephys.parameterscalers.NrnSegmentLinearScaler()

            elif param_config["dist_type"] in ["exp", "step_funct", "user_defined", "sig_increase", "sig_decrease",
                                               "decay"]:

                if "parameters" in param_config:
                    dist_param_names = param_config["parameters"]
                else:
                    dist_param_names = None

                if "soma_ref_point" in param_config:
                    ref_point = param_config["soma_ref_point"]
                    scaler = ephys.parameterscalers.NrnSegmentSomaDistanceScaler(
                        distribution=param_config["dist"],
                        soma_ref_location=ref_point,
                        dist_param_names=dist_param_names
                    )
                elif "ref_section" in param_config:
                    assert "ref_point" in param_config, "'ref_section' missing from param config"
                    ref_point = param_config["ref_point"]
                    ref_section = param_config["ref_section"]
                    scaler = ephys.parameterscalers.NrnSegmentSectionDistanceScaler(
                        distribution=param_config["dist"],
                        ref_section=ref_section,
                        ref_location=ref_point,
                        dist_param_names=dist_param_names
                    )
                else:
                    ref_point = 0.5
                    scaler = ephys.parameterscalers.NrnSegmentSomaDistanceScaler(
                        distribution=param_config["dist"],
                        soma_ref_location=ref_point,
                        dist_param_names=dist_param_names
                    )

            if "sectionlist" not in param_config:  # for meta parameters
                param_config["sectionlist"] = []

            if not isinstance(param_config["sectionlist"], list):
                param_config["sectionlist"] = [param_config["sectionlist"]]

            seclist_loc = []
            for loc in param_config["sectionlist"]:
                seclist_loc.append(ephys.locations.NrnSeclistLocation(
                    loc,
                    seclist_name=loc
                ))

            if len(seclist_loc) > 0:
                str_loc = "_".join(e for e in param_config['sectionlist'])
                name = f"{param_config['param_name']}_{str_loc}"
                param_dependancies = param_config.get("dependencies", None)
            else:
                name = param_config['param_name']
                param_dependancies = param_config.get("dependencies", None)

            if param_config["type"] == "section":
                parameters.append(
                    ephys.parameters.NrnSectionParameter(
                        name=name,
                        param_name=param_config["param_name"],
                        value_scaler=scaler,
                        value=value,
                        frozen=frozen,
                        bounds=bounds,
                        locations=seclist_loc,
                        param_dependancies=param_dependancies
                    )
                )

            elif param_config["type"] == "range":
                parameters.append(
                    ephys.parameters.NrnRangeParameter(
                        name=name,
                        param_name=param_config["param_name"],
                        value_scaler=scaler,
                        value=value,
                        frozen=frozen,
                        bounds=bounds,
                        locations=seclist_loc,
                        param_dependancies=param_dependancies
                    )
                )
            elif param_config["type"] == "meta":
                # print("Adding meta parameter", name, name.split("_")[0])
                parameters.append(
                    ephys.parameters.MetaParameter(
                        name=name,
                        obj=scaler,
                        attr_name=name,
                        frozen=frozen,
                        bounds=bounds,
                        value=value
                    )
                )
        else:
            raise Exception(
                "Param config type has to be global, section, range, or meta: %s"
                % param_config
            )

    return parameters


def define_morphology(cell_model_folder, morph_modifiers, do_replace_axon, **morph_kwargs):
    """
    Defines neuron morphology for the Hay model

    Parameters
    ----------
    model_name: str
            "hay", "hay_ais" or "hay_ais_hillock"
    morph_modifiers: list of python functions
        The modifier functions to apply to the axon
    do_replace_axon: bool
        If True axon is replaced by axon stub
    **morph_kwargs: kwargs for morphology modifiers
    

    Returns
    -------
    morphology: bluepyopt.ephys.morphologies.NrnFileMorphology
        The morphology object
    """
    morphology_files = [p for p in cell_model_folder.iterdir() if "morphology" in p.name]

    if len(morphology_files) == 1:
        path_morpho = morphology_files[0]
    elif len(morphology_files) > 1:
        print(f"Found more morphology files: {[p.name for p in morphology_files]}.\n"
              f"Using the first one")
        path_morpho = morphology_files[0]
    else:
        raise Exception("Morhology file not found! The file name should contain"
                        "'morphology'")

    return ephys.morphologies.NrnFileMorphology(
        str(path_morpho),
        morph_modifiers=morph_modifiers,
        do_replace_axon=do_replace_axon,
        morph_modifiers_kwargs=morph_kwargs
    )


def create_ground_truth_model(model_name, cell_model_folder, release=False, v_init=None, model_type="LFPy", 
                              **morph_kwargs):
    """Create ground-truth model

    Parameters
    ----------
    model_name : str
        'hay' | 'hay_ais' | 'hay_ais_hillock'
    cell_model_folder : str or Path
        Path to the cell model folder 
    release : bool, optional
        If True, release parameters are loaded (for experimental models some dummy parameters are available),
        by default False
    v_init : float, optional
        Initial membrane potential value, by default None
    **morph_kwargs: kwargs for morphology modifiers

    Returns
    -------
    bluepyopt.ephys.models.LFPyCellModel or bluepyopt.ephys.models.CellModel
        The BluePyOpt model object
    """

    if model_name == "hay":
        morph_modifiers = None
        seclist_names = None
        secarray_names = None
        do_replace_axon = True
    elif model_name == "hay_ais_hillock":
        morph_modifiers = [replace_axon_with_hillock_ais]
        seclist_names = ['all', 'somatic', 'basal', 'apical', 'axon_initial_segment', 'hillockal', 'myelinated', 
                         'axonal']
        secarray_names = ['soma', 'dend', 'apic', 'ais', 'hillock', 'myelin', 'axon']
        do_replace_axon = False
    elif model_name == "hay_ais":
        morph_modifiers = [replace_axon_with_ais]
        seclist_names = ['all', 'somatic', 'basal', 'apical', 'axon_initial_segment', 'myelinated', 'axonal']
        secarray_names = ['soma', 'dend', 'apic', 'ais', 'myelin', 'axon']
        do_replace_axon = False
    else:
        morph_modifiers = None
        seclist_names = None
        secarray_names = None
        do_replace_axon = True

    if v_init is None:
        if model_name == "hay":
            v_init = -65.
        elif model_name == "hay_ais":
            v_init = -80.
        elif model_name == "hay_ais_hillock":
            v_init = -80.

    cell_model_folder = Path(cell_model_folder)
    
    if model_type == "LFPy":
        model_class = ephys.models.LFPyCellModel
        model_kwargs = {'v_init': v_init}
    else:
        model_class = ephys.models.CellModel
        model_kwargs = {}

    cell = model_class(
        model_name,
        morph=define_morphology(cell_model_folder, morph_modifiers, do_replace_axon, **morph_kwargs),
        mechs=define_mechanisms(cell_model_folder),
        params=define_parameters(cell_model_folder, release=release, v_init=v_init),
        seclist_names=seclist_names,
        secarray_names=secarray_names,
        **model_kwargs
    )

    return cell


def create_experimental_model(morphology_file, cell_model_folder, release=False, v_init=None, model_type="LFPy",
                              **morph_kwargs):
    """Create experimental cell model


    Parameters
    ----------
    morphology_file : str or Path
        Path to the morphology file
    cell_model_folder : str or Path
        Path to the cell model folder
    release : bool, optional
        If True, release parameters are loaded (for experimental models some dummy parameters are available),
        by default False
    v_init : float, optional
        Initial membrane potential value, by default None
    model_type : str, optional
        * "neuron": instantiate a CellModel
        * "LFPy": instantiate an LFPyCellModel
        by default "LFPy"
    **morph_kwargs: kwargs for morphology modifiers

    Returns
    -------
    bluepyopt.ephys.models.LFPyCellModel or bluepyopt.ephys.models.CellModel
        The BluePyOpt model object
    """
    morph_modifiers = [fix_morphology_exp]

    seclist_names = [
        "all",
        "somatic",
        "basal",
        "apical",
        "axonal",
        "axon_initial_segment"
    ]

    secarray_names = ["soma", "dend", "apic", "axon", "ais"]

    if "abd" in morph_kwargs:
        seclist_names.append("axon_bearing_dendrite")
        secarray_names.append("abd")

    do_replace_axon = False
    model_name = "experimental"

    if v_init is None:
        v_init = -70

    morphology = ephys.morphologies.NrnFileMorphology(
        str(morphology_file),
        morph_modifiers=morph_modifiers,
        do_replace_axon=do_replace_axon,
        morph_modifiers_kwargs=morph_kwargs
    )

    if model_type == "LFPy":
        model_class = ephys.models.LFPyCellModel
        model_kwargs = {'v_init': v_init}
    else:
        model_class = ephys.models.CellModel
        model_kwargs = {}

    cell = model_class(
        model_name,
        morph=morphology,
        mechs=define_mechanisms(cell_model_folder),
        params=define_parameters(cell_model_folder, release=release, v_init=v_init),
        seclist_names=seclist_names,
        secarray_names=secarray_names,
        **model_kwargs
    )

    return cell
