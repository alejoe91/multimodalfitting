import os
import json
import pathlib

import bluepyopt.ephys as ephys
import LFPy

import numpy as np

from morphology_modifiers import replace_axon_with_hillock

script_dir = os.path.dirname(__file__)
config_dir = os.path.join(script_dir, "config")


def define_mechanisms(model):
    """Defines mechanisms"""

    path_mechs = pathlib.Path(f"{model}_model") / "mechanisms.json"

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
    electrode: LFPy.RecExtElectrode
        The LFPy electrode object
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

        with probe_file.open('r') as f:
            info = json.load(f)

        probe = mu.return_mea(info=info)

    return LFPy.RecExtElectrode(probe=probe)


def define_parameters(model, release=False):
    """
    Defines parameters

    Parameters
    ----------
    model: str
            "hay" or "hallerman"
    release: bool
        If True, the frozen release parameters are returned. Otherwise, the unfrozen parameters with bounds are
        returned (use False - default - for optimizations)

    Returns
    -------
    parameters: list
        List of BPO parameters
    """

    path_params = pathlib.Path(f"{model}_model")

    if release:
        param_configs = json.load(open(path_params / "parameters_release.json"))
    else:
        param_configs = json.load(open(path_params / "parameters.json"))

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
            parameters.append(
                ephys.parameters.NrnGlobalParameter(
                    name=param_config["param_name"],
                    param_name=param_config["param_name"],
                    frozen=frozen,
                    bounds=bounds,
                    value=value,
                )
            )

        elif param_config["type"] in ["section", "range"]:

            if param_config["dist_type"] == "uniform":
                scaler = ephys.parameterscalers.NrnSegmentLinearScaler()
            elif param_config["dist_type"] in ["exp", "step_funct"]:

                if "soma_ref_point" in param_config:
                    ref_point = param_config["soma_ref_point"]
                else:
                    ref_point = 0.5

                scaler = ephys.parameterscalers.NrnSegmentSomaDistanceScaler(
                    distribution=param_config["dist"],
                    soma_ref_location=ref_point
                )

            seclist_loc = [ephys.locations.NrnSeclistLocation(
                param_config["sectionlist"],
                seclist_name=param_config["sectionlist"]
            )]

            name = f"{param_config['param_name']}.{param_config['sectionlist']}"

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
                    )
                )
        else:
            raise Exception(
                "Param config type has to be global, section or range: %s"
                % param_config
            )

    return parameters


def define_morphology(model, morph_modifiers, do_replace_axon):
    """
    Defines neuron morphology for the Hay model

    Parameters
    ----------
    model: str
            "hay" or "hallerman"
    morph_modifiers: list of python functions
        The modifier functions to apply to the axon
    do_replace_axon: bool

    Returns
    -------
    morphology: bluepyopt.ephys.morphologies.NrnFileMorphology
        The morphology object
    """

    path_morpho = pathlib.Path(f"{model}_model") / "morphology.asc"

    return ephys.morphologies.NrnFileMorphology(
        str(path_morpho),
        morph_modifiers=morph_modifiers,
        do_replace_axon=do_replace_axon
    )


def create(model, morph_modifier="", release=False):
    """
    Create Hay cell model

    Parameters
    ----------
    model: str
            "hay" or "hallerman"
    morph_modifier: str
        The modifier to apply to the axon:
            - "hillock": the axon is replaced with an axon hillock, an AIS, and
                a myelinated linear axon.
               The hillock morphology uses the original axon reconstruction.
               The 'axon', 'ais', 'hillock', and 'myelin' sections are added.
            - "taper": the axon is replaced with a tapered hillock
            - "": the axon is replaced by a 2-segment axon stub
    release: bool
        If True, the frozen release parameters are returned. Otherwise, the
        unfrozen parameters with bounds are returned (use False for
        optimizations).

    Returns
    -------
    cell: bluepyopt.ephys.models.LFPyCellModel
        The LFPyCellModel object
    """

    if morph_modifier == 'hillock':
        morph_modifiers = [replace_axon_with_hillock]
        seclist_names = ['all', 'somatic', 'basal', 'apical', 'axonal',
                         'myelinated', 'axon_initial_segment', 'hillockal']
        secarray_names = ['soma', 'dend', 'apic', 'axon', 'myelin',
                          'ais', 'hillock']
        do_replace_axon = False

    elif morph_modifier == "":
        morph_modifiers = None
        seclist_names = None
        secarray_names = None
        do_replace_axon = True

    else:
        raise Exception("Unknown morph_modifier")

    cell = ephys.models.LFPyCellModel(
        model,
        v_init=-65.,
        morph=define_morphology(model, morph_modifiers, do_replace_axon),
        mechs=define_mechanisms(model),
        params=define_parameters(model, release),
        seclist_names=seclist_names,
        secarray_names=secarray_names
    )

    return cell