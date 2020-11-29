import os
import json

import bluepyopt.ephys as ephys
import LFPy
import numpy as np

script_dir = os.path.dirname(__file__)
config_dir = os.path.join(script_dir, "config")


def define_mechanisms():
    """Define mechanisms"""

    mech_definitions = json.load(open(os.path.join(config_dir, "mechanisms.json")))

    mechanisms = []
    for sectionlist, channels in mech_definitions.items():
        seclist_loc = ephys.locations.NrnSeclistLocation(
            sectionlist, seclist_name=sectionlist
        )
        for channel in channels:
            mechanisms.append(
                ephys.mechanisms.NrnMODMechanism(
                    name="%s.%s" % (channel, sectionlist),
                    mod_path=None,
                    suffix=channel,
                    locations=[seclist_loc],
                    preloaded=True,
                )
            )

    return mechanisms


def define_probe(probe_type="linear", num_linear=20, linear_span=[-500, 1000], z_shift=20, probe_center=[0, 300, 20],
                 mea_dim=[20, 4], mea_pitch=[50, 50]):
    """

    Parameters
    ----------
    probe_type
    num_linear
    linear_span
    z_shift
    probe_center
    mea_dim
    mea_pitch

    Returns
    -------

    """
    import MEAutility as mu

    if probe_type == 'linear':
        mea_positions = np.zeros((num_linear, 3))
        mea_positions[:, 2] = z_shift
        mea_positions[:, 1] = np.linspace(linear_span[0], linear_span[1], num_linear)
        probe = mu.return_mea(info={'pos': list([list(p) for p in mea_positions]), 'center': False, 'plane': 'xy'})

    elif probe_type == 'planar':
        mea_info = {'dim': mea_dim,
                    'electrode_name': 'hd-mea',
                    'pitch': mea_pitch,
                    'shape': 'square',
                    'size': 5,
                    'type': 'mea',
                    'plane': 'xy'}
        probe = mu.return_mea(info=mea_info)
        # Move the MEA out of the neuron plane (yz)
        probe.move(probe_center)

    # Instantiate LFPy electrode object
    electrode = LFPy.RecExtElectrode(probe=probe)

    return electrode


def define_parameters():
    """Define parameters"""

    param_configs = json.load(open(os.path.join(config_dir, "parameters.json")))
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
                "Parameter config has to have bounds or value: %s" % param_config
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
            elif param_config["dist_type"] == "exp":
                scaler = ephys.parameterscalers.NrnSegmentSomaDistanceScaler(
                    distribution=param_config["dist"]
                )
            elif param_config["dist_type"] == "step_funct":
                scaler = ephys.parameterscalers.NrnSegmentSomaDistanceScaler(
                    distribution=param_config["dist"]
                )

            seclist_loc = ephys.locations.NrnSeclistLocation(
                param_config["sectionlist"], seclist_name=param_config["sectionlist"]
            )

            name = "%s.%s" % (param_config["param_name"], param_config["sectionlist"])

            if param_config["type"] == "section":
                parameters.append(
                    ephys.parameters.NrnSectionParameter(
                        name=name,
                        param_name=param_config["param_name"],
                        value_scaler=scaler,
                        value=value,
                        frozen=frozen,
                        bounds=bounds,
                        locations=[seclist_loc],
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
                        locations=[seclist_loc],
                    )
                )
        else:
            raise Exception(
                "Param config type has to be global, section or range: %s"
                % param_config
            )

    return parameters


def define_morphology():
    """Define morphology"""

    return ephys.morphologies.NrnFileMorphology(
        os.path.join("morphology/cell1.asc"), do_replace_axon=True
    )


def create():
    """Create cell model"""
    cell = ephys.models.LFPyCellModel(
        'hay',
        v_init=-65.,
        morph=define_morphology(),
        mechs=define_mechanisms(),
        params=define_parameters())
    # cell = ephys.models.CellModel(
    #     'hay',
    #     v_init=-65.,
    #     morph=define_morphology(),
    #     mechs=define_mechanisms(),
    #     params=define_parameters())

    return cell
