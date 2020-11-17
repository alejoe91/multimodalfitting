import os
import json

import bluepyopt.ephys as ephys

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

    return cell
