import numpy as np
import neo

from bluepyefe.reader import _check_metadata


def wcp_reader(in_data):
    """Reader for .wcp

    Args:
        in_data (dict): of the format
        {
            "filepath": "./XXX.wcp",
            "i_unit": "pA",
            "t_unit": "s",
            "v_unit": "mV",
        }
    """

    _check_metadata(
        in_data,
        wcp_reader.__name__,
        ["filepath", "i_unit", "v_unit", "t_unit"],
    )

    # Read file
    io = neo.WinWcpIO(in_data["filepath"])
    block = io.read_block()

    data = []
    for segment in block.segments:

        trace_data = {
            "voltage": np.array(segment.analogsignals[0]).flatten(),
            "current": np.array(segment.analogsignals[1]).flatten(),
            "dt": 1.0 / int(segment.analogsignals[0].sampling_rate)
        }

        data.append(trace_data)

    return data


def model_reader(in_data):
    """Reader for modeled npz data

    Args:
        in_data (dict): of the format
        {
            "filepath": "./XXX.npz",
            "i_unit": "pA",
            "t_unit": "s",
            "v_unit": "mV",
        }
    """

    _check_metadata(
        in_data,
        model_reader.__name__,
        ["filepath", "i_unit", "v_unit", "t_unit"],
    )

    # Read file
    data = np.load(in_data["filepath"])

    #TODO

    return data
