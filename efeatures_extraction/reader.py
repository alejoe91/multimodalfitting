import numpy
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
            "voltage": numpy.array(segment.analogsignals[0]).flatten(),
            "current": numpy.array(segment.analogsignals[1]).flatten(),
            "dt": 1.0 / int(segment.analogsignals[0].sampling_rate)
        }

        data.append(trace_data)

    return data

# TO TEST THE READER
# import matplotlib.pyplot as plt
# import numpy
#
# in_data = {
#     "filepath": "./exp_patch_data/cell1_run2.0.wcp",
#     "i_unit": "pA",
#     "t_unit": "s",
#     "v_unit": "mV"
# }
#
# data = wcp_reader(in_data)
#
# fig, ax = plt.subplots(nrows=2)
#
# for trace in data:
#
#     time = numpy.arange(len(trace["voltage"])) * trace["dt"]
#
#     ax[0].plot(time, trace["voltage"])
#     ax[1].plot(time, trace["current"])
