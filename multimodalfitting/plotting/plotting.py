import matplotlib.pyplot as plt
from matplotlib import path
from matplotlib.widgets import LassoSelector
from matplotlib.backend_bases import MouseButton
import numpy as np
import MEAutility as mu
from copy import deepcopy
import re

from ..utils import calculate_eap


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def plot_responses(responses, protocol_names=None,
                   max_rows=6, titles=None,
                   figsize=(10, 10), color="C0",
                   return_fig=False):
    """
    Plots one response to multiple protocols

    Parameters
    ----------
    responses: dict
        Output of run_protocols function
    protocol_names: list or None
        List of protocol names (or substrings of protocol names) to plot
    max_rows: int
        Max number of rows (default 6)
    figsize: tuple
        The figure size (default (10, 10))
    color: matplotlib color
        The color to be used
    return_fig: bool
        If True the figure is returned

    Returns
    -------
    fig: matplotlib figure
        If return_fig is True, the figure is returned

    """
    resp_no_mea = {}
    for (resp_name, response) in sorted(responses.items()):
        if 'MEA' not in resp_name and ".v" in resp_name or "i_membrane" in resp_name:
            resp_no_mea[resp_name] = response

    if protocol_names is not None:
        resp_to_plot = {}
        for pn in protocol_names:
            for resp_name in list(resp_no_mea):
                if pn in resp_name:
                    resp_to_plot[resp_name] = response
    else:
        resp_to_plot = resp_no_mea

    # sort responses if multiple runs of the same protocol
    protocol_keys = list(resp_to_plot.keys())
    if protocol_names is None:
        protocol_keys.sort(key=natural_keys)
        protocol_names = [p.split(".")[0] for p in protocol_keys]
    if len(np.unique(protocol_names)) == 1:
        sharex = True
    else:
        sharex = False

    if len(resp_to_plot) <= max_rows:
        nrows = len(resp_to_plot)
        ncols = 1
    else:
        nrows = max_rows
        ncols = int(np.ceil(len(resp_to_plot) / max_rows))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=sharex)

    max_v = -200
    min_v = 200

    resps_voltage = []
    resps_current = []
    

    for index, (resp_name) in enumerate(protocol_keys):
        c = index // nrows
        r = np.mod(index, nrows)
        response = responses[resp_name]
        if isinstance(response, dict):
            columns = response.keys()
        elif isinstance(response.response, dict):
            columns = response.response.keys()
        else:
            columns = response.response.columns
        if "voltage" in columns:
            values = response['voltage']
            resps_voltage.append(index)
            label = "$V_m$ (mV)"
        elif "current" in columns:
            values = response['current']
            resps_current.append(index)
            label = "$I_m$ (nA)"
        
        if ncols > 1:
            axes[r, c].plot(response['time'], values, label=resp_name, color=color)
            ax = axes[r, c]
        elif ncols == 1 and nrows == 1:
            axes.plot(response['time'], values, label=resp_name, color=color)
            ax = axes
        else:
            axes[r].plot(response['time'], values, label=resp_name, color=color)
            ax = axes[r]
        if titles is not None:
            title = [t for t in titles if t in resp_name][0]
            ax.set_title(title, fontsize=15)
        else:
            ax.set_title(resp_name, fontsize=15)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel(label, fontsize=12)
        ax.set_xlabel("$time$ (ms)", fontsize=12)

    # limits
    min_v = max_v = min_i = max_i = None
    for index in resps_voltage:
        v = responses[protocol_keys[index]]["voltage"]
        curr_min = min(v)
        curr_max = max(v)
        if min_v is None:
            min_v = curr_min
        else:
            min_v = min(min_v, curr_min)
        if max_v is None:
            max_v = curr_max
        else:
            max_v = max(max_v, curr_max)
    for index in resps_current:
        i = responses[protocol_keys[index]]["current"]
        curr_min = min(i)
        curr_max = max(i)
        if min_i is None:
            min_i = curr_min
        else:
            min_i = min(min_i, curr_min)
        if max_i is None:
            max_i = curr_max
        else:
            max_i = max(max_i, curr_max)

    if ncols > 1:
        for ax in axes[r + 1:, c]:
            ax.axis("off")
    elif ncols == 1 and nrows == 1:
        axes.axis("off")
        axes.set_ylim(min_v - 10, max_v + 10)
    else:
        for ax in axes[r + 1:]:
            ax.axis("off")
        for ax in axes:
            ax.set_ylim(min_v - 10, max_v + 10)
    
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for index, ax in enumerate(axes.flatten()):
        if index in resps_voltage:
            ptp = max_v - min_v
            ax.set_ylim(min_v - 0.1 * ptp, max_v + 0.1 * ptp)
        elif index in resps_current:
            ptp = max_i - min_i
            ax.set_ylim(min_i  - 0.1 * ptp, max_i + 0.1 * ptp)

    fig.subplots_adjust(hspace=0.8)

    if return_fig:
        return fig


def plot_multiple_responses(responses_list, max_rows=6, protocol_names=None,
                            colors=None, cmap="rainbow", figsize=(10, 10),
                            return_fig=False, titles=None, labels=None):
    """
    Plots a list of responses to multiple protocols

    Parameters
    ----------
    responses_list: dict
        Output of run_protocols function
    max_rows: int
        Max number of rows (default 6)
    figsize: tuple
        The figure size (default (10, 10))
    colors: list of matplotlib colors
        The colors to be used
    cmap: matplotlib colormap
        If colors is None, the colormap to be used
    return_fig: bool
        If True the figure is returned

    Returns
    -------
    fig: matplotlib figure
        If return_fig is True, the figure is returned

    """
    responses = responses_list[0]
    resp_no_mea = {}
    resp_to_plot = {}

    if labels is not None:
        assert len(labels) == len(
            responses_list), "List of labels should have same length of responses_list"

    if colors is not None:
        assert len(colors) == len(
            responses_list), "List of colors should have same length of responses_list"

    for (resp_name, response) in sorted(responses.items()):
        if 'MEA' not in resp_name:
            resp_no_mea[resp_name] = response

    if protocol_names is not None:
        resp_to_plot = {}
        for pn in protocol_names:
            for resp_name in list(resp_no_mea):
                if pn in resp_name:
                    resp_to_plot[resp_name] = response
    else:
        resp_to_plot = resp_no_mea

    # sort responses if multiple runs of the same protocol
    protocol_keys = list(resp_to_plot.keys())
    if protocol_names is None:
        protocol_keys.sort(key=natural_keys)

    if len(resp_to_plot) <= max_rows:
        nrows = len(resp_to_plot)
        ncols = 1
    else:
        nrows = max_rows
        ncols = int(np.ceil(len(resp_to_plot) / max_rows))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    max_v = -200
    min_v = 200

    for i, responses in enumerate(responses_list):

        if cmap is None and colors is None:
            color = f'C{i}'
        elif colors is not None:
            color = colors[i]
        else:
            cm = plt.get_cmap(cmap)
            color = cm(i / len(responses_list))
        # print(i, responses_list)
        for index, resp_name in enumerate(resp_to_plot):
            if labels and index == 0:
                label = labels[i]
            else:
                label = None
            c = index // nrows
            r = np.mod(index, nrows)
            # print(resp_name)
            response = responses[resp_name]
            if titles is not None:
                title = [t for t in titles if t in resp_name][0]
            else:
                title = resp_name
            if response is not None:
                if ncols > 1:
                    axes[r, c].plot(response['time'], response['voltage'], label=label, color=color)
                    ax = axes[r, c]
                elif ncols == 1 and nrows == 1:
                    axes.plot(
                        response['time'], response['voltage'], label=label, color=color)
                    ax = axes
                else:
                    axes[r].plot(response['time'], response['voltage'], label=label, color=color)
                    ax = axes[r]
                if np.max(response['voltage']) > max_v:
                    max_v = np.max(response['voltage'])
                if np.min(response['voltage']) < min_v:
                    min_v = np.min(response['voltage'])
                if label:
                    ax.legend()
                ax.set_title(title)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.set_ylabel("$V_m$ (mV)", fontsize=12)
                ax.set_xlabel("$time$ (ms)", fontsize=12)

        if ncols > 1:
            for ax in axes[r + 1:, c]:
                ax.axis("off")
            for axr in axes:
                for ax in axr:
                    ax.set_ylim(min_v - 10, max_v + 10)
        elif ncols == 1 and nrows == 1:
            axes.axis("off")
            axes.set_ylim(min_v - 10, max_v + 10)
        else:
            for ax in axes[r + 1:]:
                ax.axis("off")
            for ax in axes:
                ax.set_ylim(min_v - 10, max_v + 10)

    fig.subplots_adjust(hspace=0.8)
    fig.subplots_adjust(wspace=0.4)
    fig.show()

    if return_fig:
        return fig


def plot_multiple_eaps(responses_list, protocols, probe, protocol_name="Step1", sweep_id=0,
                       colors="C0", norm=True, figsize=(7, 12), resample_rate_khz=20, ax=None, 
                       labels=None, **eap_kwargs):
    """
    Plots multiple extracellular action potentials (EAPs)

    Parameters
    ----------
    responses_list: list
        List of responses to compute and plot EAPs from
    protocols: list
        List of BPO protocols
    probe: MEAutility.MEA
        The MEAutility probe object
    protocol_name: str
        The protocol to be used to compute EAPs
    colors: str or list
        The color (or colors) to be used. If a list, each element should be a matplotlib color
    norm: bool
        If True, EAPs are normalized
    figsize: tuple
        The figure size in inches
    ax: matplotlib axis
        The axis to plot on

    Returns
    -------
    ax: matplotlib axis
        The output axis
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    if isinstance(colors, (list, np.ndarray)):
        assert len(colors) == len(responses_list), "List of colors should have same length of responses_list"
    elif isinstance(colors, str):
        colors = [colors] * len(responses_list)
        
    if labels is not None:
        assert len(labels) == len(responses_list), "List of labels should have same length of responses_list"

    eaps = []
    for i, fitted in enumerate(responses_list):
        try:
            eap = calculate_eap(fitted, protocol_name, protocols, sweep_id=sweep_id,
                                fs=resample_rate_khz, **eap_kwargs)
        except:
            eap = np.zeros(eaps[-1].shape)
        if norm:
            eap = eap / np.max(np.abs(eap), 1, keepdims=True)
        eaps.append(eap)

    # compute vscale
    if norm:
        vscale = 2
    else:
        max_eap = 0
        for eap in eaps:
            if np.max(np.abs(eap)) > max_eap:
                max_eap = np.max(np.abs(eap))
        vscale = 1.5 * max_eap

    for i, eap in enumerate(eaps):
        ax = mu.plot_mea_recording(eap, probe, colors=colors[i], ax=ax, vscale=vscale)
        if labels:
            line = ax.get_lines()[-1]
            line.set_label(labels[i])
    if labels:
        ax.legend()

    return ax


def plot_eap(responses, protocols, probe, color="C0", protocol_name="Step1", norm=True, figsize=(7, 12), ax=None,
             **calculate_eap_kwargs):
    """
    Plots single extracellular action potential (EAP)

    Parameters
    ----------
    responses: dict
        The dictionary with protocol responses
    protocols: list
        List of BPO protocols
    probe: MEAutility.MEA
        The MEAutility probe object
    protocol_name: str
        The protocol to be used to compute EAPs
    color: str
        The color to be used
    norm: bool
        If True, EAPs are normalized
    figsize: tuple
        The figure size in inches
    ax: matplotlib axis
        The axis to plot on

    Returns
    -------
    ax: matplotlib axis
        The output axis
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    eap = calculate_eap(responses=responses, protocols=protocols, protocol_name=protocol_name,
                        **calculate_eap_kwargs)
    if norm:
        eap = eap / np.ptp(np.abs(eap), 1, keepdims=True)
        vscale = 2
    else:
        vscale = 1.5 * np.max(np.abs(eap))
    ax = mu.plot_mea_recording(eap, probe, colors=color, ax=ax, vscale=vscale)

    return ax


def plot_feature_map(feature, probe, cmap='viridis', log=False,
                     ax=None, bg=True, label_color='r'):
    """
    Plots extracellular feature map. The plot is interactive: clicking on one electrode displays the electrode number

    Parameters
    ----------
    feature: np.array
        Array with feature values for all electrodes
    probe: MEAutility.MEA
        The probe object that specifies locations
    cmap: matplotlib colormap
        The colormap to be used
    log: bool
        If True, the calues are log-scales
    ax: matplotlib axis
        The axis to use
    bg: bool
        If True, the background between electrodes is filled
    label_color: matplotlib color
        The label color for electrode labels

    Returns
    -------
    ax: matplolib axis

    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    locations = np.array([np.dot(probe.positions, probe.main_axes[0]), np.dot(probe.positions, probe.main_axes[1])]).T
    temp_map = deepcopy(feature)

    if log:
        if np.any(temp_map < 1):
            temp_map += (1 - np.min(temp_map))
        temp_map = np.log(temp_map)

    # normalize
    temp_map -= np.min(temp_map)
    temp_map /= np.ptp(temp_map)
    pitch = probe.pitch

    ax = _plot_map(temp_map, locations, pitch, cmap, bg, ax, label_color)

    return ax


def plot_feature_map_w_colorbar(feature, probe, label=None, feature_name=None, cmap='viridis', 
                                height_ratio=[10, 1]):
    import matplotlib as mpl

    fig, axs = plt.subplots(
        nrows=2, ncols=1,
        gridspec_kw={'height_ratios': height_ratio}
    )

    plot_feature_map(feature, probe, bg=False, ax=axs[0], cmap=cmap)

    cmap = plt.get_cmap(cmap)
    norm = mpl.colors.Normalize(vmin=np.min(feature),
                                vmax=np.max(feature))

    cb1 = mpl.colorbar.ColorbarBase(axs[1], cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')
    if label is not None:
        cb1.set_label(label)

    if feature_name is not None:
        fig.suptitle(feature_name, fontsize=15)

    return fig


def _plot_map(temp_map, locations, pitch, cmap, bg, ax, label_color):
    x = locations[:, 0]
    y = locations[:, 1]

    if pitch is None:
        x_un = np.unique(x)
        y_un = np.unique(y)

        if len(y_un) == 1:
            pitch_x = np.min(np.diff(x_un))
            pitch_y = pitch_x
        elif len(x_un) == 1:
            pitch_y = np.min(np.diff(y_un))
            pitch_x = pitch_y
        else:
            pitch_x = np.min(np.diff(x_un))
            pitch_y = np.min(np.diff(y_un))
    else:
        if np.isscalar(pitch):
            pitch_x = pitch
            pitch_y = pitch
        else:
            assert len(pitch) == 2
            pitch_x = pitch[0]
            pitch_y = pitch[1]

    elec_x = 0.9 * pitch_x
    elec_y = 0.9 * pitch_y

    cm = plt.get_cmap(cmap)

    if bg:
        rect = plt.Rectangle((np.min(x) - pitch_x / 2, np.min(y) - pitch_y / 2),
                             float(np.ptp(x)) + pitch_x, float(np.ptp(y)) + pitch_y,
                             color=cm(0), edgecolor=None, alpha=0.9)
        ax.add_patch(rect)

    drs = []
    for ch, (loc, tval) in enumerate(zip(locations, temp_map)):
        if np.isnan(tval):
            color = 'w'
        else:
            color = cm(tval)
        rect = plt.Rectangle((loc[0] - elec_x / 2, loc[1] - elec_y / 2), elec_x, elec_y,
                             color=color, edgecolor=None, alpha=0.9)
        ax.add_patch(rect)

    ax.set_xlim(np.min(x) - elec_x / 2, np.max(x) + elec_x / 2)
    ax.set_ylim(np.min(y) - elec_y / 2, np.max(y) + elec_y / 2)
    ax.axis('equal')
    ax.axis('off')

    return ax


def get_probe(locations, width=10):
    """
    Returns a probeinterface probe with swuare electrodes.

    Parameters
    ----------
    locations: np.array
        Locations of electrodes
    width: float
        Width of electrodes (default=10)

    Returns
    -------
    print: probeinterface.Probe
        The probe object
    """
    import probeinterface as pi
    shapes = "square"
    shape_params = {'width': width}

    probe = pi.Probe(ndim=2, si_units='um')
    probe.set_contacts(positions=locations,
                       shapes=shapes, shape_params=shape_params)
    probe.create_auto_shape(probe_type="rect")
    return probe


def plot_probe(probe, ax=None, electrode_width=10, interactive=True, **kwargs):
    """

    Parameters
    ----------
    probe
    electrode_width
    ax
    interactive
    kwargs

    Returns
    -------

    """
    from probeinterface import plotting

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    locations = probe.positions[:, :2]
    probe_pi = get_probe(locations, width=electrode_width)
    plotting.plot_probe(probe_pi, ax=ax, show_channel_on_click=interactive, **kwargs)

    return ax


def plot_cell(cell, sim, ax=None, detailed=False, param_values={}, **kwargs):
    """

    Parameters
    ----------
    cell
    sim
    ax
    detailed
    kwargs

    Returns
    -------

    """
    import neuroplotlib as nplt
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    cell.freeze(param_values)
    cell.instantiate(sim=sim)
    if detailed:
        nplt.plot_detailed_neuron(cell.lfpy_cell, ax=ax, plane="xy", **kwargs)
    else:
        nplt.plot_neuron(cell.lfpy_cell, ax=ax, plane="xy", **kwargs)
    cell.unfreeze(param_values)
    cell.destroy(sim=sim)
    ax.axis("off")


def select_single_channels(cell, sim, probe, ax=None, plot_probe_kwargs={},
                           plot_cell_kwargs={}):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()

    plot_probe(probe, ax=ax, interactive=False, **plot_probe_kwargs)
    plot_cell(cell, sim, ax=ax, **plot_cell_kwargs)
    ax.set_title("")
    ax.axis("off")

    es = SelectSingleElectrodes(ax, probe)

    def accept(event):
        if event.key == "enter":
            es.disconnect()
            es.canvas.draw()

    fig.canvas.mpl_connect("key_press_event", accept)
    ax.set_title("Left click to select, right click to remove.\nPress enter to accept selected points.")

    return es.selection


def select_mea_sections(cell, sim, probe, ax=None, plot_probe_kwargs={},
                        plot_cell_kwargs={}):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()

    plot_probe(probe, ax=ax, interactive=False, **plot_probe_kwargs)
    plot_cell(cell, sim, ax=ax, **plot_cell_kwargs)
    ax.set_title("")
    ax.axis("off")

    es = SelectMultipleElectrodes(ax, probe)

    def accept(event):
        if event.key == "enter":
            es.disconnect()
            es.canvas.draw()

    fig.canvas.mpl_connect("key_press_event", accept)
    ax.set_title("Draw by hilding left click to select, right click to remove region.\n"
                 "Press enter to accept selected sections.")

    return es.selection


class SelectSingleElectrodes:
    def __init__(self, ax, probe, selection_color="r", max_dist=30):
        self.ax = ax
        self.probe = probe
        self.canvas = ax.figure.canvas
        self.selection_color = selection_color
        self.electrode_positions = probe.positions[:, :2]
        self.selection = []
        self.lines = []
        self.max_dist = max_dist
        self.cid = self.canvas.mpl_connect('button_release_event', self.onclick)

    def onclick(self, event):
        if not event.inaxes:
            return
        if event.button is MouseButton.LEFT:
            x, y = event.xdata, event.ydata
            dists = [np.linalg.norm(np.array(p) - np.array([x, y])) for p in self.electrode_positions]
            selected_idx = np.argmin(dists)
            if dists[selected_idx] < self.max_dist:
                self.selection.append(selected_idx)
                l = self.ax.plot(self.electrode_positions[selected_idx][0],
                                 self.electrode_positions[selected_idx][1],
                                 color=self.selection_color, marker="o", markersize=5,
                                 alpha=0.5)
                self.lines.append(l)
            self.canvas.draw_idle()
        elif event.button is MouseButton.RIGHT:
            x, y = event.xdata, event.ydata
            dists = [np.linalg.norm(np.array(p) - np.array([x, y])) for p in self.electrode_positions]
            selected_idx = np.argmin(dists)
            if selected_idx in self.selection:
                index = self.selection.index(selected_idx)
                line = self.lines[index]
                line.pop(0).remove()
                _ = self.lines.pop(index)
                _ = self.selection.pop(index)
            self.canvas.draw_idle()

    def disconnect(self):
        self.canvas.mpl_disconnect(self.cid)
        self.ax.set_title(f"Selected {len(self.selection)} MEA channels")
        self.canvas.draw_idle()


class SelectMultipleElectrodes:
    def __init__(self, ax, probe, selection_color="r"):
        self.ax = ax
        self.probe = probe
        self.canvas = ax.figure.canvas
        self.selection_color = selection_color
        self.electrode_positions = probe.positions[:, :2]
        self.electrode_tuple = [(p[0], p[1]) for p in probe.positions[:, :2]]
        self.selection = []
        self.lines = []
        self.points = []
        self.lasso = LassoSelector(self.ax, onselect=self.onselect)
        self.cid = self.canvas.mpl_connect('button_release_event', self.onclick)

    def onselect(self, verts):
        p = path.Path(verts)
        v = np.array(verts)
        ind = p.contains_points(self.electrode_tuple)
        selection = []
        points = []
        for i in range(len(self.electrode_positions)):
            if ind[i]:
                selection.append(i)
                p = self.ax.plot(self.electrode_positions[i][0],
                                 self.electrode_positions[i][1],
                                 color=self.selection_color, marker="o", markersize=5,
                                 alpha=0.5)
                points.append(p)
        if len(selection) > 0:
            self.points.append(points)
            self.selection.append(selection)
            l = self.ax.plot(v[:, 0], v[:, 1], color=self.selection_color)
            self.lines.append(l)
        self.canvas.draw_idle()

    def onclick(self, event):
        if not event.inaxes:
            return
        if event.button is MouseButton.RIGHT:
            x, y = event.xdata, event.ydata
            dists = [np.linalg.norm(np.array(p) - np.array([x, y])) for p in self.electrode_positions]
            selected_idx = np.argmin(dists)

            new_selection = []
            for i, sel in enumerate(self.selection):
                if selected_idx in sel:
                    line = self.lines[i]
                    line.pop(0).remove()
                    points = self.points[i]
                    for p in points:
                        p.pop(0).remove()
                    _ = self.points.pop(i)
                    _ = self.lines.pop(i)
                else:
                    new_selection.append(sel)
            self.selection = new_selection

            self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.ax.set_title(f"Selected {len(self.selection)} MEA sections")
        self.canvas.draw_idle()
