import matplotlib.pyplot as plt
import numpy as np
import utils
import MEAutility as mu

def plot_responses(responses, max_rows=6, figsize=(10, 10), color="C0", return_fig=False):
    """
    Plots one response to multiple protocols

    Parameters
    ----------
    responses: dict
        Output of run_protocols function
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
        if 'MEA' not in resp_name:
            resp_no_mea[resp_name] = response
    if len(resp_no_mea) <= max_rows:
        nrows = len(resp_no_mea)
        ncols = 1
    else:
        nrows = max_rows
        ncols = int(np.ceil(len(resp_no_mea) / max_rows))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    max_v = -200
    min_v = 200

    for index, (resp_name, response) in enumerate(sorted(resp_no_mea.items())):
        c = index // nrows
        r = np.mod(index, nrows)
        response = responses[resp_name]
        axes[r, c].plot(response['time'], response['voltage'], label=resp_name, color=color)
        axes[r, c].set_title(resp_name)
        if np.max(response['voltage']) > max_v:
            max_v = np.max(response['voltage'])
        if np.min(response['voltage']) < min_v:
            min_v = np.min(response['voltage'])
    for ax in axes[r + 1:, c]:
        ax.axis("off")

    for axr in axes:
        for ax in axr:
            ax.set_ylim(min_v - 10, max_v + 10)
    fig.tight_layout()
    fig.show()

    if return_fig:
        return fig


def plot_multiple_responses(responses_list, max_rows=6, colors=None, cmap="rainbow", figsize=(10, 10),
                            return_fig=False):
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
    resp_no_mea = []

    if colors is not None:
        assert len(colors) == len(responses_list)

    for (resp_name, response) in sorted(responses.items()):
        if 'MEA' not in resp_name:
            resp_no_mea.append(resp_name)
    if len(resp_no_mea) <= max_rows:
        nrows = len(resp_no_mea)
        ncols = 1
    else:
        nrows = max_rows
        ncols = int(np.ceil(len(resp_no_mea) / max_rows))
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
        for index, resp_name in enumerate(sorted(resp_no_mea)):
            c = index // nrows
            r = np.mod(index, nrows)
            response = responses[resp_name]
            axes[r, c].plot(response['time'], response['voltage'], label=resp_name, color=color)
            axes[r, c].set_title(resp_name)
            if np.max(response['voltage']) > max_v:
                max_v = np.max(response['voltage'])
            if np.min(response['voltage']) < min_v:
                min_v = np.min(response['voltage'])
        for ax in axes[r + 1:, c]:
            ax.axis("off")

        for axr in axes:
            for ax in axr:
                ax.set_ylim(min_v - 10, max_v + 10)

    fig.tight_layout()
    fig.show()

    if return_fig:
        return fig


def plot_multiple_eaps(responses_list, protocols, probe, protocol_name="Step1",
                       colors="C0", norm=True, figsize=(7, 12), ax=None):
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

    eaps = []
    for i, fitted in enumerate(responses_list):
        eap = utils.calculate_eap(responses=fitted, protocols=protocols, protocol_name=protocol_name)
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

    for eap in eaps:
        ax = mu.plot_mea_recording(eap, probe, colors=colors[i], ax=ax, vscale=vscale)

    return ax


def plot_eap(responses, protocols, probe, color="C0", protocol_name="Step1", norm=True, figsize=(7, 12), ax=None):
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
    eap = utils.calculate_eap(responses=responses, protocols=protocols, protocol_name=protocol_name)
    if norm:
        eap = eap / np.max(np.abs(eap), 1, keepdims=True)
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
    temp_map = feature

    if log:
        if np.any(temp_map < 1):
            temp_map += (1 - np.min(temp_map))
        temp_map = np.log(temp_map)

    # normalize
    temp_map -= np.min(temp_map)
    temp_map /= np.ptp(temp_map)

    ax = _plot_map(temp_map, locations, cmap, bg, ax, label_color)

    return ax


def _plot_map(temp_map, locations, cmap, bg, ax, label_color):
    x = locations[:, 0]
    y = locations[:, 1]
    x_un = np.unique(x)
    y_un = np.unique(y)

    if len(y_un) == 1:
        pitch_x = np.min(np.diff(x_un))
        pitch_y = pitch_x
    elif len(x_un) == 2:
        pitch_y = np.min(np.diff(y_un))
        pitch_x = pitch_y
    else:
        pitch_x = np.min(np.diff(x_un))
        pitch_y = np.min(np.diff(y_un))

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
        dr = LabeledRectangle(rect, ch, label_color)
        dr.connect()
        drs.append(dr)

    ax.set_xlim(np.min(x) - elec_x / 2, np.max(x) + elec_x / 2)
    ax.set_ylim(np.min(y) - elec_y / 2, np.max(y) + elec_y / 2)
    ax.axis('equal')
    ax.axis('off')

    return ax


class LabeledRectangle:
    lock = None  # only one can be animated at a time

    def __init__(self, rect, channel, color):
        self.rect = rect
        self.press = None
        self.background = None
        self.channel_str = str(channel)
        axes = self.rect.axes
        x0, y0 = self.rect.xy
        self.text = axes.text(x0, y0, self.channel_str, color=color)
        self.text.set_visible(False)

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.rect.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.rect.figure.canvas.mpl_connect('button_release_event', self.on_release)

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.rect.axes:
            return
        if LabeledRectangle.lock is not None:
            return
        contains, attrd = self.rect.contains(event)
        if not contains: return
        x0, y0 = self.rect.xy
        self.press = x0, y0, event.xdata, event.ydata
        LabeledRectangle.lock = self
        self.text.set_visible(True)
        self.text.draw()

    def on_release(self, event):
        'on release we reset the press data'
        if LabeledRectangle.lock is not self:
            return
        self.press = None
        LabeledRectangle.lock = None
        self.text.set_visible(False)
        self.text.draw()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.rect.figure.canvas.mpl_disconnect(self.cidpress)
        self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
