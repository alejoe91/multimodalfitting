import matplotlib.pyplot as plt
import numpy as np

def plot_responses(responses, return_fig=False):
    resp_no_mea = {}
    for (resp_name, response) in sorted(responses.items()):
        if 'MEA' not in resp_name:
            resp_no_mea[resp_name] = response
    fig, axes = plt.subplots(len(resp_no_mea), figsize=(10,10))
    for index, (resp_name, response) in enumerate(sorted(resp_no_mea.items())):
        axes[index].plot(response['time'], response['voltage'], label=resp_name)
        axes[index].set_title(resp_name)
    fig.tight_layout()
    fig.show()

    if return_fig:
        return fig


def plot_multiple_responses(responses_list, cmap=None, colors=None, return_fig=False):
    responses = responses_list[0]
    resp_no_mea = []

    if colors is not None:
        assert len(colors) ==  len(responses)

    for (resp_name, response) in sorted(responses.items()):
        if 'MEA' not in resp_name:
            resp_no_mea.append(resp_name)
    fig, axes = plt.subplots(len(resp_no_mea), figsize=(10, 10))
    for i, responses in enumerate(responses_list):
        if cmap is None and colors is None:
            color = f'C{i}'
        elif colors is not None:
            color = colors[i]
        else:
            cm = plt.get_cmap(cmap)
            color = cm(i / len(responses_list))
        for index, resp_name in enumerate(sorted(resp_no_mea)):
            response = responses[resp_name]
            axes[index].plot(response['time'], response['voltage'], label=resp_name, color=color)
            axes[index].set_title(resp_name)
    fig.tight_layout()
    fig.show()

    if return_fig:
        return fig


def plot_feature_map(feature, probe, cmap='viridis', norm=False, log=False,
                     ax=None, bg='on', label_color='r'):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    locations = np.array([np.dot(probe.positions, probe.main_axes[0]), np.dot(probe.positions, probe.main_axes[1])]).T

    # normalize
    if norm:
        feature -= np.nanmin(feature)
        feature /= np.nanmax(feature)

    ax = _plot_map(feature, locations, cmap, bg, ax, label_color)

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

    if bg == 'on':
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
