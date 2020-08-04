import matplotlib.pyplot as plt


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
