import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from definitions import data_dir,plots_dir
import plotly.graph_objs as go

# import plotly.io as pio
# pio.renderers.default = "browser"

# Loading the dataset
results_dictionary = np.load(data_dir.joinpath("generated_and_augmented.npy"),allow_pickle=True)[()]

def loop_through_mode_and_severity(results_dictionary,function_to_apply):
    r = [ ]
    for mode,dictionary in results_dictionary.items():
        for severity, measurements in dictionary.items():
            plot_obj = function_to_apply(mode,severity,results_dictionary)
            r.append(plot_obj)
    return r


def compare_augmented_and_generated_ses(mode,severity,results_dict):
    # Compare the augmented data with the generated data from the phenomenological model.
    meas = results_dict[mode][severity]
    real_world = meas["envelope_spectrum"]
    augmented = meas["augmented_envelope_spectrum"]

    real_world_mag = real_world["mag"][0]
    real_world_freq = real_world["freq"]

    fig = go.Figure()
    # The real data
    fig.add_trace(go.Scatter(x=real_world_freq, y=real_world_mag,
                             mode='lines',
                             name='Measured'))

    augmented_mag = augmented["mag"][0]
    augmented_freq = augmented["freq"]
    # The augmented data
    fig.add_trace(go.Scatter(x=augmented_freq, y=augmented_mag,
                             mode='lines',
                             name='Augmented healthy data'))



    fig.update_layout(
        title="Failure mode:" + mode +" | Severity: " + severity,
        xaxis_title="frequency [Hz]",
        yaxis_title="envelope spectrum",
        # legend_title="Legend Title",
    )

    return fig

def show_example_time_signal(mode,severity,results_dict):
    meas = results_dict[mode][severity]
    sig = meas["time_domain"][0]
    time = meas["meta_data"]["measured_time"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=sig,
                             mode='lines',
                             name='lines'))
    fig.update_layout(
        title="Failure mode:" + mode +" | Severity: " + severity,
        xaxis_title="time [s]",
        yaxis_title="acceleration [m/s^2]",
        # legend_title="Legend Title",
    )

    return fig


def many_figures_to_single_html(pathlib_path,list_of_figure_objects):
    if pathlib_path.exists():
        os.remove(pathlib_path)
        print("File overwritten")
    with open(str(pathlib_path), 'a') as f:
        for figure in list_of_figure_objects:
            f.write(figure.to_html(full_html=False, include_plotlyjs='cdn'))

# time_series_figures = loop_through_mode_and_severity(results_dictionary, show_example_time_signal)
# many_figures_to_single_html(str(plots_dir.joinpath("time_series.html")),time_series_figures)

ses_figures = loop_through_mode_and_severity(results_dictionary, compare_augmented_and_generated_ses)
many_figures_to_single_html(plots_dir.joinpath("ses.html"),ses_figures)

