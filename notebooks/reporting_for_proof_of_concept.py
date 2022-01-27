import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from definitions import data_dir, plots_dir
import plotly.graph_objs as go
import matplotlib


def loop_through_mode_and_severity(results_dictionary, function_to_apply):
    r = []
    for mode, dictionary in results_dictionary.items():
        for severity, measurements in dictionary.items():
            plot_obj = function_to_apply(mode, severity, results_dictionary)
            r.append(plot_obj)
    return r


def compare_augmented_and_generated_ses(mode, severity, results_dict):
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
        title="Failure mode:" + mode + " | Severity: " + severity,
        xaxis_title="frequency [Hz]",
        yaxis_title="envelope spectrum",
        # legend_title="Legend Title",
    )

    return fig


def show_example_time_signal(mode, severity, results_dict):
    meas = results_dict[mode][severity]
    sig = meas["time_domain"][0]
    time = meas["meta_data"]["measured_time"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=sig,
                             mode='lines',
                             name='lines'))
    fig.update_layout(
        title="Failure mode:" + mode + " | Severity: " + severity,
        xaxis_title="time [s]",
        yaxis_title="acceleration [m/s^2]",
        # legend_title="Legend Title",
    )

    return fig


def many_figures_to_single_html(pathlib_path, list_of_figure_objects):
    if pathlib_path.exists():
        os.remove(pathlib_path)
        print("File overwritten")
    with open(str(pathlib_path), 'a') as f:
        for figure in list_of_figure_objects:
            f.write(figure.to_html(full_html=False, include_plotlyjs='cdn'))


def generate_encoding_plots(data_dict, severity_to_show, show_augmented_encoding=False):
    figures = []

    if severity_to_show=="all":
        severities = list(data_dict["ball"].keys()) # Extract all the possible severities
        print(severities)
    else:
        severities = [severity_to_show]

    # Create two different plots, one for the model trained on healthy data only and one trained on both the healthy data and the augmented data.
    for model_name, plot_title in zip(
            ["healthy_only", "healthy_and_augmented"],
            ["healthy data only", "both healthy data and augmented data"]):

        #Define the figure object
        fig = go.Figure()

        # Plot healthy data, choice of "failure mode" for healthy is arbitrary
        healthy_encoding = data_dict["ball"]["0"]["envelope_spectrum_encoding"][model_name]
        fig.add_trace(go.Scatter(x=healthy_encoding[:, 0], y=healthy_encoding[:, 1],
                                 mode='markers',
                                 name="healthy"))

        # Plot faulty data
        # colormap = plt.colormaps()[1]
        map_name = plt.colormaps()[10]
        # cmap = matplotlib.cm.get_cmap('Spectral')
        cmap = matplotlib.cm.get_cmap(map_name)
        # print(severities)
        for severity in severities:

            for mode_name, mode_data in data_dict.items():
                # Show the encoding of the test data
                encoding = mode_data[severity]["envelope_spectrum_encoding"][model_name]
                color_number = float(severity)/len(severities)
                print(color_number)
                color =matplotlib.colors.colorConverter.to_rgb(cmap(color_number))# cmap(0.1)#matplotlib.colors.colorConverter.to_rgba(cmap(0.1))
                # print(color)
                fig.add_trace(go.Scatter(x=encoding[:, 0], y=encoding[:, 1],
                                         mode='markers',
                                         # marker={"color":["#00D"]},
                                         # marker={"color":[color]},
                                         # color_discrete=matplotlib.colors.colorConverter.to_rgb(cmap(0.1)),#float(severity)/len(severities))),
                                         # marker={
                                         #     'color': matplotlib.colors.colorConverter.to_rgb(cmap(0.1)),
                                         #     # 'size': 5,
                                         #     # 'opacity': 0.6
                                         # },
                                         name=mode_name + " severity: " + severity))
                # fig.update_traces(marker=dict(
                #     color='red'))


                if show_augmented_encoding:
                    # Possibly include the encodings for the augmented data
                    augmented_encoding = mode_data[severity]["augmented_envelope_spectrum_encoding"][model_name]
                    fig.add_trace(go.Scatter(x=augmented_encoding[:, 0], y=augmented_encoding[:, 1],
                                             mode='markers',
                                             marker_symbol="x",
                                             name=mode_name + "augmented encoding, severity:" + severity))

            if show_augmented_encoding:
                # Mention that the encoded data in shown in the representation
                plot_title = plot_title + " | Augmented encoding shown as crosses"

        fig.update_layout(
            title="Model trained on " + plot_title,
            xaxis_title="Principle component 1",
            yaxis_title="Principle component 2",
        )

        figures.append(fig)

    return figures


# Loading the dataset
# results_dictionary = np.load(data_dir.joinpath("generated_and_augmented.npy"), allow_pickle=True)[()]
results_dictionary = np.load(data_dir.joinpath("data_with_encodings.npy"), allow_pickle=True)[()]

time_series_figures = loop_through_mode_and_severity(results_dictionary, show_example_time_signal)
many_figures_to_single_html(plots_dir.joinpath("time_series.html"), time_series_figures)

ses_figures = loop_through_mode_and_severity(results_dictionary, compare_augmented_and_generated_ses)
many_figures_to_single_html(plots_dir.joinpath("ses.html"), ses_figures)

pca_figs = generate_encoding_plots(results_dictionary, "8") + generate_encoding_plots(results_dictionary, "8",show_augmented_encoding=True)
many_figures_to_single_html(plots_dir.joinpath("pca.html"), pca_figs)

pca_figs_all_severity = generate_encoding_plots(results_dictionary, "all",show_augmented_encoding=True)
many_figures_to_single_html(plots_dir.joinpath("pca_all_sev.html"), pca_figs_all_severity)
