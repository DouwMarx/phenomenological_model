from pypm.phenomenological_bearing_model.bearing_model import Measurement
from pypm.utils.reading_and_writing import get_simulation_properties
import os
import plotly.graph_objs as go
from definitions import plots_dir

spec_dict = get_simulation_properties()


def plot_time_signal(signal, time, title=""):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=signal,
                             mode='lines',
                             name='lines'))
    fig.update_layout(
        title="" + title,
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


# Update the standard properties

spec_dict.update({
    "measurement_noise_standard_deviation": 0,
    "transient_amplitude_standard_deviation": 0,
    "n_measurements": 2,
})

to_show_dict = {
    "Initial conditions for transients": {},
    "Slip variance": {"slip_variance_factor": 0.1},
    "Varying speed": {"speed_profile_type": "sine"},
    "Variation in transient amplitude":{"transient_amplitude_standard_deviation": 0.2},
    "Inner fault Modulation": {"fault_type": "inner"},
    "Measurement with noise": {"measurement_noise_standard_deviation": 0.5},
}

plots = []
for title, to_update in to_show_dict.items():
    spec_copy = spec_dict.copy()
    spec_copy.update(to_update)

    m = Measurement(**spec_copy)
    measurements = m.get_measurements()

    signal = measurements[0]
    time = m.time

    o = plot_time_signal(signal, time, title=title)  # Create the plot

    plots.append(o)

many_figures_to_single_html(plots_dir.joinpath("demonstrate_functionality.html"), plots)

def main():
    return plots