from definitions import root_dir
import yaml
from pathlib import Path

def get_simulation_properties(quick_iter = False):
    """
    Recovers the simulation properties from the .yaml file, flattens it and returns it as a dictionary that is understandable by the phenomenological model
    Returns
    -------

    """
    dir = Path(__file__).parent.parent.parent
    # with open(root_dir.joinpath("simulation_properties.yml"), "r") as file:
    with open(dir.joinpath("simulation_properties.yml"), "r") as file:
            yaml_properties = yaml.safe_load(file)
            simulation_properties = flatten_dict(yaml_properties)

    if quick_iter:
        simulation_properties["t_duration"] = 1
        simulation_properties["sampling_frequency"] = 10000
        simulation_properties["n_measurements"] = 10
    return simulation_properties


def flatten_dict(dict):
    """
    Extracts the leafs of a dictionary with 2 levels
    Parameters
    ----------
    dict

    Returns
    -------

    """
    return {key: val for sub_dict in dict.values() for key, val in sub_dict.items()}

def main():
    a = get_simulation_properties()
    return a

if __name__ == "__main__":
    main()