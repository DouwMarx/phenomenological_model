from definitions import root_dir
import yaml

def get_simulation_properties():
    """
    Recovers the simulation properties from the .yaml file, flattens it and returns it as a dictionary that is understandable by the phenomenological model
    Returns
    -------

    """
    with open(root_dir.joinpath("simulation_properties.yml"), "r") as file:
        yaml_properties = yaml.safe_load(file)
        simulation_properties = flatten_dict(yaml_properties)
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

