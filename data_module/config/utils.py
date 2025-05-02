import os

import yaml


###### FROM HERE ######
def get_path_from_config(key: str) -> str:
    """
    This function is used to get the absolute path of a file or folder
    that is located inside the base project directory.
    """
    # Get the root directory of your project (two levels up from the current script's directory)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

    # Construct the path to config.yml
    config_file_path = os.path.join(os.path.dirname(__file__), "config.yml")

    # Load the config.yml file
    with open(config_file_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Check if the key exists inside the 'data_path' section
    if key in config["data_path"]:
        # Return the absolute path by joining the root directory with the relative path from config.yml
        return os.path.join(root_dir, config["data_path"][key])
    else:
        raise KeyError(f"{key} not found in the config file.")


def get_path_from_config_for_outside_base_folder(key: str) -> str:
    """
    This function is used to get the absolute path of
    a file or folder that is located outside the base project directory.
    """
    # Construct the path to config.yml
    config_file_path = os.path.join(os.path.dirname(__file__), "config.yml")

    # Load the config.yml file
    with open(config_file_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Check if the key exists inside the 'data_path' section
    if key in config["data_path"]:
        # Get the relative path from config
        relative_path = config["data_path"][key]

        # Join the relative path with the base project directory (two levels up from this script)
        base_directory = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )

        # Return the absolute path
        return os.path.abspath(os.path.join(base_directory, relative_path))
    else:
        raise KeyError(f"{key} not found in the config file.")


def get_file_name_str_from_config(key: str) -> str:
    """
    This function is used to get the file name from the config file.
    """
    # Construct the path to config.yml
    config_file_path = os.path.join(os.path.dirname(__file__), "config.yml")

    # Load the config.yml file
    with open(config_file_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    # just read the file name
    return config["data_path"][key]
