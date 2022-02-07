# Phenomenological model

Python implementation of phenomenological bearing model. 
See docs/procedure.md for an high level explanation of what the code is doing.

## Usage
- Edit the *simulation_properties.yml* file to change the simulation properties. 
- See the *make_example_dataset.py* file for example usage of generating an entire dataset
- See the html file generated by *make_plots_that_illustrate_functionality.py* to understand the various parameters that govern the model response.

## Installation / Getting the code to run on your machine
- clone the repository from Github:
  git clone https://github.com/DouwMarx/phenomenological_model.git

- create a new environment using conda or virtual environment:
   conda create --name phenomenological_model 

- Install the packaged required to run the code:
  pip install -r requirements.txt