from pathlib import Path

root_dir = Path(__file__).parent # Project Root dir
data_dir = root_dir.joinpath('data') # Data dir
models_dir = root_dir.joinpath('models') # Data dir
reports_dir = root_dir.joinpath('reports') # Data dir
plots_dir = reports_dir.joinpath('plots') # Data dir


data_write_dir = root_dir.parent.joinpath("differentiable_sigproc","data","phenomenological_data.npy")