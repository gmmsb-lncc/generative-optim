# generative-optim
Molecular optimization using generative models

## Installation
Clone this repository:

```bash
git clone git@github.com:gmmsb-lncc/generative-optim.git  # ssh
cd generative-optim
```

When using the HierVAE model, create a virtual environment with **Python 3.8** (latest tested compatible version) and install the dependencies:

```bash
conda create --prefix ./venv python=3.8  # using conda for python 3.8
conda activate ./venv
python -m pip install -r requirements-hiervae.txt
```

## Usage
First, initialize a new **aim** repository for tracking experiments (just once):
```bash
aim init
```

Run the optimization script with the desired arguments:
```bash
python optim.py --help  # show help
```

### Optimization algorithm and problems
Choose from the available optimization algorithms and problems (see `--help` for more details). Objectives are defined in the `objectives.conf.json` file.

## Experiment tracking
To visualize experiments using **aim UI**, run the following commmand in the terminal:
```bash
aim up
```

Then, open the browser at `http://localhost:43800/` to see the experiments.

By default, a checkpoint of the whole population is saved in a `.csv` file inside the `.aim/meta/chunks/{run_hash}/` folder, at the end of each generation.
The final population of generated molecules in saved at `.aim/meta/chunks/{run_hash}/generated_mols.txt`.
