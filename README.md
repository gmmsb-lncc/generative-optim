# generative-optim
Molecular optimization using generative models

## Installation
When using the HierVAE model, create a virtual environment with **Python 3.8** (latest tested compatible version) and install the dependencies:

```bash
conda create --prefix ./venv-hiervae python=3.8  # using conda for python 3.8
conda activate ./venv-hiervae
python -m pip install -r requirements-hiervae.txt
```

## Usage
First, initialize a new **aim** repository for tracking experiments (just once):
```bash
cd generative-optim
aim init
```


Run the optimization script with the desired arguments:
```bash
python optim.py --help  # show help
```

### Optimization algorithm and problems
Inside the `optim.py` script, set the desired algorithm for optimization.
Check available algorithms in the [pymoo documentation page](https://pymoo.org/algorithms/list.html#nb-algorithms-list).

Molecular optimization problems are implemented inside the `problems/{single, multi}` folders.

## Experiment tracking
To visualize experiments using **aim UI**, run the following commmand in the terminal:
```bash
aim up
```

Then, open the browser at `http://localhost:43800/` to see the experiments.

By default, a checkpoint of the whole population is saved in a `.csv` file inside the `.aim/meta/chunks/{run_hash}/` folder, at the end of each generation.

