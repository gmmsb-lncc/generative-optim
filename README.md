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
Check available algorithms in the pymoo [documentation page](https://pymoo.org/algorithms/list.html#nb-algorithms-list).

Optimization problems are implemented inside the `problems/` folder. So far, the following problems are available:

- single-objective:
    - `MolecularWeight`: generate molecules with a target molecular weight;
    - `SAScore`: generate molecules with a target synthetic accessibility score (calculations done using [RDKit Contrib script](https://github.com/rdkit/rdkit/tree/880a8e5725cf842091c3f273da2b35b13e88fffb/Contrib/SA_Score));
- multi/many-objective:
    - `MWSA`: generate molecules with a target molecular weight and synthetic accessibility score (nobjs=2);

## Experiment tracking
To visualize experiments using **aim UI**, run the following commmand in the terminal:
```bash
aim up
```

Then, open the browser at `http://localhost:43800/` to see the experiments.

By default, a checkpoint of the whole population is saved in a `.csv` file inside the `.aim/meta/chunks/{run_hash}/` folder, at the end of each generation.

