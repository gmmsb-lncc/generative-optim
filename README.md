# generative-optim
Molecular optimization using generative models

## Installation
Clone this repository:

```bash
git clone git@github.com:gmmsb-lncc/generative-optim.git  # ssh
cd generative-optim
```

When using the HierVAE model, create a virtual environment with **Python 3.8** (the latest tested compatible version) and install the dependencies:

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

See the `runs-example.sh` script for an example of how to run the optimization script.

### Optimization algorithm and problems
Choose from the available optimization algorithms and problems (see `--help` for more details). Objectives are defined in the `objectives.conf.json` file.

## Experiment tracking
To visualize experiments using **aim UI**, run the following command in the terminal:
```bash
aim up
```

Then, open the browser at `http://localhost:43800/` to see the experiments.

By default, a checkpoint of the whole population is saved in a `.csv` file inside the `.aim/meta/chunks/{run_hash}/` folder at the end of each generation.
The final population of generated molecules is saved at `.aim/meta/chunks/{run_hash}/generated_mols.txt`.

## Citing
> Matheus Müller Pereira da Silva, Jaqueline da Silva Angelo, Isabella Alvim Guedes, and Laurent Emmanuel Dardenne. 2024. A Generative Evolutionary Many-Objective Framework: A Case Study in Antimicrobial Agent Design. In _Genetic and Evolutionary Computation Conference (GECCO ’24 Companion), July 14–18, 2024, Melbourne, VIC, Australia_. ACM, New York, NY, USA, 8 pages. https://doi.org/10.1145/3638530.3664159
