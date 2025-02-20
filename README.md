# generative-optim
 Multi- and many-objective optimization in generative chemistry model latent spaces.

## Installation
1. Clone this repository:

```bash
git clone git@github.com:gmmsb-lncc/generative-optim.git  # ssh
cd generative-optim
```

2. This code uses Python 3.8; to install Python 3.8 in your machine you can use **conda** or follow the [steps](https://askubuntu.com/questions/682869/how-do-i-install-a-different-python-version-using-apt-get) below:
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.8 python3.8-dev python3.8-venv
```

3. Create a virtual environment and activate it:
```bash
python3.8 -m venv env
source env/bin/activate
```

### Optional (integration with DockThor, requires access to the original repo)

4. Install **openbabel** and **dockthor** from source. Instructions can be found at the **dockthor** git repo.

5. _Install_ docktgrid by downloading the package source-code and adapt it to Python 3.8:
```
(...)
```

6. Install `requirements.txt` deps:
```bash
python -m pip install -r requirements-dockthor.txt
```

6. Copy the test data files, network weights (optionally, copy the `receptors/` dir).
7. Execute tests:
```bash
python -m pytest tests/ -vs
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
