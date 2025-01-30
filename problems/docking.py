"""Optimze the molecular weight of a molecule."""

import os
import random
import string
import subprocess
from multiprocessing import Pool
from typing import Any, List

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from utils import docktdeep

from .molecular_problem import DecoderInterface, MolecularProblem

__all__ = ["DockingProblem"]


def _process_mmff_file(ligand_path: str):
    mmffligand_path = "dockthor-suite/build/bin/mmffligand"
    try:
        subprocess.run(
            [mmffligand_path, "-l", ligand_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        return (ligand_path, True, None)
    except subprocess.CalledProcessError as e:
        return (
            ligand_path,
            False,
            e.stderr.strip() or "Unexpected MMFFLigand convertion error",
        )


def _run_dockthor(
    ligand_path: str,
    output_dir: str,
    receptor_path: str,  #  = "2ylc_receptor.in",
    grid_path: str,  # = "2ylc_receptor.grid",
    dockthor_path: str = "dockthor-suite/build/bin/dockthor-1.3.11",
    grid_center: List[str] = ["-1.2415", "-6.9365", "-14.0990"],
    grid_size: List[str] = ["22.000"] * 3,
):
    params = [
        dockthor_path,
        "--receptor",
        receptor_path,
        "--ligand",
        ligand_path,
        "--grid",
        grid_path,
        "--grid-center",
        *grid_center,
        "--grid-rstep",
        "0.2500",
        "--grid-size",
        *grid_size,
        "--cluster-max-num",
        "10000",
        "--ga-nrun",
        "1",
        "--ga-evaluations",
        "700000",
        "--ga-init-evaluations",
        "300000",
        "--grid-soft-vdw",
        "0.35",
        "--grid-soft-type",
        "1",
        "--grid-all-types",
        "--output-dir",
        output_dir,
    ]

    try:
        subprocess.run(
            params,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        return (ligand_path, True, None)
    except subprocess.CalledProcessError as e:
        return (
            ligand_path,
            False,
            e.stderr.strip() or "Unexpected DockThor error",
        )


class DockingProblem(MolecularProblem):
    """Optimize the the predicted binding affinity using dockthor for docking compounds and docktdeep for scoring them.

    Minimize the predicted binding affinity in kcal/mol.
    """

    def __init__(
        self,
        receptor_name: str,
        receptor_path: str,
        grid_path: str,
        target_value: float,
        n_var: int,
        lbound: float,
        ubound: float,
        decoder: DecoderInterface,
        run_hash: str = "",
        *args,
        **kwargs,
    ):
        super().__init__(target_value, n_var, lbound, ubound, decoder, *args, **kwargs)
        self.n_evals = 0
        self.receptor_name = receptor_name
        self.receptor_path = receptor_path
        self.grid_path = grid_path
        self.docktdeep_weights = "utils/docktdeep-weights.ckpt"
        self.run_hash = run_hash
        self.docking_dir = self.mk_docking_dir()

    def mk_docking_dir(self):
        """Create a new directory for storing docking files."""
        dir = os.path.join("dockings", self.run_hash, self.receptor_name)
        if not dir:
            return ""  # no directory specified

        os.makedirs(dir, exist_ok=True)
        return dir

    def convert_smiles_to_pdb(
        self, smiles_list: List[str], output_files: List[str], root_dir: str = ""
    ) -> None:
        """
        Convert a list of SMILES strings to PDB files with 3D coordinates generated using RDKit's ETKDG method.

        Args:
            smiles_list (list): List of SMILES strings to convert
            output_files (list): List of output file paths
        """
        if root_dir:
            output_files = [os.path.join(root_dir, f) for f in output_files]

        for smi, out_path in zip(smiles_list, output_files):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smi}")

            mol = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.maxAttempts = 1000  # increase number of attempts
            params.pruneRmsThresh = 0.1  # adjust pruning threshold
            params.randomSeed = 0xF00D  # set random seed

            code = AllChem.EmbedMolecule(mol, params)
            if code < 0:  # fallback to random coordinates
                params.useRandomCoords = True
                code = AllChem.EmbedMolecule(mol, params)
                if code < 0:
                    raise ValueError(f"3D embedding failed for SMILES: {smi}")

            # AllChem.UFFOptimizeMolecule(mol)
            Chem.MolToPDBFile(mol, out_path)

    def run_mmffligand_parallel(
        self, file_paths: List[str], root_dir: str = ""
    ) -> List[Any]:
        """
        Executes 'mmffligand -l <file>' for each file in parallel using all CPU cores.

        Args:
            file_paths (list): List of PDB file paths to process

        Returns:
            list: Tuples of (file_path, success_status, error_message)
        """
        if root_dir:
            file_paths = [os.path.join(root_dir, f) for f in file_paths]

        with Pool() as pool:
            results = pool.map(_process_mmff_file, file_paths)

        return results

    def run_dockthor_parallel(
        self,
        file_paths: List[str],
        output_dir: str,
        receptor_path: str,
        grid_path: str,
        root_dir: str = "",
    ) -> List[Any]:
        """
        Executes a sigle docking run with 'dockthor -r <receptor> -l <ligand> -o <output_dir>' for each file in parallel using all CPU cores.

        Args:
            file_paths (list): List of PDB file paths to process
            output_dir (str): Output directory for DockThor results

        Returns:
            list: Tuples of (file_path, success_status, error_message)
        """
        if root_dir:
            file_paths = [os.path.join(root_dir, f) for f in file_paths]

        with Pool() as pool:
            results = pool.starmap(
                _run_dockthor,
                [(path, output_dir, receptor_path, grid_path) for path in file_paths],
            )

        return results

    def get_first_molecule(self, mol2_file: str, output_file: str = None) -> None:
        """
        Overwrites a multi-mol2 file with only its first molecule entry.

        Args:
            mol2_file (str): Path to the .mol2 file to modify
        """
        if output_file is None:
            output_file = mol2_file

        with open(mol2_file, "r") as f:
            lines = f.readlines()

        start = None
        end = None
        for i, line in enumerate(lines):
            if line.startswith("@<TRIPOS>MOLECULE"):
                start = i
                break

        if start is None:
            raise ValueError("No molecules found in the file")

        for i in range(start + 1, len(lines)):
            if lines[i].startswith("@<TRIPOS>MOLECULE"):
                end = i
                break

        with open(output_file, "w") as f:
            f.writelines(lines[start:end] if end else lines[start:])

    def prepare_docktdeep_input(
        self, file_paths: List[str], root_dir: str = ""
    ) -> None:
        """
        Prepare input files for docktdeep by extracting the first molecule from each mol2 file from dockthor's output.

        Args:
            file_paths (list): List of PDB file paths to process
        """
        if root_dir:
            file_paths = [os.path.join(root_dir, f) for f in file_paths]

        for file in file_paths:
            self.get_first_molecule(file)

    def run_docktdeep_inference(
        self,
        ligand_paths: List[str],
        receptor_path: str,
        ckpt_path: str = "utils/docktdeep-weights.ckpt",
        batch_size: int = 32,
        root_dir: str = "",
    ) -> np.ndarray:
        # self.prepare_docktdeep_input(ligand_paths, root_dir)
        dataset = docktdeep.get_dataset(
            protein_files=[receptor_path] * len(ligand_paths),
            ligand_files=[os.path.join(root_dir, f) for f in ligand_paths],
            root_dir="",
        )
        model = docktdeep.get_model(ckpt_path)
        preds = docktdeep.inference(dataset, model, batch_size=batch_size)

        return preds

    def generate_file_name_ids(self, n: int, prefix: str) -> List[str]:
        """Generate random Ids for the file names."""
        return [f"gen={prefix}chr={i}.pdb" for i in range(n)]

    def evaluate_mols(self, mols: List[str]) -> np.ndarray:
        """Calculates the fitness of a list of molecules based on the target value."""

        lig_files = self.generate_file_name_ids(len(mols), self.n_evals)
        self.convert_smiles_to_pdb(mols, lig_files, self.docking_dir)
        mmff_res = self.run_mmffligand_parallel(lig_files, self.docking_dir)

        for res in mmff_res:  # log errors
            if not res[1]:
                print(f"Error in MMFFLigand file {res[0]}: {res[2]}")

        # copy grid to docking dir
        grid_name = os.path.basename(self.grid_path)
        grid_path = os.path.join(self.docking_dir, f"{self.receptor_name}_receptor")
        os.makedirs(grid_path, exist_ok=True)
        if not os.path.exists(os.path.join(grid_path, grid_name)):
            subprocess.run(
                ["cp", self.grid_path, os.path.join(grid_path, grid_name)], check=True
            )

        dockthor_res = self.run_dockthor_parallel(
            [f.replace(".pdb", ".top") for f in lig_files],
            root_dir=self.docking_dir,
            output_dir=self.docking_dir,
            receptor_path=self.receptor_path,  # pdb file
            grid_path=self.grid_path,  # grid file
        )

        for res in dockthor_res:  # log errors
            if not res[1]:
                print(f"Error in DockThor file {res[0]}: {res[2]}")

        self.prepare_docktdeep_input(  # exclude all other molecules from the mol2 files
            [f.replace(".pdb", "_docked.mol2") for f in lig_files],
            os.path.join(self.docking_dir, f"{self.receptor_name}_receptor"),
        )

        preds = self.run_docktdeep_inference(
            [f.replace(".pdb", "_docked.mol2") for f in lig_files],
            receptor_path=self.receptor_path,
            ckpt_path=self.docktdeep_weights,
            root_dir=os.path.join(self.docking_dir, f"{self.receptor_name}_receptor"),
        )

        self.n_evals += 1
        fitness = np.abs(preds - self.target)
        return fitness

    def _evaluate(self, x, out, *args, **kwargs):
        mols = self.decode_population(x)
        out["F"] = self.evaluate_mols(mols)
