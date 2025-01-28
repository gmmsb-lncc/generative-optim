import os

import numpy as np

from problems.docking import DockingProblem


class MockDecoder:
    def decode_population(self, x):
        return x


def test_smiles_to_pdb_generate_files():
    problem = DockingProblem(
        target_value=-np.inf,
        n_var=0,
        lbound=0,
        ubound=0,
        decoder=MockDecoder(),
    )
    smiles_list = ["CCO", "c1ccccc1", "CN(C)C(=O)N"]
    output_files = ["test1.pdb", "test2.pdb", "test3.pdb"]
    problem.convert_smiles_to_pdb(smiles_list, output_files)
    assert all([os.path.exists(file) for file in output_files])
    for file in output_files:
        os.remove(file)


def test_mmffligand_creates_output_files():
    problem = DockingProblem(
        target_value=-np.inf,
        n_var=0,
        lbound=0,
        ubound=0,
        decoder=MockDecoder(),
    )
    smiles_list = ["CCO", "c1ccccc1", "CN(C)C(=O)N"] * 10
    output_files = [f"test{n}.pdb" for n in range(1, 31)]
    problem.convert_smiles_to_pdb(smiles_list, output_files)
    problem.run_mmffligand_parallel(output_files)

    assert all([os.path.exists(file.replace(".pdb", ".top")) for file in output_files])

    for file in output_files:
        os.remove(file)
        os.remove(file.replace(".pdb", ".top"))


def test_dockthor_executes():
    problem = DockingProblem(
        target_value=-np.inf,
        n_var=0,
        lbound=0,
        ubound=0,
        decoder=MockDecoder(),
    )
    smiles_list = ["c1ccccc1", "CN(C)C(=O)N"]
    output_files = ["test1.pdb", "test2.pdb"]
    problem.convert_smiles_to_pdb(smiles_list, output_files)
    problem.run_mmffligand_parallel(output_files)
    problem.run_dockthor_parallel(
        [f.replace(".pdb", ".top") for f in output_files],
        output_dir="tests/data",
        receptor_path="tests/data/2ylc_receptor/2ylc_receptor.in",
        grid_path="tests/data/2ylc_receptor/2ylc_receptor.grid",
    )
    assert all(
        [
            os.path.exists(file.replace(".pdb", "_docked.csv"))
            for file in [
                os.path.join("tests/data", "2ylc_receptor", f) for f in output_files
            ]
        ]
    )
    assert all(
        [
            os.path.exists(file.replace(".pdb", "_docked.mol2"))
            for file in [
                os.path.join("tests/data", "2ylc_receptor", f) for f in output_files
            ]
        ]
    )

    for file in output_files:
        os.remove(file)
        os.remove(file.replace(".pdb", ".top"))
        os.remove(
            os.path.join(
                "tests/data", "2ylc_receptor", file.replace(".pdb", "_docked.csv")
            )
        )
        os.remove(
            os.path.join(
                "tests/data", "2ylc_receptor", file.replace(".pdb", "_docked.mol2")
            )
        )


def test_get_first_molecule_works():
    problem = DockingProblem(
        target_value=-np.inf,
        n_var=0,
        lbound=0,
        ubound=0,
        decoder=MockDecoder(),
    )
    mol2_path = "tests/data/2ylc_receptor/2ylc_ligand_rnum_maestro_docked.mol2"
    problem.get_first_molecule(mol2_path, "tmp.mol2")
    assert os.path.exists("tmp.mol2")
    with open("tmp.mol2") as f:
        mols_count = 0
        for line in f:
            if line.startswith("@<TRIPOS>MOLECULE"):
                mols_count += 1
        assert mols_count == 1
    os.remove("tmp.mol2")


def test_docktdeep_inference():
    problem = DockingProblem(
        target_value=-np.inf,
        n_var=0,
        lbound=0,
        ubound=0,
        decoder=MockDecoder(),
    )
    ligand_paths = [
        "tests/data/2ylc_receptor/2ylc_ligand_rnum_maestro_docked.mol2",
        "tests/data/2ylc_receptor/benzene_docked.mol2",
        "tests/data/2ylc_receptor/urea_docked.mol2",
    ]
    receptor_path = "tests/data/2ylc_receptor/2ylc_hexamer_maestro_noWat_noLig_prep.pdb"
    preds = problem.run_docktdeep_inference(ligand_paths, receptor_path)
    # print("preds", preds)
