# function to get the model from a checkpoint

import os

import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import docktgrid
from docktgrid.voxel_dataset import VoxelDataset
from utils.docktdeep_model import Baseline


def get_model(ckpt_path: str, module: pl.LightningModule = Baseline):
    model = module.load_from_checkpoint(ckpt_path)
    model.eval().cuda()
    return model


def get_voxel_grid():
    voxel = docktgrid.VoxelGrid(
        views=[docktgrid.view.VolumeView(), docktgrid.view.BasicView()],
        vox_size=1.0,
        box_dims=[24.0, 24.0, 24.0],
    )
    return voxel


def get_dataset(
    protein_files,  #: list[str],
    ligand_files,  # : list[str],
    root_dir="",
):

    if root_dir:
        protein_files = [os.path.join(root_dir, f"{f}") for f in protein_files]
        ligand_files = [os.path.join(root_dir, f"{f}") for f in ligand_files]

    data = VoxelDataset(
        protein_files=protein_files,
        ligand_files=ligand_files,
        labels=np.zeros(len(protein_files)),
        voxel=get_voxel_grid(),
    )

    return data


def inference(
    dataset: Dataset, model: pl.LightningModule, batch_size: int = 32
) -> np.ndarray:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []

    with torch.no_grad():
        for batch in dataloader:
            x, _ = batch
            batch_pred = model(x)
            preds.append(batch_pred.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    return preds.squeeze()
