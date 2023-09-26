from dataclasses import dataclass

import numpy as np
import rdkit
import torch

import hgraph

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)
__all__ = ["Config", "Decoder"]


def get_vocab(vocab_file="data/chembl/vocab.txt"):
    with open(vocab_file) as f:
        tokens = [x.strip("\r\n ").split() for x in f]
    return hgraph.PairVocab(tokens)


@dataclass
class Config:
    vocab = get_vocab()
    atom_vocab = hgraph.common_atom_vocab
    rnn_type = "LSTM"
    embed_size = 250
    hidden_size = 250
    depthT = 15
    depthG = 15
    diterT = 1
    diterG = 3
    dropout = 0.0
    latent_size = 32


class HierVAEDecoder:
    def __init__(self):
        self.model = self.get_pretrained_model()

    def decode(self, vecs: torch.Tensor) -> str:
        return self(vecs)

    def get_pretrained_model(self):
        """Pretrained model from https://github.com/wengong-jin/hgraph2graph."""

        model = hgraph.HierVAE(Config()).cuda()
        model.load_state_dict(torch.load("ckpt/chembl/model.ckpt")[0])
        model.eval()

        return model

    def __call__(self, vecs: torch.Tensor) -> str:
        if isinstance(vecs, np.ndarray):
            vecs = torch.from_numpy(vecs)
            vecs = torch.as_tensor(vecs, dtype=torch.float32)

        if len(vecs.shape) == 1:
            vecs = vecs.unsqueeze(0)

        vecs = vecs.cuda()
        with torch.no_grad():
            smiles = self.model.decoder.decode(
                (vecs, vecs, vecs), greedy=True, max_decode_step=150
            )

        return smiles
