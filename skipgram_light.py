import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np
from collections import defaultdict
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.tensorboard import SummaryWriter


# --- Dataset --------------------------------------------------------
class SkipGramDataset(Dataset):
    def __init__(self, sequences, node2id, window_size, neg_prob, neg_samples):
        self.pairs = []
        self.vocab_size = len(node2id)
        self.neg_prob = neg_prob
        self.neg_samples = neg_samples

        # Generate positive pairs
        for seq in sequences:
            ids = [node2id[n] for n in seq]
            for i, u in enumerate(ids):
                left = max(0, i - window_size)
                right = min(len(ids), i + window_size + 1)
                for j in range(left, right):
                    if i != j:
                        self.pairs.append((u, ids[j]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        u, v = self.pairs[idx]
        neg_v = np.random.choice(self.vocab_size, self.neg_samples, p=self.neg_prob)
        return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(v, dtype=torch.long),
            torch.tensor(neg_v, dtype=torch.long),
        )


# --- Model ----------------------------------------------------------
class SkipGramNeg(pl.LightningModule):
    def __init__(self, num_nodes, emb_dim, lr):
        super().__init__()
        self.save_hyperparameters()
        self.center = nn.Embedding(num_nodes, emb_dim)
        self.context = nn.Embedding(num_nodes, emb_dim)
        nn.init.xavier_uniform_(self.center.weight)
        nn.init.xavier_uniform_(self.context.weight)

    def forward(self, u, pos_v, neg_v):
        u_emb = self.center(u)
        pos_emb = self.context(pos_v)
        neg_emb = self.context(neg_v)

        pos_score = torch.sum(u_emb * pos_emb, dim=1)
        neg_score = torch.bmm(neg_emb, u_emb.unsqueeze(2)).squeeze(-1)

        loss = -torch.log(torch.sigmoid(pos_score) + 1e-8)
        loss -= torch.sum(torch.log(torch.sigmoid(-neg_score) + 1e-8), dim=1)
        return loss.mean()

    def training_step(self, batch, batch_idx):
        u, pos_v, neg_v = batch
        loss = self(u, pos_v, neg_v)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# --- Helper to train ------------------------------------------------
def train_skipgram_lightning(sequences, emb_dim=64, window_size=5,
                             neg_samples=5, epochs=5, lr=0.01, batch_size=512):
    # Vocabulary
    nodes = sorted({n for seq in sequences for n in seq})
    node2id = {n: i for i, n in enumerate(nodes)}
    id2node = {i: n for n, i in node2id.items()}
    vocab_size = len(nodes)

    # Negative-sampling distribution
    freq = defaultdict(int)
    for seq in sequences:
        for n in seq:
            freq[n] += 1
    freq_arr = np.array([freq[id2node[i]] for i in range(vocab_size)])
    neg_prob = freq_arr ** 0.75
    neg_prob /= neg_prob.sum()

    dataset = SkipGramDataset(sequences, node2id, window_size, neg_prob, neg_samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = SkipGramNeg(vocab_size, emb_dim, lr)
    logger = TensorBoardLogger(save_dir="lightning_logs", name="hin_embeddings")

    trainer = pl.Trainer(max_epochs=epochs, accelerator="auto", devices="auto", log_every_n_steps=1,logger = logger)
    trainer.fit(model, loader)

    # Extract learned embeddings
    emb = model.center.weight.detach().cpu().numpy()
    embeddings = {id2node[i]: emb[i] for i in range(vocab_size)}
    writer = SummaryWriter("lightning_logs/hin_embeddings")
    writer.add_embedding(
        torch.tensor(list(embeddings.values())),
        metadata=list(embeddings.keys()),
        tag="HIN_Embeddings"
    )
    writer.close()

    return embeddings
