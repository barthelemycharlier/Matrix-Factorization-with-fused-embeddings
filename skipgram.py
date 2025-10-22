import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import random
import numpy as np

class SkipGramNeg(nn.Module):
    def __init__(self, num_nodes, emb_dim):
        super().__init__()
        self.center = nn.Embedding(num_nodes, emb_dim)
        self.context = nn.Embedding(num_nodes, emb_dim)
        nn.init.xavier_uniform_(self.center.weight)
        nn.init.xavier_uniform_(self.context.weight)

    def forward(self, u, pos_v, neg_v):
        u_emb = self.center(u)                        # [B, d]
        pos_emb = self.context(pos_v)                 # [B, d]
        neg_emb = self.context(neg_v)                 # [B, K, d]

        pos_score = torch.sum(u_emb * pos_emb, dim=1)
        neg_score = torch.bmm(neg_emb, u_emb.unsqueeze(2)).squeeze(-1)

        # Ensure 2D shape for neg_score
        if neg_score.dim() == 1:
            neg_score = neg_score.unsqueeze(0)

        loss = -torch.log(torch.sigmoid(pos_score) + 1e-8) \
               -torch.sum(torch.log(torch.sigmoid(-neg_score) + 1e-8), dim=1)
        return loss.mean()

def train_skipgram(sequences, emb_dim=64, window_size=5, neg_samples=5, epochs=5, lr=0.01):
    # --- build vocabulary ---
    nodes = sorted({n for seq in sequences for n in seq})
    node2id = {n: i for i, n in enumerate(nodes)}
    id2node = {i: n for n, i in node2id.items()}
    vocab_size = len(nodes)

    model = SkipGramNeg(vocab_size, emb_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr) ## sparse it up

    # --- build node frequency for negative sampling ---
    freq = defaultdict(int)
    for seq in sequences:
        for n in seq:
            freq[n] += 1
    freq_arr = np.array([freq[id2node[i]] for i in range(vocab_size)])
    neg_prob = freq_arr ** 0.75
    neg_prob /= neg_prob.sum()

    # --- training loop ---
    for epoch in range(epochs):
        total_loss = 0.0
        random.shuffle(sequences)
        for seq in sequences:
            ids = [node2id[n] for n in seq]
            for i, u in enumerate(ids):
                left = max(0, i - window_size)
                right = min(len(ids), i + window_size + 1)
                for j in range(left, right):
                    if i == j: continue
                    pos_v = ids[j]
                    neg_v = np.random.choice(vocab_size, size=neg_samples, p=neg_prob)
                    u_t = torch.tensor([u])
                    pos_t = torch.tensor([pos_v])
                    neg_t = torch.tensor(neg_v).unsqueeze(0)
                    loss = model(u_t, pos_t, neg_t)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
        print(f"Epoch {epoch+1}: loss={total_loss:.4f}")

    # Return embeddings as dict {node: vector}
    with torch.no_grad():
        emb = model.center.weight.cpu().numpy()
    embeddings = {id2node[i]: emb[i] for i in range(vocab_size)}
    return embeddings
