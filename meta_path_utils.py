import random
import numpy as np
import pandas as pd
import networkx as nx
from skipgram import train_skipgram
from skipgram_light import train_skipgram_lightning

def process_name_genre(df):

    df["year"] = df["raw_title"].str.extract(r"\((\d{4})\)").astype(float)
    df["title"] = df["raw_title"].str.replace(r"\s*\(\d{4}\)", "", regex=True)

    # --- Process genre column ---
    # Split genres by "|" into list of strings
    df["genres"] = df["genre"].fillna("").apply(lambda x: x.split("|") if x else [])

    # --- Drop or reorder columns if desired ---
    df = df[["title", "year", "genres"]]
    return df



def build_hin(R_combined: np.ndarray, df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    n_users, n_movies = R_combined.shape
    # Fix missing years manually
    df.loc[df["title"].str.contains("Babylon 5", case=False), "year"] = 1998
    df.loc[df["title"].str.contains("Moonlight", case=False), "year"] = 2016
    df.loc[df["title"].str.contains("The OA", case=False), "year"] = 2016

    # Confirm


    # Users
    for u in range(n_users):
        G.add_node(f"U{u}", type="user")

    # Movies
    for m in range(n_movies):
        G.add_node(f"M{m}", type="movie")

    # User–Movie edges 
    for u in range(n_users):
        rated = np.where(~np.isnan(R_combined[u]))[0]
        for m in rated:
            G.add_edge(f"U{u}", f"M{m}", etype="U-M", weight=float(R_combined[u, m]))

    # Movie–Genre and Movie–Year edges
    for m, row in df.reset_index(drop=True).iterrows():
        # Genres
        for g in row["genres"]:
            gnode = f"G_{g}"
            if gnode not in G:
                G.add_node(gnode, type="genre")
            G.add_edge(f"M{m}", gnode, etype="M-G")

        # Year
        y = row["year"]
        if pd.notna(y):
            ynode = f"Y_{int(y)}"
            if ynode not in G:
                G.add_node(ynode, type="year")
            G.add_edge(f"M{m}", ynode, etype="M-Y")

    return G

def meta_path_walks_single(G: nx.Graph,
                           meta_types: list,
                           target_type: str,
                           d: int = 64,
                           wl: int = 40,
                           ns: int = 5,
                           r: int = 10,
                           epochs = 5,
                           lr = 0.001):
    """
    Implements Algorithm 1:
    - For each node v of target_type:
        - Do r walks, each of length wl steps (counting only target_type nodes)
        - At each step, move according to the meta-path (cycled) and pick a neighbor
          uniformly among nodes of the required type.
        - Append to path ONLY when the visited node has target_type (type filtering).
    - Return the list of homogeneous sequences (paths) and the model trained on them.
    """
    # Helper: get node type quickly
    def ntype(n): return G.nodes[n].get("type")

    # Precompute start nodes (target type)
    node_types = nx.get_node_attributes(G, "type")
    start_nodes = [n for n, t in node_types.items() if t == target_type]

    from collections import defaultdict
    neighbors_by_type = {n: defaultdict(list) for n in G.nodes}
    for n in G.nodes:
        for nbr in G.neighbors(n):
            nbr_type = node_types[nbr]
            neighbors_by_type[n][nbr_type].append(nbr)

    paths = []
    meta_len = len(meta_types)

    for v in start_nodes:
        for _ in range(r):
            path = []
            curr = v
            step_count = 0

            # We’ll advance along meta_types cycle; ensure current matches its position
            # Find position in meta_types where type == ntype(curr)
            # (If multiple, choose the first occurrence; then we’ll cycle.)
            try:
                pos = meta_types.index(node_types[curr])
            except ValueError:
                # If the current node type isn't in meta_types, skip this start node
                continue

            while step_count < wl:
                # (Type filtering) — append only if current node is of the target type
                if node_types[curr] == target_type:
                    path.append(curr)
                    step_count += 1
                    if step_count >= wl:
                        break

                # Move to next required type in the meta-path
                next_pos = (pos + 1) % meta_len
                required_type = meta_types[next_pos]

                # Candidate neighbors of the required type
                candidates = neighbors_by_type[curr].get(required_type, [])
                if not candidates:
                    break  # dead end

                curr = random.choice(candidates)
                pos = next_pos  # advance position in the meta path

            if len(path) >= 2:  # keep non-trivial sequences # remove as this breaks later, and also they don't mention it in the article
                paths.append(path)

    
    # window = ns (neighborhood size), vector_size = d (embedding dim)
    # embeddings = train_skipgram(paths, emb_dim=d, window_size=ns, neg_samples=5, epochs=5, lr=0.001)


    embeddings = train_skipgram_lightning(paths, emb_dim=d, window_size=ns, neg_samples=5, epochs=epochs, lr=lr)

    return paths, embeddings