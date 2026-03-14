# Cryptocurrency Recommender System using Knowledge Graph Convolutional Networks (KGCN)

> **Published Research** · [IEEE Xplore — 2024 12th International Conference of Information and Communication Technology (ICoICT)](https://ieeexplore.ieee.org/document/10698135)

---

## Background & Motivation

The cryptocurrency market has grown dramatically, but investors especially newcomers face a daunting challenge: thousands of coins exist, each with its own technology, use case, and community, yet most recommendation tools rely purely on price signals or simple collaborative filtering that ignores rich contextual relationships between assets.

This project addresses that gap by adapting **Knowledge Graph Convolutional Networks (KGCN)** — originally proposed by Wang et al. (WWW 2019) for movie and music recommendation — to the cryptocurrency domain. The key insight is that a knowledge graph can capture semantic relationships between cryptocurrencies (e.g., *"Bitcoin uses Proof-of-Work"*, *"Ethereum supports smart contracts"*), enabling a model that recommends coins not only based on what users have rated, but also based on the **structural properties** and **contextual attributes** of the assets themselves.

The result is a graph-based deep learning recommender that:
- **Aggregates multi-hop neighborhood information** from a crypto knowledge graph
- **Encodes user preference** through learned user embeddings
- **Scores user–coin affinity** via inner-product similarity in embedding space
- **Achieves strong CTR-prediction performance** (AUC & F1) evaluated on a curated cryptocurrency rating dataset

This project was completed as an undergraduate final-year thesis and subsequently accepted and published at **2024 12th ICoICT**.

---

## How It Works

The system builds on the KGCN architecture:

1. **Knowledge Graph Construction** — Cryptocurrency entities and their attributes (consensus mechanism, category, blockchain platform, etc.) are modelled as a knowledge graph of `(head, relation, tail)` triples.
2. **Entity & Relation Embedding** — Each entity and relation in the KG is represented as a learnable dense vector.
3. **Graph Convolution with User-Aware Attention** — For each item (coin), the model iteratively aggregates representations from its KG neighbors. The aggregation is **user-specific**: neighbor relations are weighted by their dot-product similarity with the current user's embedding, so different users attend to different aspects of the graph.
4. **Score Prediction** — The final item embedding and the user embedding are combined via inner product and passed through a sigmoid function to produce a recommendation score.
5. **Training** — The model is optimised with binary cross-entropy loss + L2 regularisation using the Adam optimiser.

Three aggregation strategies are supported:
| Aggregator | Description |
|---|---|
| `sum` | Element-wise sum of self and aggregated neighbor vectors (default) |
| `concat` | Concatenation of self and aggregated neighbor vectors |
| `neighbor` | Use only the aggregated neighbor vector |

![KGCN Framework](framework.png)

---

## Repository Structure

```
crypto-kgcn/
│
├── data/
│   ├── crypto/                        # Cryptocurrency dataset (primary)
│   │   ├── ratings.csv                # Raw user–coin interaction data (ratings 1–5)
│   │   ├── item_index2entity_id.txt   # Maps coin indices → KG entity IDs
│   │   ├── kg.txt                     # Raw knowledge graph triples (head, relation, tail)
│   │   ├── ratings_final.txt          # Processed binary rating file
│   │   ├── ratings_final.npy          # Cached NumPy version of processed ratings
│   │   ├── kg_final.txt               # Processed KG with re-indexed entity/relation IDs
│   │   └── kg_final.npy               # Cached NumPy version of processed KG
│   │
│   ├── movie/                         # MovieLens dataset (baseline comparison)
│   │   ├── item_index2entity_id.txt
│   │   └── kg.txt
│   │
│   └── music/                         # Last.FM dataset (baseline comparison)
│       ├── item_index2entity_id.txt
│       ├── kg.txt
│       └── user_artists.dat
│
├── src/
│   ├── preprocess.py   # Data preprocessing: converts raw ratings & KG into model-ready format
│   ├── data_loader.py  # Loads processed data, constructs adjacency matrices for the KG
│   ├── aggregators.py  # Sum, Concat, and Neighbor aggregator implementations (TensorFlow)
│   ├── model.py        # KGCN model: embedding layers, graph convolution, score computation
│   ├── train.py        # Training loop, CTR evaluation (AUC, F1), and top-K evaluation
│   └── main.py         # Entry point: argument parsing and orchestration
│
├── framework.png       # Architecture diagram
├── LICENSE
└── README.md
```

---

## Requirements

- Python 3.6+
- TensorFlow 1.x (the model uses `tf.contrib` APIs)
- NumPy
- scikit-learn

Install dependencies:
```bash
pip install tensorflow==1.15 numpy scikit-learn
```

---

## Running the Code

### 1 · Preprocess the data

Run `preprocess.py` from inside the `src/` directory. This reads the raw ratings and KG files and generates the `*_final.txt` / `*_final.npy` files the model needs.

```bash
cd src
python preprocess.py -d crypto
```

Replace `crypto` with `movie` or `music` to preprocess those datasets instead.

> **Note for MovieLens:** The raw rating file is too large to include in this repo. Download it first:
> ```bash
> wget http://files.grouplens.org/datasets/movielens/ml-20m.zip
> unzip ml-20m.zip
> mv ml-20m/ratings.csv ../data/movie/
> ```

---

### 2 · Configure hyperparameters

Open `src/main.py` and ensure the correct dataset block is **uncommented**. By default, the `crypto` configuration is active:

```python
# crypto  ← currently active
parser.add_argument('--dataset',              default='crypto')
parser.add_argument('--aggregator',           default='sum')
parser.add_argument('--n_epochs',             default=10)
parser.add_argument('--neighbor_sample_size', default=4)
parser.add_argument('--dim',                  default=32)
parser.add_argument('--n_iter',               default=2)
parser.add_argument('--batch_size',           default=65536)
parser.add_argument('--l2_weight',            default=1e-7)
parser.add_argument('--lr',                   default=2e-2)
```

Comment out the active block and uncomment the `music` block to switch datasets.

---

### 3 · Train the model

```bash
cd src
python main.py
```

The script will print per-epoch AUC and F1 metrics on the training, validation, and test splits:

```
epoch 0    train auc: 0.xxxx  f1: 0.xxxx    eval auc: 0.xxxx  f1: 0.xxxx    test auc: 0.xxxx  f1: 0.xxxx
...
```

---

## Key Hyperparameters

| Parameter | Default (Crypto) | Description |
|---|---|---|
| `--aggregator` | `sum` | Aggregation strategy (`sum`, `concat`, `neighbor`) |
| `--n_epochs` | `10` | Number of training epochs |
| `--neighbor_sample_size` | `4` | Number of KG neighbors sampled per entity |
| `--dim` | `32` | Dimensionality of user, entity, and relation embeddings |
| `--n_iter` | `2` | Number of graph convolution hops |
| `--batch_size` | `65536` | Mini-batch size |
| `--l2_weight` | `1e-7` | L2 regularisation coefficient |
| `--lr` | `2e-2` | Adam learning rate |

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{luthfi2024kgcn,
  title     = {Cryptocurrency Recommender System Using Knowledge Graph Convolutional Networks},
  booktitle = {2024 12th International Conference of Information and Communication Technology},
  year      = {2024},
  publisher = {IEEE},
  url       = {https://ieeexplore.ieee.org/document/10698135}
}
```

---

## Acknowledgements

This work builds on the original KGCN paper:

> Hongwei Wang, Miao Zhao, Xing Xie, Wenjie Li, Minyi Guo.  
> **Knowledge Graph Convolutional Networks for Recommender Systems.**  
> *The Web Conference (WWW 2019)*  
> [ACM DL](https://dl.acm.org/citation.cfm?id=3313417) · [arXiv](https://arxiv.org/abs/1904.12575)
