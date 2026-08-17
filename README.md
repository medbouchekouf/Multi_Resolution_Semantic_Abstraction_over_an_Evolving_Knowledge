# Multi_Resolution_Semantic_Abstraction_over_an_Evolving_Knowledge
"Semantic Level of Detail over a > Temporal Knowledge Hypergraph


<div align="center">

# 🌌 Multi-Resolution Semantic Abstraction over an Evolving Knowledge Base

**A living, zoomable map of your knowledge — from raw facts to global themes, always up to date.**

[![Stars](https://img.shields.io/github/stars/your-org/mrsa?style=social)](https://github.com/your-org/mrsa/stargazers)
[![Forks](https://img.shields.io/github/forks/your-org/mrsa?style=social)](https://github.com/your-org/mrsa/network/members)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Paper](https://img.shields.io/badge/paper-coming%20soon-orange.svg)](#citation)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

[✨ Features](#-key-features) •
[🏗 Architecture](#-architecture) •
[🚀 Quick Start](#-quick-start) •
[📊 Evaluation](#-evaluation) •
[📖 Citation](#-citation)

</div>

---

## 🧭 Overview

Large knowledge bases — Wikidata-style graphs, enterprise KBs, streaming fact stores — are **never static**: entities appear, relations are corrected, entire subtopics drift. Flat retrieval forces a painful trade-off: drowning-level detail, or lossy one-shot summaries.

This project takes a different path — **multi-resolution semantic abstraction**:

> 🧩 **Cluster** facts into semantically coherent communities at several granularities
> 📝 **Abstract** each community into a natural-language + embedding summary
> 🌳 **Organize** these nodes into a resolution tree — raw triples up to domain themes
> ♻️ **Maintain** the tree *incrementally*: when the KB evolves, only affected branches are re-abstracted
> 🔍 **Query** at any resolution — drill down for evidence, roll up for gist

The result: a **living, zoomable map** of the knowledge base that stays fresh without full recomputation.

---

## ✨ Key Features

| | |
|---|---|
| 🌲 **Multi-resolution hierarchy** | L0 raw triples → L1 entity/event clusters → L2 topic abstractions → L3 global themes, with cross-level provenance links |
| ♻️ **Incremental evolution** | Insertions, deletions, and corrections trigger *localized* subtree updates — never a global rebuild |
| 🧠 **Hybrid semantic nodes** | Every node carries an LLM-generated summary **and** a dense embedding — symbolic and neural access |
| 🎯 **Resolution-aware retrieval** | Queries route to the cheapest sufficient level, with automatic drill-down when confidence is low |
| ⏳ **Temporal awareness** | Versioned abstractions — ask *"what did this topic look like at time t?"* |
| 🔌 **Pluggable backends** | RDF/SPARQL, Neo4j property graphs, or plain JSONL fact streams; swappable LLM & embedding providers |

---

## 🏗 Architecture

```
                ┌──────────────────────────────┐
                │        🔍 Query Router        │   resolution-aware retrieval
                └──────────────┬───────────────┘
                               │
        ┌──────────────────────▼──────────────────────┐
        │     🌳 Multi-Resolution Abstraction Tree     │
        │   L3 themes ── L2 topics ── L1 clusters ── L0│
        └───────▲───────────────────────────▲─────────┘
                │ abstraction               │ provenance
        ┌───────┴───────────┐      ┌────────┴─────────┐
        │  📝 Abstractor     │      │  ♻️ Evolution     │  diff detection,
        │  (LLM + embedder)  │      │  Monitor         │  impact analysis
        └───────▲───────────┘      └────────▲─────────┘
                │                           │ change stream
        ┌───────┴───────────────────────────┴─────────┐
        │   🗄 Evolving Knowledge Base                 │
        │   (RDF / property graph / JSONL stream)      │
        └─────────────────────────────────────────────┘
```

**Pipeline stages**

1. **📥 Ingestion** — normalize facts into `(subject, relation, object, timestamp, provenance)` quintuples
2. **🧩 Community detection** — embedding-assisted graph clustering (Leiden / HDBSCAN) → L1 communities
3. **📝 Abstraction** — LLM summarizes each community; nodes are recursively clustered & re-summarized → L2/L3
4. **♻️ Evolution monitor** — KB changes map to impacted communities; only *dirty* subtrees are re-abstracted
5. **🔍 Query routing** — top-down embedding match: answer at the coarsest sufficient level, expand provenance for evidence

---

## 📦 Installation

```bash
git clone https://github.com/your-org/mrsa.git
cd mrsa

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Optional backends:

```bash
pip install ".[neo4j]"     # 🕸 property-graph backend
pip install ".[rdf]"       # 🔗 RDFLib / SPARQL endpoint backend
pip install ".[openai]"    # 🤖 OpenAI-compatible LLM + embeddings
```

---

## 🚀 Quick Start

```python
from mrsa import KnowledgeBase, AbstractionTree, EvolutionMonitor

kb = KnowledgeBase.from_jsonl("data/facts.jsonl")

tree = AbstractionTree(kb, levels=3, llm="gpt-4o-mini", embedder="bge-large")
tree.build()                      # 🌳 initial multi-resolution build

print(tree.summarize(level=3))    # 🌐 coarse themes over the whole KB

# The KB evolves...
monitor = EvolutionMonitor(tree)
monitor.apply_delta("data/facts_delta.jsonl")   # ♻️ incremental re-abstraction

# Query at any resolution
answer = tree.query("How has the EV battery supply chain changed?",
                    resolution="auto")
print(answer.text, answer.evidence)
```

---

## 🛠 Usage

<details open>
<summary><b>🌳 Building the abstraction hierarchy</b></summary>

```bash
python -m mrsa.build \
    --kb data/facts.jsonl \
    --levels 3 \
    --cluster-algo leiden \
    --out checkpoints/tree_v1.pkl
```
</details>

<details>
<summary><b>♻️ Incremental updates on an evolving KB</b></summary>

```bash
python -m mrsa.evolve \
    --tree checkpoints/tree_v1.pkl \
    --delta data/facts_delta.jsonl \
    --strategy local-reabstract \
    --out checkpoints/tree_v2.pkl
```

`--strategy` options: `local-reabstract` (default, cheapest) · `recluster-dirty` · `full-rebuild` (baseline)
</details>

<details>
<summary><b>🔍 Multi-resolution querying</b></summary>

```bash
python -m mrsa.query \
    --tree checkpoints/tree_v2.pkl \
    --q "What are the main research trends in this corpus?" \
    --resolution 2          # 0=raw triples … 3=global themes, or "auto"
```
</details>

---

## ⚙️ Configuration

All knobs live in `configs/default.yaml`:

| 🔧 Key | Default | Description |
|---|---|---|
| `levels` | `3` | Number of abstraction levels above raw facts |
| `cluster.algo` | `leiden` | `leiden`, `hdbscan`, or `louvain` |
| `cluster.resolution` | `[0.5, 1.0, 2.0]` | Per-level clustering granularity |
| `abstractor.max_facts_per_node` | `200` | Facts folded into one summary prompt |
| `evolution.impact_radius` | `2` | Graph hops for change-impact analysis |
| `evolution.dirty_threshold` | `0.15` | Changed-fact fraction that marks a community dirty |
| `query.confidence_threshold` | `0.7` | Below this, drill down one level |

---

## 📚 Datasets

- 🧪 **Synthetic evolving KB** — `data/synthetic/`, scripted fact streams with controlled drift
- 🌐 **Temporally sliced Wikidata** — monthly dumps, 2019–2024 (`scripts/prepare_wikidata.py`)
- 📰 **Event/news fact streams** — GDELT-derived quintuples (`scripts/prepare_gdelt.py`)

---

## 📊 Evaluation

```bash
python -m mrsa.eval --tree <ckpt> --suite all
```

| Metric | What it measures | Target |
|---|---|---|
| ⚡ **Freshness** | Abstraction latency per KB delta vs. full rebuild | ≥10× faster, <1% quality loss |
| 📝 **Abstraction quality** | Faithfulness & coverage per level (G-Eval / human) | — |
| 🎯 **Retrieval** | Answer accuracy & evidence recall vs. flat RAG | — |
| 🧊 **Stability** | Tree churn per update (lower = better) | — |

---

## 🗂 Project Structure

```
mrsa/
├── 📁 kb/          # knowledge base adapters (jsonl, rdf, neo4j)
├── 📁 cluster/     # community detection at each resolution
├── 📁 abstract/    # LLM summarization + embedding of communities
├── 📁 tree/        # hierarchy, versioning, provenance
├── 📁 evolve/      # change detection, impact analysis, incremental update
├── 📁 query/       # resolution-aware router & answer composition
└── 📁 eval/        # benchmarks and metrics
configs/            # ⚙️ YAML configs
scripts/            # 🛠 dataset preparation
data/               # 📚 sample fact streams
```

---

## 🤝 Contributing

Contributions are welcome! 🎉 Please open an issue to discuss your idea, then submit a PR against `main`. See [CONTRIBUTING.md](CONTRIBUTING.md) for the dev setup.

---

## 📖 Citation

```bibtex
@article{mrsa2026,
  title  = {Multi-Resolution Semantic Abstraction over an Evolving Knowledge Base},
  author = {Anonymous},
  year   = {2026},
  note   = {Manuscript in preparation}
}
```

---

## 📄 License

Apache-2.0 — see [LICENSE](LICENSE).

---

<div align="center">

## ⭐ Star History

**If this project helps you, please consider giving it a star — it means a lot! 🌟**

[![Star History Chart](https://api.star-history.com/svg?repos=your-org/mrsa&type=Date)](https://star-history.com/#your-org/mrsa&Date)

[![Stargazers repo roster](https://reporoster.com/stars/your-org/mrsa)](https://github.com/your-org/mrsa/stargazers)



</div>
