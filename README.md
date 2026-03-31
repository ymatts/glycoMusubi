# glycoMusubi

**Integrated knowledge graph and hierarchical benchmark for glycan prediction from protein sequence.**

*Musubi (結) — Japanese for "binding" or "tying together," reflecting both the covalent attachment of glycans to proteins and the integration of six heterogeneous databases into a unified knowledge graph.*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

glycoMusubi is a bioinformatics platform that constructs a heterogeneous knowledge graph integrating glycobiology data from six public databases, and provides a hierarchical benchmark for glycan prediction from protein sequence context. The platform reveals a quantitative determination boundary: protein sequence reliably predicts the structural class of glycosylation, but exact glycan identity requires additional cellular information.

### Key features

- **Knowledge graph**: 78,263 nodes (10 entity types) and 2.5M edges (14 relation types) from GlyGen, GlyTouCan, UniProt, ChEMBL, PTMCode, and Reactome
- **Hierarchical benchmark**: Six tasks of increasing specificity, from N-linked site classification (F1 = 0.924) to exact glycan identification (MRR = 0.066)
- **KG embedding model**: GlycoKGNet with modality-specific encoders (GlycanTreeEncoder, ESM-2, PubMedBERT)
- **Cell-type analysis**: Integration of Tabula Sapiens single-cell enzyme expression (748 cell types, 339 enzymes)
- **Interactive Shiny app**: Seven-module web interface for KG exploration, prediction, and visualisation

## Quick start

```bash
# Clone
git clone https://github.com/ymatts/glycoMusubi.git
cd glycoMusubi

# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the full pipeline
python scripts/pipeline.py

# Launch the Shiny app (Docker)
cd shiny_app && docker compose up -d
# Open http://localhost:3838
```

## Documentation

| Document | Language | Description |
|----------|----------|-------------|
| [Vignettes & Technical Guide](docs/book-vignette/) | EN | Glycobiology primer, architecture, methods |
| [ヴィネット＆技術ガイド](docs/book-vignette-ja/) | JA | 日本語版技術ガイド |
| [Shiny App Guide](docs/book-shiny/) | EN | Module-by-module user guide |
| [Shinyアプリガイド](docs/book-shiny-ja/) | JA | 日本語版アプリガイド |

## Pipeline

```
DOWNLOAD → CLEAN → BUILD → VALIDATE
   ↓         ↓       ↓        ↓
data_raw/ → data_clean/ → kg/ → logs/
```

Six data sources are integrated through an automated four-stage ETL pipeline with schema enforcement, identifier normalisation, and validation with optional auto-fix.

## Hierarchical benchmark

| Task | Candidates | Metric | Value |
|------|-----------|--------|-------|
| N-linked site classification | Binary | F1 | 0.924 |
| Site ranking | Per protein | Recall@3 | 0.683 |
| Glycosylation type | 4 classes | Accuracy | 0.844 |
| Structural family | 8 classes | Top-2 Acc | 0.858 |
| Structural cluster (K=20) | 20 clusters | H@5 | 0.786 |
| Exact glycan (site-level) | 2,357 glycans | MRR | 0.066 |

The 10-fold gap between cluster-level and exact-ID performance reveals a **determination boundary**: protein sequence predicts structural class reliably, but exact glycan identity requires cellular context.

## Shiny app

Seven interactive modules:

1. **KG Explorer** — Search and visualise knowledge graph subgraphs
2. **Glycan Predictor** — N-linked site and structure class prediction
3. **Glycan Retrieval** — Protein-to-glycan similarity ranking
4. **Cell-type Expression** — Enzyme expression across 748 cell types
5. **Benchmarks** — Performance dashboard for all experiments
6. **Embedding Explorer** — UMAP visualisation of learned representations
7. **Hierarchy Browser** — Glycan structural taxonomy tree

## Citation

```bibtex
@article{matsui2026glycomusubi,
  title={Hierarchical glycan prediction from protein sequence reveals a determination boundary at the structural-class level},
  author={Matsui, Yusuke},
  journal={TBD,
  year={2026}
}
```

## Contact

Yusuke Matsui
- Biomedical and Health Informatics Unit, Nagoya University Graduate School of Medicine
- Institute for Glyco-core Research (iGCORE), Nagoya University
- Email: matsui.yusuke.d4@f.mail.nagoya-u.ac.jp

## License

MIT License. See [LICENSE](LICENSE) for details.
