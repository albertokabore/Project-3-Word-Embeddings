# Project 3: Word Embeddings

**Albert Kabore**

PhD student in AI

TF-IDF document vectors and PCA on the BBC News Archive. The notebook and report calculate the feature means, centered data, covariance matrix, sorted eigenvalues and eigenvectors, and projection retaining at least 90% of variance.

## Submission files

- [Notebook](Project3_Word_Embeddings.ipynb)
- [Project 3: Word Embeddings](<Project 3 - Word Embeddings.docx>)

## Run the notebook

Run from the project root with Python 3.10 or later:

```bash
pip install -r requirements.txt
pip install -e .
jupyter notebook Project3_Word_Embeddings.ipynb
```

The input data is included in `data/`. Source code is in `src/word_embeddings/`.
The notebook writes the measured means, covariances, component coefficients, and
reconstruction comparisons to `results/output_analysis.json`.

## Checks

```bash
python -m pytest
```
