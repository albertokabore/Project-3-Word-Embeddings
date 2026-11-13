# Assignment Checklist

**Albert Kabore**

PhD student in AI

| Requirement | Report | Notebook |
|---|---|---|
| Introduction | Section 1 | Section 1 |
| Data cleaning | Section 3 | Section 3 |
| TF-IDF document representation | Section 4 | Section 4 |
| Feature means and centering | Sections 5.1, 6, and 8.1 | Sections 5.1 and 5.2 |
| Covariance matrix | Sections 5.2, 6, and 8.1 | Section 5.3 |
| Sorted eigenvalues and matching eigenvectors | Sections 5.3, 6, and 8.2 | Section 5.4 |
| Reduction preserving variance | Sections 5.4, 6, and 8.3 | Section 5.5 |
| Plain-English explanation | Section 7 | Text beside each calculation |
| Observations: 100 words in bullet points | Section 9 | Section 7 |
| Final insights and recommendations: 300 words combined, in separate bullet sections | Sections 10 and 11 | Sections 8 and 9 |
| Conclusion | Section 12 | Section 10 |
| Clear headings and mathematical explanations | Sections 1–14 | Sections 1–12 |
| External sources cited | Section 13 | Section 11 |
| Two-sentence AI use disclosure at the end | Section 14 | Section 12 |

The reported values are calculated from the local dataset. The report code and notebook use a 90% variance target, retaining 553 of 1,000 components. The Word and PDF reports are generated from the Markdown source.

## Output Analysis

The report now interprets the measured means, variances, covariances, and component
coefficients. It checks the 552/553 boundary for the selected variance threshold,
quantifies reconstruction loss against the mean-only baseline, and compares PCA
with one fixed-seed random orthonormal basis of the same dimension. These results
are computed by the notebook and saved in `results/output_analysis.json`.

The supplied assignment requires the PCA computations and explanations above; it
does not prescribe a 90% threshold or require classification. The threshold and
additional reconstruction comparison are choices used to analyze this dataset.
