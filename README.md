# Interview_BioAI4LCSM
# Breast Cancer Subtype Analysis with LLMs, Multi-Omics & Machine Learning

This repository contains a modular pipeline integrating clinical data, gene expression analysis, classical machine learning, and large language models (LLMs) for breast cancer subtype classification and interpretability.

## 🔬 Project Overview

We aim to systematically investigate breast cancer heterogeneity across subtypes (LumA, LumB, Basal, Her2, Normal) using:

- **Gene expression data (microarray/RNA-seq)**
- **Clinical variables (ER/PR/HER2/Ki67, tumor size, survival, etc.)**
- **Supervised and unsupervised learning techniques**
- **LLM-driven literature mining for gene-function and method identification**

---

## 📁 Repository Structure

| File | Description |
|------|-------------|
| `data_process.ipynb` | Preprocessing of raw expression and clinical datasets. Includes filtering, normalization, and replicate handling. |
| `supervised_pam50_subtype_analysis.ipynb` | Classical and deep learning models for supervised classification of PAM50 subtypes. Includes evaluation on raw and reduced data. |
| `unsupervised_pam50_subtype_analysis.ipynb` | Dimensionality reduction (PCA, t-SNE, UMAP, autoencoders) + clustering methods (KMeans, GMM, Spectral) for subtype discovery. |
| `clinical_analysis.ipynb` | Statistical analysis of clinical variables across subtypes. Includes chi-square, ANOVA, logistic regression, and survival modeling. |
| `LLM_analysis.py` | LLM-based pipeline for mining biomedical techniques from PubMed abstracts and associating genes with breast cancer subtypes. |
| `pam50_analysis_detailed.nb.html` | Rendered version of the PAM50 subtype analysis notebook. Ideal for sharing and viewing results without executing code. |
| `README.md` | This file. Summary of modules, tools, and insights. |


---

## ⚙️ Methods Used

### 💡 Subtype Classification

- **Supervised**: Logistic Regression, Random Forest, Feedforward Neural Network (FFNN)
- **Unsupervised**: KMeans, Hierarchical Clustering, GMM, Spectral Clustering
- **Dimensionality Reduction**: PCA, Autoencoder, t-SNE, UMAP

### 🧠 LLM Integration

- PubMed search via Biopython (Entrez)
- Method extraction with `falcon-rw-1b`, `BioGPT`, and instruction-following LLMs
- Technique matching using keyword vocabularies + fuzzy logic

### 🧬 Gene-Level Interpretation

- limma for differential expression
- Subtype-wise up/downregulated gene sets
- Pathway annotations via MyGene.info API

### 📈 Clinical Modeling

- Statistical testing (Chi-square, ANOVA, Kruskal-Wallis)
- Survival analysis (Kaplan-Meier, Cox models)
- Treatment prediction models (logistic regression, FFNN)

---

## 📊 Sample Results

- Supervised models achieved **up to 87% accuracy** in subtype classification.
- Unsupervised clustering yielded low ARI/NMI, indicating challenges in latent subtype discovery.
- LLM-assisted literature mining extracts context-aware insights on gene roles and ML techniques from PubMed abstracts.
- Clinical variable significance aligns with known subtype biology (e.g., LumA → high ER, low Ki67; Basal → triple-negative, high proliferation).
- CNAttention model achieved **~95% accuracy**, highlighting the power of attention-based MIL in modeling gene expression.

---

## 🧠 Dependencies

- `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
- `torch`, `transformers`
- `Bio.Entrez` (biopython)
- `requests`, `re`, `time`

---

## 📌 Citation

If you find this repository useful, please consider citing the associated analysis or methods.

---

## 🙋 Acknowledgments

This project builds on ideas from:

- PAM50 subtype classification
- limma for differential analysis
- Open-source LLMs like `falcon-rw-1b`, `BioGPT`
- MyGene.info API
- METABRIC and TCGA datasets

---

## 📬 Contact

For questions or collaboration, feel free to reach out to the author(s).
