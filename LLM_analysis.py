#%%
import pandas as pd
from Bio import Entrez
import time
import requests

# === Set parameters ===
Entrez.email = "your.email@example.com"  # Set your email (required by Entrez API)
MAX_RESULTS = 5  # Maximum number of publications to return for each gene

# Example gene input list: with optional subtype and direction
gene_inputs = [
    {"gene": "FOXA1", "subtype": "LumA", "direction": "UP"},
    {"gene": "SOX11", "subtype": "Her2", "direction": "DOWN"},
    {"gene": "ERBB2", "subtype": "Her2"}
]

# === Function to search PubMed ===
def search_pubmed(query, max_results=5):
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]

# Fetch metadata for a given PubMed ID
def fetch_pubmed_metadata(pmid):
    handle = Entrez.efetch(db="pubmed", id=pmid, rettype="medline", retmode="text")
    text = handle.read()
    handle.close()

    # Simple parser
    title = "NA"
    abstract = "NA"
    journal = "NA"
    year = "NA"
    for line in text.split("\n"):
        if line.startswith("TI  -"):
            title = line.replace("TI  - ", "").strip()
        elif line.startswith("AB  -"):
            abstract = line.replace("AB  - ", "").strip()
        elif line.startswith("JT  -"):
            journal = line.replace("JT  - ", "").strip()
        elif line.startswith("DP  -"):
            year = line.replace("DP  - ", "").strip().split(" ")[0]
    return title, abstract, journal, year

# === Function to extract supporting sentence from abstract ===
def extract_support_sentence(abstract, gene):
    if abstract == "NA":
        return "NA"
    gene_lower = gene.lower()
    sentences = abstract.split(". ")
    keyword_hits = []

    for sent in sentences:
        s = sent.lower()
        if gene_lower in s and ("breast cancer" in s or "subtype" in s or "pathway" in s or "tumor" in s):
            keyword_hits.append(sent.strip())

    if keyword_hits:
        return keyword_hits[0]  # Return the most relevant sentence
    return "NA"

# === Function to query gene-related pathways ===
def query_pathways(gene):
    try:
        r = requests.get(f"https://mygene.info/v3/gene/{gene}?fields=pathway")
        if r.status_code == 200:
            data = r.json()
            pathways = []

            # Reactome-style structure
            pw_data = data.get("pathway")
            if isinstance(pw_data, dict):
                for k, v in pw_data.items():
                    if isinstance(v, list):
                        pathways.extend([p["name"] for p in v if "name" in p])
                    elif isinstance(v, dict):
                        pathways.append(v.get("name", ""))
            elif isinstance(pw_data, list):
                pathways.extend([p.get("name", "") for p in pw_data if "name" in p])

            pathways = [p for p in pathways if p]
            return "; ".join(set(pathways)) if pathways else "NA"
    except:
        pass
    return "NA"

# === Main execution loop ===
results = []

for item in gene_inputs:
    gene = item["gene"]
    subtype = item.get("subtype", "NA")
    direction = item.get("direction", "NA")
    query = f"{gene} breast cancer"

    try:
        pmids = search_pubmed(query, MAX_RESULTS)
        time.sleep(0.3)
        for pmid in pmids:
            title, abstract, journal, year = fetch_pubmed_metadata(pmid)
            support = extract_support_sentence(abstract, gene)
            pathway = query_pathways(gene)
            results.append({
                "Gene": gene,
                "Subtype": subtype,
                "Direction": direction,
                "PMID": pmid,
                "Title": title,
                "Journal": journal,
                "Year": year,
                "Abstract": abstract,
                "Support": support,
                "Pathway": pathway
            })
            time.sleep(0.5)
    except Exception as e:
        print(f"Error for gene {gene}: {e}")

# === Output results as a DataFrame ===
df_results = pd.DataFrame(results)
print(df_results.head())

# === Optional: Save as Excel or CSV file ===
# df_results.to_excel("gene_lit_summary.xlsx", index=False)
# df_results.to_csv("gene_lit_summary.csv", index=False)

#%%
import pandas as pd
import time
import requests
from Bio import Entrez

# === Set Entrez API parameters ===
Entrez.email = "your.email@example.com"  # Required email for NCBI Entrez access
MAX_RESULTS = 5  # Maximum number of articles per gene to retrieve

# === Input gene list (with optional subtype and expression direction) ===
gene_inputs = [
    {"gene": "FOXA1", "subtype": "LumA", "direction": "UP"},
    {"gene": "SOX11", "subtype": "Her2", "direction": "DOWN"},
    {"gene": "ERBB2", "subtype": "Her2"}
]

# === PubMed search function ===
def search_pubmed(query, max_results=5):
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
        record = Entrez.read(handle)
        handle.close()
        return record["IdList"]
    except Exception as e:
        print(f"[Search error] {query}: {e}")
        return []

# === Fetch publication metadata (title, abstract, journal, year) using XML ===
def fetch_pubmed_metadata(pmid):
    try:
        handle = Entrez.efetch(db="pubmed", id=pmid, rettype="xml", retmode="xml")
        records = Entrez.read(handle)
        handle.close()
        article = records["PubmedArticle"][0]["MedlineCitation"]["Article"]
        title = article.get("ArticleTitle", "NA")
        abstract = " ".join(article.get("Abstract", {}).get("AbstractText", ["NA"]))
        journal = article.get("Journal", {}).get("Title", "NA")
        year = article.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {}).get("Year", "NA")
        return title, abstract, journal, year
    except Exception as e:
        print(f"[Parsing error] PMID {pmid}: {e}")
        return "NA", "NA", "NA", "NA"

# === Extract supporting sentence mentioning the gene and relevant biological terms ===
def extract_support_sentence(abstract, gene):
    if abstract == "NA":
        return "NA"
    gene_lower = gene.lower()
    sentences = abstract.split(". ")

    strong_keywords = [
        "upregulated", "downregulated", "associated", "correlated", "activates",
        "represses", "marker", "expressed", "expression", "prognosis", "tumor"
    ]

    # Search for a sentence that contains the gene and a strong keyword
    for sent in sentences:
        s = sent.lower()
        if gene_lower in s and any(k in s for k in strong_keywords):
            return sent.strip()

    # Fallback: return the first sentence that mentions the gene
    for sent in sentences:
        if gene_lower in sent.lower():
            return sent.strip()

    return "NA"

# === Query pathways associated with a gene symbol via MyGene.info ===
def query_pathways(gene_symbol):
    try:
        # First get Entrez gene ID
        r1 = requests.get(f"https://mygene.info/v3/query?q={gene_symbol}&species=human&fields=entrezgene")
        hits = r1.json().get("hits", [])
        if hits and "entrezgene" in hits[0]:
            gene_id = hits[0]["entrezgene"]
            # Then get pathway data
            r2 = requests.get(f"https://mygene.info/v3/gene/{gene_id}?fields=pathway")
            pw_data = r2.json().get("pathway", {})
            pathways = []
            if isinstance(pw_data, dict):
                for v in pw_data.values():
                    if isinstance(v, list):
                        pathways.extend([p.get("name") for p in v if p.get("name")])
                    elif isinstance(v, dict):
                        pathways.append(v.get("name", ""))
            elif isinstance(pw_data, list):
                pathways.extend([p.get("name") for p in pw_data if p.get("name")])
            return "; ".join(set(pathways)) if pathways else "NA"
    except Exception as e:
        print(f"[Pathway query error] {gene_symbol}: {e}")
    return "NA"

# === Main processing loop ===
results = []

for item in gene_inputs:
    gene = item["gene"]
    subtype = item.get("subtype", "NA")
    direction = item.get("direction", "NA")
    query = f"{gene}[Title/Abstract] AND breast cancer[Title/Abstract]"

    if subtype != "NA":
        query += f" AND {subtype}[Title/Abstract]"

    print(f"🔍 Processing gene: {gene} | Query: {query}")

    pmids = search_pubmed(query, MAX_RESULTS)
    time.sleep(1)  # Avoid overwhelming PubMed servers

    for pmid in pmids:
        title, abstract, journal, year = fetch_pubmed_metadata(pmid)
        support = extract_support_sentence(abstract, gene)
        pathway = query_pathways(gene)
        results.append({
            "Gene": gene,
            "Subtype": subtype,
            "Direction": direction,
            "PMID": pmid,
            "Title": title,
            "Journal": journal,
            "Year": year,
            "Support": support,
            "Pathway": pathway
        })
        time.sleep(1)  # Throttle requests

# === Export to DataFrame ===
df_results = pd.DataFrame(results)
print(df_results.head())

# === Optional: Save to Excel file ===
# df_results.to_excel("optimized_gene_lit_summary.xlsx", index=False)

#%%
import pandas as pd
import time
from Bio import Entrez
from transformers import pipeline
from collections import Counter

# ==== Configuration ====
Entrez.email = "your@email.com"  # Required for using Entrez API
QUERY = "breast cancer subtype classification"
MAX_RESULTS = 10  # You can increase this to retrieve more articles

# ==== Technical keywords (used for fuzzy matching after LLM extraction) ====
TECH_KEYWORDS = [
    "support vector machine", "svm", "random forest", "decision tree", "naive bayes",
    "k-nearest neighbor", "xgboost", "logistic regression", "lasso",
    "hierarchical clustering", "k-means", "consensus clustering", "pca", "t-sne", "umap",
    "autoencoder", "cnn", "convolutional neural network", "graph neural network", "gcn",
    "deep learning", "multi-omics integration", "pam50", "gene expression profiling"
]

# ==== Load a lightweight open-source language model ====
llm = pipeline("text-generation", model="tiiuae/falcon-rw-1b", max_new_tokens=128)

# ==== Search PubMed ====
def search_pubmed(query, max_results=10):
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]

# ==== Fetch metadata: title, abstract, journal, and year ====
def fetch_metadata(pmid):
    handle = Entrez.efetch(db="pubmed", id=pmid, rettype="medline", retmode="text")
    text = handle.read()
    handle.close()
    title = abstract = journal = year = "NA"
    for line in text.split("\n"):
        if line.startswith("TI  -"):
            title = line.replace("TI  - ", "").strip()
        elif line.startswith("AB  -"):
            abstract = line.replace("AB  - ", "").strip()
        elif line.startswith("JT  -"):
            journal = line.replace("JT  - ", "").strip()
        elif line.startswith("DP  -"):
            year = line.replace("DP  - ", "").split(" ")[0]
    return title, abstract, journal, year

# ==== Use LLM to extract computational techniques from abstract ====
def extract_with_llm(abstract):
    prompt = f"""You are a biomedical research assistant. Extract the machine learning, statistical, or computational techniques used in the following abstract (for breast cancer subtype classification):

{abstract}

Return a clean comma-separated list of techniques:"""
    try:
        result = llm(prompt, do_sample=True)[0]['generated_text']
        text = result.split("list of techniques:")[-1].strip()
        return [t.strip().lower() for t in text.split(",") if len(t.strip()) > 1]
    except:
        return []

# ==== Match extracted techniques to standard vocabulary ====
def match_to_vocab(extracted, vocab):
    matched = []
    for kw in vocab:
        for e in extracted:
            if kw in e or e in kw:
                matched.append(kw)
    return list(set(matched))

# ==== Main pipeline: search → extract → match → summarize ====
def run_llm_match_pipeline():
    pmids = search_pubmed(QUERY, MAX_RESULTS)
    results = []
    counter = Counter()

    for pmid in pmids:
        title, abstract, journal, year = fetch_metadata(pmid)
        if abstract == "NA":
            continue
        extracted = extract_with_llm(abstract)
        matched = match_to_vocab(extracted, TECH_KEYWORDS)
        for m in matched:
            counter[m] += 1
        results.append({
            "PMID": pmid,
            "Title": title,
            "Matched_Techniques": ", ".join(matched),
            "LLM_Extracted": ", ".join(extracted),
            "Year": year,
            "Journal": journal,
            "Abstract": abstract
        })
        time.sleep(0.5)  # Rate-limiting to avoid API overload

    df_articles = pd.DataFrame(results)
    df_stats = pd.DataFrame(counter.items(), columns=["Technique", "Count"]).sort_values("Count", ascending=False)
    return df_articles, df_stats

# ==== Execute the pipeline ====
df_articles, df_stats = run_llm_match_pipeline()
print(df_stats.head())

# ==== Optional: Save results ====
# df_articles.to_csv("llm_techniques_articles.csv", index=False)
# df_stats.to_csv("technique_freq.csv", index=False)
