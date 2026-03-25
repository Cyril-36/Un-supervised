**Social Media Topic Discovery for Disaster Response Intelligence**
INT 396 — Unsupervised Learning | Final Revised Plan

---

**1. Project Overview**

This project studies unsupervised topic discovery in disaster-related tweets using semantic and lexical representations, low-dimensional structure learning, and clustering. No labeled data is used for training or primary model selection. Annotated humanitarian categories are used only after clustering for supplementary agreement analysis. The main contribution is a controlled comparison of unsupervised pipelines and their transfer across disaster events.

The system automatically groups disaster-related social media posts into meaningful topics — rescue requests, infrastructure damage, resource needs, displaced populations — without any supervision, enabling faster situational awareness for disaster responders.

**Deliverables:** Research paper (8–12 pages, IEEE-style), working code pipeline (Jupyter Notebooks + Python scripts), interactive Streamlit demo (4 pages), GitHub repository with documentation and reproducibility artifacts.

---

**2. Datasets**

**Primary — HumAID (76,484 tweets)**
Loaded directly from HuggingFace (QCRI/HumAID-all). English-only tweets across 19 disaster events from 2016–2019 (earthquakes, hurricanes, wildfires, floods). 10 humanitarian category labels. Zero missing text, average tweet length 142 characters. Reasonably balanced distribution with the largest class at 27.8%. All 12 core experiments run on this dataset. Labels used only for post-hoc supplementary agreement analysis — never for training or model selection.

**Validation — CrisisBench Humanitarian Config, English Only (132,619 tweets)**
Loaded from HuggingFace (QCRI/CrisisBench-all-lang, humanitarian config). Filtered to English only. 61 disaster events with event and source metadata, enabling temporal and cross-disaster analysis. 16 label categories.

Pre-selected 3 events for validation using the rule: highest English tweet volume, clear time span, and different disaster types. The three events are: 2015 Nepal Earthquake (11,032 tweets, earthquake), Hurricane Harvey (3,992 tweets, hurricane), and 2013 Alberta Floods (8,724 tweets, flood). The "not_humanitarian" class is removed before clustering. The task is framed as "topic discovery within humanitarian crisis content," not full-stream disaster filtering.

CrisisBench is used only for external transfer testing. No model selection or tuning happens on this dataset — all tuning uses HumAID only.

---

**3. Preprocessing**

Light normalization only. Remove URLs, @mentions, RT markers, and exact duplicates. Normalize hashtags by stripping the # symbol but keeping the text (e.g., #earthquake becomes earthquake). No aggressive lemmatization — tweets are already short and over-cleaning destroys useful signal. No near-duplicate removal unless a simple, defensible rule emerges during EDA.

For CrisisBench: additionally filter to English only using the lang column and remove all rows labeled "not_humanitarian."

---

**4. Feature Extraction (Two Approaches)**

Approach A — SBERT Embeddings. Use all-MiniLM-L6-v2 from Sentence-Transformers to generate 384-dimensional dense vectors. Captures semantic meaning and contextual relationships between words. This is treated as feature extraction (preprocessing), not the core contribution.

Approach B — TF-IDF Vectors. Classical bag-of-words baseline with max 10,000 features, bigrams included. Produces sparse high-dimensional vectors. Serves as comparison to demonstrate the difference between classical and modern representations.

Both approaches feed into Stage 3. Results are compared to show how embedding quality affects downstream clustering.

---

**5. Dimensionality Reduction (Unsupervised Core #1)**

**UMAP (primary method).** Hyperparameters tuned via grid search on a stratified random sample of ~15,000 tweets from HumAID. Grid: n_neighbors in (5, 10, 15, 30) and min_dist in (0.0, 0.1, 0.25, 0.5). Optimized on Silhouette Score. Best parameters applied to the full 76K dataset. Reduces to 5D for clustering input and 2D for visualization.

**PCA (linear baseline for SBERT).** Standard PCA applied to dense SBERT embeddings. Reduces to 50D for clustering, 2D for visualization. Fast, simple, interpretable baseline.

**TruncatedSVD (linear baseline for TF-IDF).** PCA is not appropriate for sparse matrices because it centers the data and destroys sparsity. TruncatedSVD is the correct linear reduction method for TF-IDF. Same dimensionality targets as PCA.

**t-SNE (visualization only).** Used separately with perplexity=30 on a sampled subset to produce 2D visualization plots for qualitative comparison against UMAP projections. Not used as clustering input because it is stochastic, non-parametric, and does not preserve global structure reliably.

---

**6. Clustering (Unsupervised Core #2)**

**Core 3 algorithms (in the 12-config matrix):**

HDBSCAN — density-based, finds clusters of varying shapes and sizes, handles noise naturally by labeling uncertain points as -1. Expected best performer on UMAP-reduced embeddings. Key parameter: min_cluster_size (tuned). Noise handling: report percentage of tweets assigned to clusters vs labeled as noise. Intrinsic metrics computed on non-noise points only, with noise rate reported alongside.

K-Means — centroid-based, classic baseline. Requires pre-specifying K. K selected via Elbow method and Silhouette analysis over range 3–20. All points assigned to a cluster (no noise concept).

Agglomerative Clustering — hierarchical approach using Ward linkage. Number of clusters determined from dendrogram analysis. Provides a different perspective on cluster structure (bottom-up merging vs top-down partitioning).

**Optional appendix baseline:**

DBSCAN — standard density-based method. Expected to perform poorly on text embeddings because tweets rarely have uniform density. Most data will likely be labeled as noise or lumped into one giant cluster. This is included specifically to demonstrate why HDBSCAN (which handles variable density) is necessary. Not part of the core 12-config matrix.

---

**7. Core Experimental Matrix**

2 embeddings (SBERT, TF-IDF) × 2 reducers (UMAP, PCA/TruncatedSVD) × 3 clusterers (HDBSCAN, K-Means, Agglomerative) = 12 core configurations.

Each configuration is evaluated on the appropriate metrics (see Section 8). The 12-config comparison table forms the backbone of the results section.

The top 3 performing configurations are then run 3 additional times each with different random seeds to report mean ± standard deviation, since UMAP and K-Means are stochastic. This tests stability.

The single best configuration from HumAID is then applied to the 3 pre-selected CrisisBench events to test cross-dataset transfer.

---

**8. Evaluation**

**For K-Means and Agglomerative (convex-cluster methods):**
Silhouette Score (higher is better, max +1.0), Davies-Bouldin Index (lower is better, closer to 0.0), and Calinski-Harabasz Index (higher is better). These metrics assume convex, spherical clusters and are appropriate for centroid-based and hierarchical methods.

**For HDBSCAN and DBSCAN (density-based methods):**
DBCV — Density-Based Cluster Validity (higher is better). This metric is designed for arbitrarily shaped, density-based clusters and does not penalize non-convex shapes. Plus noise rate: percentage of tweets assigned to actual clusters vs labeled as noise (-1).

**Cross-method comparison:**
Silhouette Score is also computed on HDBSCAN's non-noise points for side-by-side comparison across all methods, with the explicit caveat that this metric is biased toward convex clusters and may undervalue HDBSCAN's results.

**Supplementary agreement analysis:**
HumAID's 10 humanitarian category labels are used after clustering to measure how well discovered clusters align with human-annotated categories. Reported using Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI). Explicitly framed as supplementary — not called "validation" or "ground truth" since humanitarian labels are not true topic labels.

**Qualitative check:**
Representative tweets per cluster (5–10 per cluster) presented as evidence that clusters are semantically coherent.

**Reproducibility:**
Fixed random seed (42) for all experiments. Top 3 configs repeated with seeds 42, 123, 456. All embeddings saved as .npy files. Configuration stored in a YAML file. One master results CSV with all metrics across all configs.

---

**9. Topic Interpretation**

**Primary method:** c-TF-IDF (class-based TF-IDF) to extract top distinguishing terms per cluster. Each cluster is treated as a "document," and TF-IDF scores reveal which terms are most distinctive to that cluster compared to others. Top-10 keywords per cluster reported in ranked tables with scores.

**Supporting methods:** Representative tweets per cluster (closest to cluster centroid or highest membership probability). Word clouds per cluster — used in the Streamlit demo for visual appeal but not as primary evidence in the research paper.

**Temporal analysis (CrisisBench only):** For the 3 pre-selected events with timestamp metadata, show how topic proportions shift over the course of a disaster. Early tweets tend to focus on the event itself, mid-crisis tweets shift to rescue and aid, later tweets focus on recovery and donations.

**Cluster labeling:** Map discovered clusters to interpretable labels based on their top keywords and representative tweets (e.g., "Infrastructure Damage," "Rescue Requests," "Donations and Volunteering," "Sympathy and Support").

---

**10. Streamlit Demo App (4 Pages)**

Page 1 — Overview + EDA. Project description, pipeline diagram, dataset statistics. Tweet length histograms, top hashtag bar charts, label distribution (for context only), event type breakdown for CrisisBench.

Page 2 — Clustering Explorer. Dropdown selectors for embedding type, reduction method, and clustering algorithm. Interactive 2D scatter plot (Plotly) colored by cluster assignment. Cluster size bar chart. Noise point visualization for HDBSCAN.

Page 3 — Topic Inspector. Per-cluster c-TF-IDF keyword tables. Word clouds per cluster. Representative tweets per cluster. Click a cluster to drill down.

Page 4 — Comparison + Temporal. Heatmap of metrics across all 12 configurations. Bar charts comparing best configs. Temporal topic proportions for CrisisBench events with a slider or dropdown to select the disaster event.

---

**11. Research Paper Structure**

Target length: 8–12 pages, IEEE or similar conference format.

Abstract (~0.25 pages) — Problem, approach (12-config unsupervised comparison), key finding (which pipeline works best and transfers across events).

Introduction (~1 page) — Disaster response challenge, social media volume during crises, why unsupervised discovery matters, contribution statement.

Related Work (~1.5 pages) — CrisisNLP and HumAID papers, topic modeling approaches (LDA, BERTopic), UMAP and HDBSCAN in text clustering, evaluation of unsupervised clustering.

Methodology (~2.5 pages) — 6-stage pipeline with a diagram. Preprocessing choices and justification. Embedding methods. Reduction methods (UMAP vs linear baselines, why t-SNE is visualization-only). Clustering algorithms. Evaluation protocol with separate metric families for convex vs density-based methods. Noise handling for HDBSCAN.

Experimental Setup (~1 page) — Dataset descriptions (HumAID and CrisisBench). 12-config matrix design. Hyperparameter tuning strategy (grid search on 15K sample). CrisisBench event selection criteria. Reproducibility details.

Results and Discussion (~3 pages) — Comparison tables across 12 configs. Best configuration analysis. Stability testing (3 seeds, mean ± std). Topic interpretation with c-TF-IDF keyword tables. Supplementary agreement with HumAID labels (ARI, NMI). Cross-dataset transfer results on CrisisBench. Temporal analysis. Discussion of why HDBSCAN outperforms (or doesn't) on different embedding types.

Limitations and Ethics (~0.5 pages) — English-only bias, platform bias (Twitter/X only), potential misinformation in the corpus, annotation subjectivity, discovered clusters as decision-support signals not ground truth.

Conclusion (~0.5 pages) — Best method, practical implications for disaster response agencies, future work (real-time streaming, multilingual support, multimodal with images).

References.

---

**12. Technical Stack**

Python 3.10+. Sentence-Transformers for SBERT embeddings. scikit-learn for TF-IDF, PCA, TruncatedSVD, K-Means, Agglomerative, DBSCAN, Silhouette, Davies-Bouldin, Calinski-Harabasz, ARI, NMI. umap-learn for UMAP. hdbscan library for HDBSCAN and DBCV. Matplotlib and Plotly for visualizations. WordCloud library for demo. Streamlit for the web app. Pandas and NumPy for data handling. HuggingFace datasets for data loading.

---

**13. Week-by-Week Timeline**

**Week 1 — Data Foundation.**
Days 1–2: Download HumAID and CrisisBench from HuggingFace. Explore structure, check distributions, verify text quality. Days 3–4: Build preprocessing pipeline — cleaning, exact duplicate removal, hashtag normalization, CrisisBench English filtering and "not_humanitarian" removal. Day 5: EDA — tweet length distributions, top hashtags, label breakdowns, CrisisBench event volume analysis to confirm the 3 pre-selected events. Days 6–7: Generate TF-IDF vectors and SBERT embeddings for the full HumAID corpus and the 3 CrisisBench event subsets. Save as .npy and .pkl files. Deliverable: clean datasets + embeddings ready.

**Week 2 — Core Unsupervised Pipeline.**
Days 1–2: Implement UMAP (with grid search on 15K sample), PCA, and TruncatedSVD reduction. Days 3–4: Implement HDBSCAN, K-Means, and Agglomerative clustering with initial hyperparameters. Day 5: Build evaluation module — Silhouette, DB, CH for convex methods; DBCV + noise rate for density methods; ARI and NMI for supplementary agreement. Days 6–7: Run initial experiments on HumAID, debug edge cases, finalize UMAP hyperparameters. Deliverable: working pipeline notebook with initial results for all 12 configs.

**Week 3 — Full Experiments and Analysis.**
Days 1–2: Run all 12 configurations systematically on HumAID. Log all metrics to a master CSV. Day 3: Build comparison tables, identify top 3 configs, run stability testing (3 seeds each). Days 4–5: Topic interpretation — c-TF-IDF keyword extraction, representative tweets, cluster labeling. Run optional DBSCAN appendix experiment. Days 6–7: Apply best config to CrisisBench events. Temporal analysis on the 3 events. Compute supplementary agreement (ARI, NMI) on HumAID. Deliverable: complete results, all visualizations, comparison tables.

**Weeks 4–5 — Demo UI, Report, and Polish.**
Days 1–3 (Week 4): Build Streamlit app — start with Clustering Explorer (most important page), then add Overview + EDA, Topic Inspector, and Comparison + Temporal pages. Days 4–7 (Week 4): Write research paper — Abstract, Introduction, Related Work, Methodology sections. Days 1–3 (Week 5): Complete Experimental Setup, Results and Discussion, Limitations and Ethics, Conclusion. Days 4–5 (Week 5): Polish report formatting, proofread, finalize GitHub repo with README, config YAML, saved embeddings, and results CSV. Days 6–7 (Week 5): Buffer for revisions and unexpected issues. Deliverable: final research paper + Streamlit app + GitHub repository.

---

**14. Risk Mitigation**

HumAID download issues — CrisisBench serves as backup; also check Kaggle mirrors and Harvard Dataverse. HDBSCAN finds too few clusters — tune min_cluster_size down; fall back to K-Means as the primary result. HDBSCAN assigns too much to noise — report noise rate transparently; this becomes a discussion point, not a flaw. 12 configs take too long — use Google Colab GPU; grid search on 15K sample; reduce to SBERT-only if pressed (6 configs). Streamlit app complexity — build incrementally; Clustering Explorer page first, others are additive. Poor cluster separation — try different UMAP params; filter dataset to single disaster event for a focused experiment. CrisisBench transfer fails — this is a valid negative result; discuss why in the paper (different annotation schemes, different event types, temporal distribution shift).

---

**15. Key Defense Points**

This is a pure unsupervised learning project. No labeled data is used for training or primary model selection. SBERT embeddings are pre-trained feature extractors (preprocessing), not the core contribution.

The core contribution is a controlled comparison of 12 unsupervised configurations across 2 embedding methods, 2 dimensionality reduction techniques, and 3 clustering algorithms, with appropriate evaluation metrics for each method family.

Evaluation uses separate metric families — convex metrics (Silhouette, DB, CH) for K-Means and Agglomerative, density-based metrics (DBCV) for HDBSCAN — avoiding the common mistake of evaluating all methods with metrics that assume one cluster shape.

The dual-dataset approach demonstrates generalizability. All tuning on HumAID, external transfer testing on CrisisBench with pre-defined event selection criteria.

Temporal analysis adds novelty by showing how disaster topics evolve over time, turning static clustering into dynamic topic discovery.

Real-world application: this pipeline could be deployed by disaster response agencies (NDMA, FEMA, UN OCHA) to monitor social media during crises for rapid situational awareness.