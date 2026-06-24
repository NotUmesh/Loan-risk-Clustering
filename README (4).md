# Customer Segmentation — Unsupervised Clustering

A customer personality analysis project that segments a retail company's customers into distinct behavioral groups using unsupervised machine learning. The goal is to help a business understand *who* its customers are, so marketing and product decisions can be targeted at the right segment instead of broadcast to everyone.

## Problem Statement

Instead of spending budget marketing a new product to every customer in a database, a company can identify which customer segment is most likely to respond and target only that segment. This project builds that segmentation from raw customer and purchase-history data.

## Dataset

The dataset (`marketing_campaign.csv`) contains 2,216 customers across 29 original features, grouped into four categories:

| Group | Features |
|---|---|
| **People** | Birth year, education, marital status, income, kids/teens at home, enrollment date, recency, complaints |
| **Products** | Amount spent (last 2 years) on wine, fruit, meat, fish, sweets, gold |
| **Promotion** | Deals used, response to 5 past marketing campaigns |
| **Place** | Purchases via web, catalogue, and in-store; monthly web visits |

## Approach

1. **Data Cleaning** — Dropped rows with missing income values; checked for and removed duplicates; parsed `Dt_Customer` to a proper datetime type.

2. **Feature Engineering** — Derived several higher-signal features from the raw columns:
   - `Age` from birth year
   - `Spent` — total 2-year spend across all product categories
   - `Living_With` — collapsed marital status into Partner / Alone
   - `Children`, `Family_Size`, `Is_Parent` — household composition
   - Collapsed `Education` into three ordinal tiers (Undergraduate, Graduate, Postgraduate)
   - Dropped redundant/constant columns (`Marital_Status`, `Z_CostContact`, `Z_Revenue`, `ID`, etc.)

3. **EDA & Outlier Detection** — Used pair plots, distribution plots, and IQR-based boxplots to inspect `Age`, `Income`, and `Spent`. Removed clear outliers (e.g., age > 100, income > 600,000) that would otherwise distort scaling and clustering.

4. **Encoding & Scaling** — Ordinal-encoded `Education` and `Living_With`; applied `StandardScaler` to all numeric features so no single feature (e.g., raw income) dominates distance-based clustering.

5. **Dimensionality Reduction** — Used PCA to reduce the scaled feature space to 3 components for clustering and visualization.

6. **Clustering** — Used the Elbow Method (via `KElbowVisualizer`) to select the number of clusters, then applied **Agglomerative Hierarchical Clustering** to assign each customer to one of 4 segments.

7. **Cluster Profiling** — Analyzed each cluster's income, spending, family structure, age, and promotional response to build a behavioral profile for each segment.

## Results — Customer Segments

| Cluster | Profile |
|---|---|
| **0** | Mostly parents, small families (≤3 members), one young child, relatively younger, **low income & low spending** |
| **1** | Parents with teenagers, larger families (2–4 members), includes single parents, relatively older, **high spending, average income** |
| **2** | Not parents, smallest households (≤2 members), span all ages, **high income & high spending** — the most valuable segment |
| **3** | Parents with teenagers, largest families (2–5 members), relatively older, **low spending, average income** |

Additional findings:
- Marketing campaign acceptance was low across all clusters — no customer accepted all 5 campaigns, suggesting better-targeted campaigns are needed.
- Deal-based promotions performed well for **Clusters 1 and 3**, but had little effect on **Cluster 2** (the high-value segment), which doesn't respond to discounting.

## Tech Stack

- **Data handling:** pandas, numpy
- **Visualization:** seaborn, matplotlib, plotly, missingno
- **Preprocessing:** scikit-learn (`StandardScaler`)
- **Dimensionality reduction:** scikit-learn (`PCA`)
- **Clustering:** scikit-learn (`AgglomerativeClustering`, `KMeans`), yellowbrick (`KElbowVisualizer`)

## How to Run

```bash
pip install pandas numpy seaborn matplotlib missingno plotly scikit-learn yellowbrick
jupyter notebook Customer_Segmentation__Clustering_.ipynb
```

Ensure `marketing_campaign.csv` is in the same directory as the notebook before running.

## Possible Extensions

- Validate cluster stability with silhouette score or other internal validation metrics
- Try alternative clustering algorithms (DBSCAN, Gaussian Mixture Models) for comparison
- Build a classifier to predict cluster membership for new customers in real time
- Translate cluster profiles into concrete, segment-specific marketing campaigns
