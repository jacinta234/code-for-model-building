# Step 1 - Setup
# 1.1 import requests, os
import requests, os

# Configure matplotlib to use non-interactive backend for headless environments
import matplotlib
matplotlib.use('Agg')

SAVE_DIR = "data/raw"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/results", exist_ok=True)
os.makedirs("data/embeddings", exist_ok=True)
os.makedirs("figures", exist_ok=True)

datasets = {
    "rppa.tsv.gz": (
        "https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/"
        "RPPA_RBN.gz"
    ),
    "rnaseq.tsv.gz": (
        "https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/"
        "HiSeqV2_percentile.gz"
    ),
    "clinical.tsv": (
        "https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/"
        "BRCA_clinicalMatrix"
    ),
}

for fname, url in datasets.items():
    out_path = os.path.join(SAVE_DIR, fname)
    if os.path.exists(out_path):
        print(f"  {fname} already exists, skipping download")
    else:
        print(f"Downloading {fname}...")
        r = requests.get(url, stream=True)
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  Saved to {out_path}")

print("All downloads complete.")

# 1.2 Load data
import pandas as pd

# Xena format: genes/proteins as rows, samples as columns
# Transpose so rows = samples, columns = features
rppa_raw = pd.read_csv("data/raw/rppa.tsv.gz", sep="\t", index_col=0, compression='gzip').T
rnaseq_raw = pd.read_csv("data/raw/rnaseq.tsv.gz", sep="\t", index_col=0, compression='gzip').T
clinical = pd.read_csv("data/raw/clinical.tsv", sep="\t", index_col=0)

print("RPPA shape (samples x proteins):", rppa_raw.shape)
print("RNA-seq shape (samples x genes):", rnaseq_raw.shape)
print("Clinical shape:", clinical.shape)

# Quick check of PAM50 subtype labels
print("\nSubtype counts:")
print(clinical["PAM50Call_RNAseq"].value_counts())

# Step 2 - Preprocessing
# 2.1 Finding samples present in all three datasets for alignment
common = (
    rppa_raw.index
    .intersection(rnaseq_raw.index)
    .intersection(clinical.index)
)
print(f"Samples in all three datasets: {len(common)}")

rppa   = rppa_raw.loc[common].copy()
rnaseq = rnaseq_raw.loc[common].copy()
clin   = clinical.loc[common].copy()

# Extract PAM50 labels and encode them as integers
label_col = "PAM50Call_RNAseq"
subtypes   = clin[label_col].dropna()
common_labeled = common.intersection(subtypes.index)

rppa   = rppa.loc[common_labeled]
rnaseq = rnaseq.loc[common_labeled]
clin   = clin.loc[common_labeled]
labels_str = subtypes.loc[common_labeled]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
labels = le.fit_transform(labels_str)
print("Subtypes:", list(le.classes_))

import numpy as np
from sklearn.impute import SimpleImputer

#2.2 Handling missing values

def clean_missing(df, threshold=0.2):
    """
    Drop features with >threshold missing, then median-impute the rest.
    """
    missing_rate = df.isnull().mean(axis=0)
    keep = missing_rate[missing_rate <= threshold].index
    df_clean = df[keep].copy()
    print(f"  Features before: {df.shape[1]}, after removing >{threshold*100:.0f}% missing: {df_clean.shape[1]}")

    imputer = SimpleImputer(strategy="median")
    df_imputed = pd.DataFrame(
        imputer.fit_transform(df_clean),
        index=df_clean.index,
        columns=df_clean.columns
    )
    return df_imputed

print("Cleaning RPPA...")
rppa_clean   = clean_missing(rppa)

print("Cleaning RNA-seq...")
rnaseq_clean = clean_missing(rnaseq)

#2.3 Normalisation
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# RNA-seq: already percentile-ranked 0-100, scale to [0,1]
rnaseq_norm = rnaseq_clean / 100.0
print("RNA-seq range:", rnaseq_norm.min().min().round(3), "-", rnaseq_norm.max().max().round(3))

# RPPA: z-score normalise per protein across samples
scaler_rppa = StandardScaler()
rppa_norm = pd.DataFrame(
    scaler_rppa.fit_transform(rppa_clean),
    index=rppa_clean.index,
    columns=rppa_clean.columns
)
print("RPPA mean:", rppa_norm.mean().mean().round(4), "(should be ~0)")
print("RPPA std:", rppa_norm.std().mean().round(4),  "(should be ~1)")

# 2.4 Batch Effect Correction
try:
    from combat.pycombat import pycombat
    has_pycombat = True
except ImportError:
    has_pycombat = False
    print("Warning: pycombat not installed. Batch correction will be skipped.")


# pycombat expects features x samples (transpose in, transpose out)
# Here we create a simple 2-batch vector (RPPA lab vs RNAseq lab)
# In practice, check clinical metadata for a 'plate' or 'batch' column
import numpy as np

# Example: assign batches from clinical metadata if available.
if has_pycombat and "batch" in clin.columns:
    batch = clin.loc[rppa_norm.index, "batch"]
    rppa_corrected = pd.DataFrame(
        pycombat(rppa_norm.T, batch).T,
        index=rppa_norm.index,
        columns=rppa_norm.columns
    )
    print("Batch correction applied.")
else:
    rppa_corrected = rppa_norm
    if not has_pycombat:
        print("pycombat not installed - skipping batch correction.")
    else:
        print("No batch column found - skipping ComBat.")

#2.5 Feature Selection - Filtering down from 20,531 genes to a manageable number for modeling
# Variance filtering for RNA-seq
N_TOP_GENES = 3000

gene_var = rnaseq_norm.var(axis=0).sort_values(ascending=False)
top_genes = gene_var.head(N_TOP_GENES).index

rnaseq_filtered = rnaseq_norm[top_genes]
print(f"RNA-seq reduced: {rnaseq_norm.shape[1]} -> {rnaseq_filtered.shape[1]} genes")

# RPPA usually has far fewer features (~200), so no filtering needed
print(f"RPPA features retained: {rppa_corrected.shape[1]}")

# Save final datasets
rnaseq_filtered.to_csv("data/processed/rnaseq_filtered.csv")
rppa_corrected.to_csv("data/processed/rppa_corrected.csv")
np.save("data/processed/labels.npy", labels)
print("Processed data saved.")

# Step 3 - Model Building
# 3.1 Single-Omics Baselines - Random Forest and SVM Classifiers are trained on each modality separately and these values become the benchmark to beat.

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, balanced_accuracy_score
import numpy as np

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scorers = {
    "accuracy": "accuracy",
    "f1_macro": "f1_macro",
    "balanced_acc": make_scorer(balanced_accuracy_score),
}

results = {}

for mod_name, X in [("RNA-seq", rnaseq_filtered.values),
                    ("RPPA",    rppa_corrected.values)]:
    for clf_name, clf in [
        ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ("SVM",           SVC(kernel="rbf", C=1.0, probability=True, random_state=42)),
    ]:
        key = f"{clf_name} ({mod_name})"
        cv = cross_validate(clf, X, labels, cv=skf, scoring=scorers, n_jobs=-1)
        results[key] = {
            "accuracy":  cv["test_accuracy"].mean(),
            "f1_macro":  cv["test_f1_macro"].mean(),
            "bal_acc":   cv["test_balanced_acc"].mean(),
        }
        print(f"{key}: acc={results[key]['accuracy']:.3f}, f1={results[key]['f1_macro']:.3f}")

        #3.2 Multi-Omics Integration - Concatenate features from both modalities and train the same classifiers on the combined dataset. A Multi-Omics Autoencoder is a neural network architecture designed to learn a compressed representation of multi-omics data. It consists of an encoder that maps the input data to a lower-dimensional latent space and a decoder that reconstructs the original data from this latent representation. The autoencoder is trained to minimize the reconstruction error, allowing it to capture the underlying structure and relationships between different omics modalities.
import torch
import torch.nn as nn

class MultiOmicsAE(nn.Module):
    def __init__(self, rna_dim, rppa_dim, latent_dim=128):
        super().__init__()
        # Separate encoders for each modality
        self.rna_enc = nn.Sequential(
            nn.Linear(rna_dim, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256),     nn.ReLU(),
        )
        self.rppa_enc = nn.Sequential(
            nn.Linear(rppa_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),       nn.ReLU(),
        )
        # Joint embedding from concatenated representations
        self.joint = nn.Sequential(
            nn.Linear(256 + 64, latent_dim), nn.ReLU()
        )
        # Shared decoder to reconstruct both modalities
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.ReLU(),
            nn.Linear(512, rna_dim + rppa_dim)
        )
        self.rna_dim = rna_dim
        self.rppa_dim = rppa_dim

    def forward(self, rna, rppa):
        z_rna  = self.rna_enc(rna)
        z_rppa = self.rppa_enc(rppa)
        z      = self.joint(torch.cat([z_rna, z_rppa], dim=1))
        recon  = self.decoder(z)
        return z, recon[:, :self.rna_dim], recon[:, self.rna_dim:]

    def encode(self, rna, rppa):
        z_rna  = self.rna_enc(rna)
        z_rppa = self.rppa_enc(rppa)
        return self.joint(torch.cat([z_rna, z_rppa], dim=1))

#Data is now converted to pyTorch tensors and split into training and validation sets. The autoencoder is trained using the Adam optimizer and mean squared error loss function. After training, the encoder part of the autoencoder is used to extract latent representations of the multi-omics data, which can then be used for downstream classification tasks.
import torch
from torch.utils.data import DataLoader, TensorDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Convert to tensors
X_rna  = torch.tensor(rnaseq_filtered.values, dtype=torch.float32).to(device)
X_rppa = torch.tensor(rppa_corrected.values,  dtype=torch.float32).to(device)

dataset    = TensorDataset(X_rna, X_rppa)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

model = MultiOmicsAE(
    rna_dim=rnaseq_filtered.shape[1],
    rppa_dim=rppa_corrected.shape[1],
    latent_dim=128
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn   = nn.MSELoss()

for epoch in range(100):
    model.train()
    total_loss = 0
    for rna_b, rppa_b in dataloader:
        optimizer.zero_grad()
        _, recon_rna, recon_rppa = model(rna_b, rppa_b)
        loss = loss_fn(recon_rna, rna_b) + loss_fn(recon_rppa, rppa_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {total_loss/len(dataloader):.4f}")

# Extract latent embeddings for all samples
model.eval()
with torch.no_grad():
    latent_ae = model.encode(X_rna, X_rppa).cpu().numpy()
print("Latent embedding shape:", latent_ae.shape)
np.save("data/embeddings/latent_ae.npy", latent_ae)

#3.3 - Introducing a Multi-Omics Variational Autoencoder (VAE) for more robust latent representation learning. The VAE introduces a probabilistic framework that allows for better generalization and captures the underlying distribution of the data. The encoder outputs mean and log-variance parameters, which are used to sample from the latent space, while the decoder reconstructs the input data from these samples. 
class MultiOmicsVAE(nn.Module):
    def __init__(self, rna_dim, rppa_dim, latent_dim=128):
        super().__init__()
        self.rna_enc = nn.Sequential(
            nn.Linear(rna_dim, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256),     nn.ReLU(),
        )
        self.rppa_enc = nn.Sequential(
            nn.Linear(rppa_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),       nn.ReLU(),
        )
        hidden = 256 + 64
        # VAE outputs mean and log-variance (not a single point)
        self.mu     = nn.Linear(hidden, latent_dim)
        self.log_var = nn.Linear(hidden, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.ReLU(),
            nn.Linear(512, rna_dim + rppa_dim)
        )
        self.rna_dim  = rna_dim
        self.rppa_dim = rppa_dim

    def reparameterise(self, mu, log_var):
        """Sample z = mu + eps * std, where eps ~ N(0,1)."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, rna, rppa):
        h = torch.cat([self.rna_enc(rna), self.rppa_enc(rppa)], dim=1)
        mu, lv = self.mu(h), self.log_var(h)
        z = self.reparameterise(mu, lv)
        recon = self.decoder(z)
        return z, recon[:, :self.rna_dim], recon[:, self.rna_dim:], mu, lv

def vae_loss(recon_rna, rna, recon_rppa, rppa, mu, lv, beta=1.0):
    recon = nn.functional.mse_loss(recon_rna, rna) \
          + nn.functional.mse_loss(recon_rppa, rppa)
    # KL divergence: measures how much the latent space deviates from N(0,1)
    kld = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
    return recon + beta * kld, recon.item(), kld.item()

#3.4 Training the Multi-Omics VAE
vae = MultiOmicsVAE(
    rna_dim=rnaseq_filtered.shape[1],
    rppa_dim=rppa_corrected.shape[1],
    latent_dim=128
).to(device)

opt_vae = torch.optim.Adam(vae.parameters(), lr=1e-3)

for epoch in range(100):
    vae.train()
    total = 0
    for rna_b, rppa_b in dataloader:
        opt_vae.zero_grad()
        _, r_rna, r_rppa, mu, lv = vae(rna_b, rppa_b)
        loss, recon_l, kl_l = vae_loss(r_rna, rna_b, r_rppa, rppa_b, mu, lv)
        loss.backward()
        opt_vae.step()
        total += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Total: {total/len(dataloader):.4f}")

vae.eval()
with torch.no_grad():
    _, _, _, mu_all, _ = vae(X_rna, X_rppa)
    latent_vae = mu_all.cpu().numpy()

np.save("data/embeddings/latent_vae.npy", latent_vae)
print("VAE embeddings saved:", latent_vae.shape)


#3.5 Plotting the training of VAE and AE 
import matplotlib.pyplot as plt

epochs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
ae_loss  = [1.1267, 0.5307, 0.4272, 0.3690, 0.3281, 0.3000, 0.2721, 0.2491, 0.2327, 0.2158]
vae_loss = [1.1871, 0.7800, 0.7153, 0.6842, 0.6601, 0.6401, 0.6220, 0.5969, 0.5786, 0.5594]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(epochs, ae_loss, marker='o', label="AE (reconstruction loss)", color="#2a78d6")
ax.plot(epochs, vae_loss, marker='o', label="VAE (reconstruction + KL loss)", color="#D85A30")

ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training convergence: AE vs VAE")
ax.legend(frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("figures/loss_curves_coarse.png", dpi=180, bbox_inches="tight")
plt.show() 

#3.6 Supervised Classification on Latent Representations - attaching a small neural network classifier to the latent embeddings learned by the autoencoder and variational autoencoder. The classifier is trained to predict PAM50 subtypes from the compressed representations, allowing us to evaluate how well the latent space captures relevant biological information for classification tasks.

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, balanced_accuracy_score, cohen_kappa_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scorers = {
    "accuracy":  "accuracy",
    "f1_macro":  "f1_macro",
    "bal_acc":   make_scorer(balanced_accuracy_score),
    "kappa":     make_scorer(cohen_kappa_score),
}

for emb_name, X_emb in [("AE", latent_ae), ("VAE", latent_vae)]:
    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=300,
        random_state=42
    )
    cv = cross_validate(clf, X_emb, labels, cv=skf, scoring=scorers)
    print(f"\n{emb_name} multi-omics classifier:")
    for k, v in scorers.items():
        print(f"  {k}: {cv[f'test_{k}'].mean():.3f} ± {cv[f'test_{k}'].std():.3f}")

#Step 4 - Clustering, to discover whether the latent representations can reveal novel subtypes or groupings within the breast cancer samples. This unsupervised learning approach can help identify patterns that may not be apparent through supervised classification alone.
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

print("k | Silhouette | ARI vs PAM50")
print("-" * 35)
for k in range(3, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    preds = km.fit_predict(latent_vae)
    sil = silhouette_score(latent_vae, preds)
    ari = adjusted_rand_score(labels, preds)
    print(f"{k} | {sil:.3f}      | {ari:.3f}")

# Save k=5 cluster assignments
km5 = KMeans(n_clusters=5, random_state=42, n_init=20)
cluster_labels = km5.fit_predict(latent_vae)
np.save("data/results/kmeans_clusters.npy", cluster_labels)

#4.1 Hierarchical Clustering - to visualize the relationships between samples based on their latent representations. This method builds a tree-like structure (dendrogram) that illustrates how samples cluster together at various levels of similarity, providing insights into potential subtypes and their hierarchical relationships.
# Here, a dendogram will be built to see if or how the subtypes relate hierarchically. This can help identify potential subtypes and their relationships, which may not be apparent through other clustering methods.

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

# Fit hierarchical clustering
hc = AgglomerativeClustering(n_clusters=5, linkage="ward")
hc_labels = hc.fit_predict(latent_vae)

sil = silhouette_score(latent_vae, hc_labels)
ari = adjusted_rand_score(labels, hc_labels)
print(f"Hierarchical (Ward, k=5): silhouette={sil:.3f}, ARI={ari:.3f}")

# Plot dendrogram on a sample (full dendrogram is too large)
sample_idx = np.random.choice(len(latent_vae), 200, replace=False)
Z = sch.linkage(latent_vae[sample_idx], method="ward")

plt.figure(figsize=(12, 4))
sch.dendrogram(Z, no_labels=True, color_threshold=0.7*max(Z[:,2]))
plt.title("Hierarchical clustering dendrogram (200 sample subset)")
plt.ylabel("Ward linkage distance")
plt.tight_layout()
plt.savefig("figures/dendrogram.png", dpi=150)
# plt.show()  # Commented to avoid display issues

# 4.3 UMAP Visualization - to project the high-dimensional latent representations into a 2D space for visualization. UMAP (Uniform Manifold Approximation and Projection) is a non-linear dimensionality reduction technique that preserves both local and global structure, making it suitable for visualizing complex multi-omics data.
import umap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
embedding_2d = reducer.fit_transform(latent_vae)

subtype_names = le.classes_
colors = ["#1D9E75","#D85A30","#7F77DD","#BA7517","#D4537E"]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: colour by PAM50 true label
for i, name in enumerate(subtype_names):
    mask = labels == i
    axes[0].scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                    c=colors[i], label=name, s=12, alpha=0.7)
axes[0].set_title("UMAP - true PAM50 subtypes")
axes[0].legend(markerscale=2, fontsize=9)

# Right: colour by K-means cluster
for i in range(5):
    mask = cluster_labels == i
    axes[1].scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                    c=colors[i], label=f"Cluster {i}", s=12, alpha=0.7)
axes[1].set_title("UMAP - K-means clusters (k=5)")
axes[1].legend(markerscale=2, fontsize=9)

plt.tight_layout()
plt.savefig("figures/umap.png", dpi=150)
# plt.show()  # Commented to avoid display issues

# Step 5 - Analysis and Interpretation - to interpret the results of the clustering and classification analyses. This step involves examining the relationships between the identified clusters, the PAM50 subtypes, and other clinical variables. It may also include identifying key features that contribute to the separation of clusters or subtypes, as well as exploring potential biological insights derived from the multi-omics data integration.

# 5.1 SHAP explainability to identify which features (genes/proteins) are most influential in the models predictions. This will determine which genes/proteins significantly drive subtype predictions.
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Combine features for a directly interpretable model
X_combined = np.hstack([
    rnaseq_filtered.values,
    rppa_corrected.values
])
feature_names = list(rnaseq_filtered.columns) + list(rppa_corrected.columns)

# Train RF on full combined features
rf_full = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_full.fit(X_combined, labels)

# SHAP TreeExplainer is fast for tree-based models
explainer = shap.TreeExplainer(rf_full)
shap_values = explainer.shap_values(X_combined)

# Summary plot - shows top features across all subtypes
shap.summary_plot(
    shap_values, X_combined,
    feature_names=feature_names,
    class_names=le.classes_,
    max_display=20,
    show=False
)
plt.tight_layout()
plt.savefig("figures/shap_summary.png", dpi=150, bbox_inches="tight")
# plt.show()  # Commented to avoid display issues

import shap
from sklearn.ensemble import RandomForestClassifier

X_combined = np.hstack([rnaseq_filtered.values, rppa_norm.values])
feature_names = list(rnaseq_filtered.columns) + list(rppa_norm.columns)

rf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_combined, labels)

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_combined)  # shape: (694, 3131, 5)

for i, subtype in enumerate(le.classes_):
    importance = np.abs(shap_values[:,:,i]).mean(axis=0)
    top20 = np.argsort(importance)[::-1][:20]
    print(subtype, [feature_names[j] for j in top20])


# Top 20 features per subtype 
rows = []
for i, subtype in enumerate(le.classes_):
    imp = np.abs(shap_values[:,:,i]).mean(axis=0)
    top20 = pd.Series(imp, index=feature_names).nlargest(20)
    print(f"\nTop 20 features for {subtype}:")
    print(top20.to_string())
    for feat, val in top20.items():
        rows.append({"subtype": subtype, "feature": feat, "mean_abs_shap": val})

pd.DataFrame(rows).to_csv("data/results/shap_top20_per_subtype.csv", index=False)



# 5.3 Survival Analysis - to assess the clinical relevance of the identified clusters and subtypes. This involves analyzing patient survival data in relation to the clusters derived from the multi-omics integration, providing insights into potential prognostic implications of the discovered patterns.
# Kaplan-Meier survival curves are generated for each cluster, and statistical tests (e.g., log-rank test) are performed to evaluate differences in survival distributions between clusters. This analysis helps determine whether the identified clusters have distinct survival outcomes, which can inform clinical decision-making and potential therapeutic strategies.


from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
import matplotlib.pyplot as plt

# Correct column names, verified against actual clinical.tsv
surv = clin[['OS_Time_nature2012', 'OS_event_nature2012']].copy()
surv.columns = ['time', 'event']
valid = surv.notna().all(axis=1).values

T = surv['time'][valid].values
E = surv['event'][valid].values
labels_valid = labels[valid]
clusters_valid = cluster_labels[valid]  # K-means assignments

colors = ["#1D9E75","#D85A30","#7F77DD","#BA7517","#D4537E"]
kmf = KaplanMeierFitter()

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Panel 1: true PAM50 subtype
ax = axes[0]
for i, subtype in enumerate(le.classes_):
    mask = labels_valid == i
    if mask.sum() < 5: continue
    kmf.fit(T[mask], event_observed=E[mask], label=f"{subtype} (n={mask.sum()})")
    kmf.plot_survival_function(ax=ax, ci_show=False, color=colors[i])
lr_true = multivariate_logrank_test(T, labels_valid, E)
ax.set_title(f"KM by true PAM50 subtype (log-rank p={lr_true.p_value:.4f})")
ax.set_xlabel("Days"); ax.set_ylabel("Survival probability")

# Panel 2: K-means clusters
ax = axes[1]
for c in range(5):
    mask = clusters_valid == c
    if mask.sum() < 5: continue
    kmf.fit(T[mask], event_observed=E[mask], label=f"Cluster {c} (n={mask.sum()})")
    kmf.plot_survival_function(ax=ax, ci_show=False, color=colors[c])
lr_cluster = multivariate_logrank_test(T, clusters_valid, E)
ax.set_title(f"KM by K-means cluster (log-rank p={lr_cluster.p_value:.4f})")
ax.set_xlabel("Days"); ax.set_ylabel("Survival probability")

plt.tight_layout()
plt.savefig("figures/kaplan_meier_curves.png", dpi=180, bbox_inches="tight")
plt.show()

print(f"True PAM50: p={lr_true.p_value:.4f}")
print(f"K-means cluster: p={lr_cluster.p_value:.4f}")
# plt.show()  # Commented to avoid display issues

# 5.5 An ablation study to evaluate the contribution of each omics modality to the overall model performance. This involves systematically removing one modality at a time and retraining the model to observe changes in performance metrics. The results of this study can provide insights into the relative importance of each omics layer in predicting breast cancer subtypes and inform future multi-omics integration strategies.
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, balanced_accuracy_score
from sklearn.neural_network import MLPClassifier   # <-- add this import
import pandas as pd

clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)  # <-- define locally

scorers = {
    "accuracy": "accuracy",
    "f1_macro": "f1_macro",
    "balanced_acc": make_scorer(balanced_accuracy_score),
}

ablation_results = {}
conditions = {
    "RNA-seq only": rnaseq_filtered.values,
    "RPPA only": rppa_corrected.values,
    "Naive concatenation": np.hstack([rnaseq_filtered.values, rppa_corrected.values]),
    "AE (multi-omics)": latent_ae,
    "VAE (multi-omics)": latent_vae,
}

for name, X in conditions.items():
    cv = cross_validate(clf, X, labels, cv=skf, scoring=scorers, n_jobs=-1)
    ablation_results[name] = {
        "accuracy":     cv["test_accuracy"].mean(),
        "f1_macro":     cv["test_f1_macro"].mean(),
        "balanced_acc": cv["test_balanced_acc"].mean(),
        "f1_macro_std": cv["test_f1_macro"].std(),
    }

df_ablation = pd.DataFrame(ablation_results).T.round(3)
print(df_ablation)
df_ablation.to_csv("data/results/ablation_study.csv")


#5.6 Confusion Matrix 
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred = cross_val_predict(rf, X_combined, labels, cv=skf, n_jobs=-1)
cm = confusion_matrix(labels, y_pred)

print(classification_report(labels, y_pred, target_names=le.classes_))

fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(ax=ax, cmap='Blues', colorbar=True, values_format='d')
ax.set_title('Cross-validated confusion matrix\nRandom Forest (RNA-seq + RPPA combined)')
plt.tight_layout()
plt.savefig("figures/confusion_matrix.png", dpi=180, bbox_inches="tight")
plt.show()
df_ablation.to_csv("data/results/ablation_study.csv")

#Step 6 Reproducability - All results and figures must be saved for reproducibility and further analysis. This includes saving processed datasets, model embeddings, performance metrics, clustering assignments, and visualizations. By organizing and storing these outputs, we ensure that the analysis can be revisited, validated, and extended in future research. The seeding function ensures that every time the code is run, identical results are produced.
import random, os, numpy as np, torch

def set_seeds(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    print(f"All seeds set to {seed}")

# 5.7 Results Comparison Table - to summarize the performance metrics of the various models and clustering methods. This table provides a clear overview of how each approach performed in terms of accuracy, F1 score, balanced accuracy, and other relevant metrics, allowing for easy comparison and identification of the best-performing methods.
import pandas as pd

# Build results dict from cross_validate outputs (after all CV steps)
all_results = {
    "SVM (RNA-seq)":          results["SVM (RNA-seq)"],
    "SVM (RPPA)":             results["SVM (RPPA)"],
    "RF (RNA-seq)":           results["Random Forest (RNA-seq)"],
    "RF (RPPA)":              results["Random Forest (RPPA)"],
    "AE multi-omics":         "ae_cv_results",
    "VAE multi-omics":        "vae_cv_results",
}

df_results = pd.DataFrame(all_results).T
df_results.columns = ["Accuracy", "F1 (macro)", "Balanced Acc"]
df_results = df_results.round(3)

print(df_results.to_string())
df_results.to_csv("data/results/performance_table.csv")

# Call at the top of every script:
set_seeds(42)

# 6.1 Project structure and file organization - to ensure that all data, code, and results are organized in a clear and reproducible manner. This includes creating directories for raw data, processed data, model outputs, figures, and scripts. A well-structured project layout facilitates collaboration, version control, and future extensions of the analysis.
# Dependencies (requirements.txt):
# numpy==1.26.4
# pandas==2.2.2
# scikit-learn==1.5.0
# torch==2.3.0
# matplotlib==3.9.0
# seaborn==0.13.2
# umap-learn==0.5.6
# shap==0.45.1
# lifelines==0.29.0
# scipy==1.13.0
# pyComBat==0.3.2

# Project directory structure:
# multi_omics_project/
# ├── data/
# │   ├── raw/          # Downloaded from UCSC Xena (never edited)
# │   ├── processed/    # Normalised, filtered, aligned
# │   ├── embeddings/   # AE and VAE latent vectors (.npy)
# │   └── results/      # CSVs of metrics, cluster assignments
# ├── src/
# │   ├── data_download.py
# │   ├── preprocessing.py
# │   ├── autoencoder.py
# │   ├── vae.py
# │   ├── classify.py
# │   ├── cluster.py
# │   ├── shap_explain.py
# │   ├── survival.py
# │   └── utils/seed.py
# ├── figures/          # All saved plots
# ├── requirements.txt

import sqlite3

conn = sqlite3.connect("breast_cancer.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS subtype_counts (
    subtype TEXT PRIMARY KEY,
    count INTEGER NOT NULL
)
""")

data = [
    ("LumA", 434),
    ("LumB", 194),
    ("Basal", 142),
    ("Normal", 119),
    ("Her2", 67)
]

cursor.executemany(
    "INSERT OR REPLACE INTO subtype_counts (subtype, count) VALUES (?, ?)",
    data
)

conn.commit()

# Verify
for row in cursor.execute("SELECT * FROM subtype_counts"):
    print(row)

conn.close()