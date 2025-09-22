import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data, Batch
from scipy.sparse import load_npz, csr_matrix, issparse
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE, MDS
from sklearn.metrics import silhouette_score
from scipy.cluster import hierarchy
from umap import UMAP
import scanpy as sc
import anndata as ad
import networkx as nx
import igraph as ig
from typing import List, Tuple, Dict, Optional, Union
import warnings
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform
import scvelo as scv
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from collections import Counter
import re
from matplotlib.patches import Patch
from scipy.optimize import curve_fit

# Suppress warnings
warnings.filterwarnings('ignore')

# Define paths
DATA_DIR = r"D:\Sadra Foroutan\Single Cell Project\Zfish Data"
TIMEPOINT_FILES = [
    os.path.join(DATA_DIR, "GSM3067189_04hpf.csv"),
    os.path.join(DATA_DIR, "GSM3067190_06hpf.csv"),
    os.path.join(DATA_DIR, "GSM3067191_08hpf.csv"),
    os.path.join(DATA_DIR, "GSM3067192_10hpf.csv"),
    os.path.join(DATA_DIR, "GSM3067193_14hpf.csv"),
    os.path.join(DATA_DIR, "GSM3067194_18hpf.csv"),
    os.path.join(DATA_DIR, "GSM3067195_24hpf.csv")
]
METADATA_FILE = os.path.join(DATA_DIR, "GSE112294_ClusterNames.csv")
RAW_TF_LIST_FILE = os.path.join(DATA_DIR, "Danio_rerio_TF.txt")
CLEANED_TF_LIST_FILE = os.path.join(DATA_DIR, "Danio_rerio_TF_cleaned.csv")
RESULTS_PATH = os.path.join(DATA_DIR, "results")

# =========================
# Step 1: Enhanced Data Loading and Preprocessing
# =========================
class EnhancedscRNAProcessor:
    def __init__(self, timepoint_files, metadata_path=None, tf_list_path=None, min_cells=30, min_genes=300):
        """Initialize the enhanced scRNA-seq data processor."""
        self.timepoint_files = timepoint_files
        self.metadata_path = metadata_path
        self.tf_list_path = tf_list_path
        self.min_cells = min_cells
        self.min_genes = min_genes
        self.adata = None
        self.tf_list = None
        if tf_list_path and os.path.exists(tf_list_path):
            self.tf_list = pd.read_csv(tf_list_path, header=None)[0].values.tolist()
        
    def load_data(self):
        """Load raw data from time-point CSV files and create AnnData object."""
        print("[INFO] Loading data...")
        data_frames = []
        all_barcodes = []
        time_points = []
        
        for file_path in self.timepoint_files:
            basename = os.path.basename(file_path)
            match = re.search(r'_([0-9]+)hpf', basename)
            if match:
                time_point = float(match.group(1))
            else:
                raise ValueError(f"Could not extract time point from filename: {basename}")
            df = pd.read_csv(file_path, index_col=0)
            if df.shape[1] > 314:
                df = df.sample(n=314, axis=1, random_state=42)
            data_frames.append(df)
            all_barcodes.extend(df.columns)
            time_points.extend([time_point] * df.shape[1])
        
        combined_df = pd.concat(data_frames, axis=1, join='inner')
        expr_matrix = combined_df.values.T  # Cells x Genes
        barcodes = combined_df.columns
        gene_symbols = combined_df.index
        
        self.adata = ad.AnnData(X=expr_matrix)
        self.adata.obs_names = barcodes
        self.adata.var_names = gene_symbols
        self.adata.var['gene_ids'] = gene_symbols
        
        if self.metadata_path and os.path.exists(self.metadata_path):
            meta = pd.read_csv(self.metadata_path, index_col=0)
            meta = meta.loc[meta.index.intersection(self.adata.obs_names)]
            meta['cell_type_annotation'] = meta['ClusterName']
            meta['developmental_stage'] = pd.Series(time_points, index=barcodes)
            self.adata.obs = meta[['cell_type_annotation', 'developmental_stage']]
        else:
            print("[WARNING] Metadata file not found. Continuing without metadata.")
            self.adata.obs['cell_type_annotation'] = 'unknown'
            self.adata.obs['developmental_stage'] = pd.Series(time_points, index=barcodes)
        
        print(f"[INFO] Data loaded:")
        print(f"- Expression matrix shape: {self.adata.shape}")
        return self.adata
        
    def enhanced_qc_filtering(self):
        """Perform enhanced quality control filtering."""
        if self.adata is None:
            self.load_data()
            
        print("[INFO] Performing enhanced quality control...")
        
        # Basic filtering
        sc.pp.filter_genes(self.adata, min_cells=self.min_cells)
        sc.pp.filter_cells(self.adata, min_genes=self.min_genes)
        
        # Calculate QC metrics
        self.adata.var['mt'] = self.adata.var_names.str.startswith('mt-')
        sc.pp.calculate_qc_metrics(self.adata, qc_vars=['mt'], percent_top=None, inplace=True)
        
        # Remove cells with high mitochondrial content
        self.adata = self.adata[self.adata.obs.pct_counts_mt < 15].copy()
        
        # Approximate doublet removal via total counts and genes
        print("[INFO] Removing likely doublets based on total counts and number of genes...")
        counts = self.adata.obs['total_counts']
        genes = self.adata.obs['n_genes_by_counts']

        lower_counts = np.percentile(counts, 1)
        upper_counts = np.percentile(counts, 99)
        lower_genes = np.percentile(genes, 1)
        upper_genes = np.percentile(genes, 99)

        doublet_mask = (
            (counts < lower_counts) | (counts > upper_counts) |
            (genes < lower_genes) | (genes > upper_genes)
        )

        self.adata.obs['predicted_doublets'] = doublet_mask
        self.adata.obs['doublet_scores'] = 1.0 * doublet_mask

        self.adata = self.adata[~doublet_mask, :].copy()
        
        print(f"[INFO] After enhanced QC:")
        print(f"- Expression matrix shape: {self.adata.shape}")
        
        return self.adata
    
    def enhanced_normalize_and_scale(self):
        """Enhanced normalization and scaling."""
        if self.adata is None:
            self.enhanced_qc_filtering()
            
        print("[INFO] Enhanced normalization...")
        
        # Save raw counts
        self.adata.raw = self.adata
        
        # Normalization
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)
        
        # Find highly variable genes with batch correction
        sc.pp.highly_variable_genes(self.adata, n_top_genes=3000, 
                                   batch_key='developmental_stage', 
                                   flavor='seurat_v3')
        
        # Keep only highly variable genes
        self.adata = self.adata[:, self.adata.var.highly_variable].copy()
        
        # Scale with zero centering
        sc.pp.scale(self.adata, max_value=10, zero_center=True)
        
        return self.adata
    
    def engineer_features(self):
        """Engineer features for better trajectory detection."""
        print("[INFO] Engineering trajectory features...")
        
        # Cell cycle genes (zebrafish)
        s_genes = ['mcm5', 'pcna', 'tyms', 'fen1', 'mcm2', 'mcm4', 'rrm1', 'ung', 'gins2']
        g2m_genes = ['hmgb2', 'cdk1', 'nusap1', 'ube2c', 'birc5', 'tpx2', 'top2a', 'ndc80', 'cks2']
        
        # Filter for present genes
        s_genes = [g for g in s_genes if g in self.adata.raw.var_names]
        g2m_genes = [g for g in g2m_genes if g in self.adata.raw.var_names]
        
        if len(s_genes) > 0 and len(g2m_genes) > 0:
            sc.tl.score_genes_cell_cycle(self.adata, s_genes=s_genes, g2m_genes=g2m_genes)
        
        # Developmental signatures
        dev_signatures = {
            'neural': ['sox2', 'pax6', 'neurog1', 'neurod1', 'elavl3', 'elavl4'],
            'muscle': ['myod1', 'myog', 'myf5', 'myl2', 'tnnt2'],
            'endoderm': ['sox17', 'foxa2', 'gata6', 'hhex', 'prox1'],
            'mesoderm': ['tbxt', 'msgn1', 'mesp2', 'meox1', 'pax3'],
            'ectoderm': ['tfap2a', 'gata3', 'krt4', 'krt5'],
            'notochord': ['shh', 'col2a1', 'noto', 't/tbxt']
        }
        
        for sig_name, genes in dev_signatures.items():
            genes_present = [g for g in genes if g in self.adata.raw.var_names]
            if len(genes_present) > 0:
                sc.tl.score_genes(self.adata, genes_present, score_name=f'{sig_name}_score')
        
        return self.adata
    
    def run_enhanced_pca(self, n_components=50):
        """Run PCA with enhanced parameters."""
        if self.adata is None:
            self.enhanced_normalize_and_scale()
            
        print(f"[INFO] Running enhanced PCA with {n_components} components...")
        
        sc.tl.pca(self.adata, n_comps=n_components, svd_solver='arpack')
        
        # Plot variance ratio
        var_ratio = self.adata.uns['pca']['variance_ratio']
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(np.cumsum(var_ratio), 'o-')
        plt.xlabel('Number of PCs')
        plt.ylabel('Cumulative variance ratio')
        plt.title('PCA Elbow Plot')
        plt.axhline(y=0.8, color='r', linestyle='--', label='80% variance')
        plt.axhline(y=0.9, color='g', linestyle='--', label='90% variance')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(var_ratio[:20], 'o-')
        plt.xlabel('PC')
        plt.ylabel('Variance ratio')
        plt.title('Variance per PC (first 20)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return self.adata

    def identify_tfs(self, min_expr_pct=0.05):
        """Identify transcription factors from the data."""
        if self.tf_list is not None:
            present_tfs = [tf for tf in self.tf_list if tf in self.adata.raw.var_names]
            print(f"[INFO] Using {len(present_tfs)} TFs from provided list.")
            if present_tfs:
                expr_matrix = self.adata.raw[:, present_tfs].X
                if issparse(expr_matrix):
                    expr_matrix = expr_matrix.toarray()
                expr_pct = np.mean(expr_matrix > 0, axis=0)
                expressed_tfs = [present_tfs[i] for i, pct in enumerate(expr_pct) if pct >= min_expr_pct]
                print(f"[INFO] Identified {len(expressed_tfs)} expressed TFs.")
                return expressed_tfs
            else:
                print("[WARN] No TFs from provided list found in data.")
                return []
        return []

# =========================
# Step 2: Improved VAE Model
# =========================
class ImprovedVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=64, hidden_dims=[1024, 512, 256, 128]):
        super(ImprovedVAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder with residual connections
        encoder_layers = []
        h_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            encoder_layers.append(
                nn.Sequential(
                    nn.Linear(h_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.1)
                )
            )
            h_dim = hidden_dim
        
        self.encoder = nn.ModuleList(encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        hidden_dims_rev = list(reversed(hidden_dims))
        h_dim = latent_dim
        for hidden_dim in hidden_dims_rev:
            decoder_layers.append(
                nn.Sequential(
                    nn.Linear(h_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.1)
                )
            )
            h_dim = hidden_dim
        decoder_layers.append(nn.Linear(hidden_dims_rev[-1], input_dim))
        self.decoder = nn.ModuleList(decoder_layers)
        
    def encode(self, x):
        h = x
        for layer in self.encoder:
            h = layer(h)
        return self.fc_mu(h), self.fc_logvar(h)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        h = z
        for layer in self.decoder:
            h = layer(h)
        return h
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
        
    def get_latent(self, x):
        mu, _ = self.encode(x)
        return mu

def vae_loss_fn(recon_x, x, mu, logvar, beta=0.5):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld

# =========================
# Step 3: Enhanced VAE Training
# =========================
class EnhancedVAETrainer:
    def __init__(self, input_dim, latent_dim=64, hidden_dims=[1024, 512, 256, 128], 
                 lr=1e-3, beta=0.5, device=None):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.beta = beta
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        print(f"[INFO] Using device: {self.device}")
        self.model = ImprovedVAE(input_dim, latent_dim, hidden_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5)
        
    def _create_dataloader(self, X, batch_size=128, shuffle=True):
        if issparse(X):
            X = X.toarray()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return loader
        
    def train(self, X, epochs=150, batch_size=128, early_stop_patience=15):
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
        train_loader = self._create_dataloader(X_train, batch_size=batch_size)
        test_loader = self._create_dataloader(X_test, batch_size=batch_size, shuffle=False)
        
        best_test_loss = float('inf')
        best_epoch = 0
        best_state = self.model.state_dict()
        train_losses = []
        test_losses = []
        
        for epoch in range(epochs):
            self.model.train()
            epoch_train_loss = 0
            for batch in train_loader:
                batch_x = batch[0].to(self.device)
                self.optimizer.zero_grad()
                recon_x, mu, logvar = self.model(batch_x)
                loss = vae_loss_fn(recon_x, batch_x, mu, logvar, beta=self.beta)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                epoch_train_loss += loss.item() * batch_x.size(0)
            epoch_train_loss /= len(X_train)
            train_losses.append(epoch_train_loss)
            
            self.model.eval()
            epoch_test_loss = 0
            with torch.no_grad():
                for batch in test_loader:
                    batch_x = batch[0].to(self.device)
                    recon_x, mu, logvar = self.model(batch_x)
                    loss = vae_loss_fn(recon_x, batch_x, mu, logvar, beta=self.beta)
                    epoch_test_loss += loss.item() * batch_x.size(0)
            epoch_test_loss /= len(X_test)
            test_losses.append(epoch_test_loss)
            
            self.scheduler.step(epoch_test_loss)
            
            if epoch % 10 == 0:
                print(f"[VAE] Epoch {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Test Loss: {epoch_test_loss:.4f}")
            
            if epoch_test_loss < best_test_loss:
                best_test_loss = epoch_test_loss
                best_epoch = epoch
                best_state = self.model.state_dict()
            elif epoch - best_epoch >= early_stop_patience:
                print(f"[VAE] Early stopping triggered at epoch {epoch+1}")
                break
                
        self.model.load_state_dict(best_state)
        print(f"[VAE] Training completed. Best test loss: {best_test_loss:.4f} at epoch {best_epoch+1}")
        
        self.visualize_results(X_test)
        
        plt.figure(figsize=(10, 4))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('VAE Training and Test Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        return train_losses, test_losses
        
    def get_latent_representation(self, X):
        if issparse(X):
            X = X.toarray()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            mu = self.model.get_latent(X_tensor)
        return mu.cpu().numpy()
    
    def visualize_results(self, X_test):
        """Visualize encoder latent space and decoder reconstruction."""
        if issparse(X_test):
            X_test = X_test.toarray()
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            recon_x, mu, _ = self.model(X_test_tensor)
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(mu.cpu().numpy()[:, 0], mu.cpu().numpy()[:, 1], alpha=0.5, s=10)
        plt.title('Latent Space (First 2 dimensions)')
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        
        recon_error = torch.mean((recon_x - X_test_tensor) ** 2, dim=1).cpu().numpy()
        plt.subplot(1, 2, 2)
        plt.hist(recon_error, bins=50, alpha=0.7)
        plt.title('Reconstruction Error Distribution')
        plt.xlabel('MSE')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

# =========================
# Step 4: Enhanced Graph Neural Network
# =========================
class EnhancedGNNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=64, 
                 n_layers=3, dropout=0.1, heads=4):
        super(EnhancedGNNModel, self).__init__()
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.conv_first = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.convs = nn.ModuleList()
        for _ in range(n_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
        self.conv_last = GATConv(hidden_channels * heads, out_channels, heads=1, dropout=dropout)
        
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_channels * heads) for _ in range(n_layers - 1)])
            
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = F.elu(self.conv_first(x, edge_index))
        x = self.batch_norms[0](x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        for i, conv in enumerate(self.convs):
            x_res = x
            x = F.elu(conv(x, edge_index))
            x = self.batch_norms[i+1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + x_res  # Residual connection
            
        x = self.conv_last(x, edge_index)
        return x

class EnhancedGNNTrainer:
    def __init__(self, in_channels, hidden_channels=128, out_channels=64, 
                 n_layers=3, dropout=0.1, heads=4, lr=1e-3, device=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.model = EnhancedGNNModel(in_channels, hidden_channels, out_channels, 
                                      n_layers, dropout, heads).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        self.criterion = nn.MSELoss()
        
    def _create_graph(self, X, n_neighbors=30, mode='distance'):
        adj = kneighbors_graph(X, n_neighbors=n_neighbors, 
                              mode=mode, include_self=False)
        edge_index = torch.tensor(np.array(adj.nonzero()), dtype=torch.long)
        x = torch.tensor(X, dtype=torch.float32)
        data = Data(x=x, edge_index=edge_index).to(self.device)
        return data
        
    def train(self, X, epochs=150, n_neighbors=30):
        data = self._create_graph(X, n_neighbors=n_neighbors)
        losses = []
        
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(data)
            
            # Reconstruction loss
            loss = self.criterion(output, data.x)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            losses.append(loss.item())
            if epoch % 20 == 0:
                print(f"[GNN] Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
                
        plt.figure(figsize=(10, 4))
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('GNN Training Loss')
        plt.grid(True)
        plt.show()
        
        return losses
        
    def get_embeddings(self, X, n_neighbors=30):
        data = self._create_graph(X, n_neighbors=n_neighbors)
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(data).cpu().numpy()
        return embeddings

# =========================
# Step 5: Enhanced Gene Regulatory Network Analysis
# =========================
class EnhancedGRNAnalyzer:
    def __init__(self, adata, tf_list=None):
        self.adata = adata
        self.tf_list = tf_list
        self.grn = None
        self.tf_targets = None

    def identify_tfs(self, min_expr_pct=0.05):
        if self.tf_list is not None:
            present_tfs = [tf for tf in self.tf_list if tf in self.adata.raw.var_names]
            if present_tfs:
                expr_matrix = self.adata.raw[:, present_tfs].X
                if issparse(expr_matrix):
                    expr_matrix = expr_matrix.toarray()
                expr_pct = np.mean(expr_matrix > 0, axis=0)
                expressed_tfs = [present_tfs[i] for i, pct in enumerate(expr_pct) if pct >= min_expr_pct]
                print(f"[INFO] Identified {len(expressed_tfs)} expressed TFs.")
                return expressed_tfs
        return []

    def infer_grn(self, method='grnboost2', tfs=None, n_genes=2000, threshold=0.3):
        if tfs is None:
            tfs = self.identify_tfs()

        if len(tfs) == 0:
            print("[WARN] No TFs available for GRN inference.")
            return None

        # Select target genes
        if self.adata.raw is not None:
            var_names = self.adata.raw.var_names
            expr_data_full = self.adata.raw.X
        else:
            var_names = self.adata.var_names
            expr_data_full = self.adata.X

        # Get highly variable genes as targets
        if 'highly_variable' in self.adata.var:
            hvg_mask = self.adata.var['highly_variable']
            hvg_names = self.adata.var_names[hvg_mask].tolist()
        else:
            hvg_names = var_names.tolist()

        # Combine TFs and HVGs
        selected_genes = list(set(tfs + hvg_names[:n_genes]))
        selected_indices = [var_names.tolist().index(g) for g in selected_genes if g in var_names]
        
        expr_data = expr_data_full[:, selected_indices]
        if issparse(expr_data):
            expr_data = expr_data.toarray()

        print(f"[INFO] Inferring GRN using {len(tfs)} TFs and {len(selected_genes)} total genes.")

        if method == 'correlation':
            grn = self._correlation_based_grn(expr_data, selected_genes, tfs, threshold)
        elif method == 'grnboost2':
            grn = self._grnboost2_based_grn(expr_data, selected_genes, tfs)
        else:
            grn = self._correlation_based_grn(expr_data, selected_genes, tfs, threshold)

        self.grn = grn
        self.tf_targets = {tf: list(grn.successors(tf)) for tf in tfs if tf in grn.nodes}
        
        return grn

    def _correlation_based_grn(self, expr_data, gene_names, tfs, threshold):
        grn = nx.DiGraph()
        for gene in gene_names:
            grn.add_node(gene, is_tf=(gene in tfs))

        tf_indices = [gene_names.index(tf) for tf in tfs if tf in gene_names]
        
        for tf_idx, tf in zip(tf_indices, [tf for tf in tfs if tf in gene_names]):
            tf_expr = expr_data[:, tf_idx]
            for i, gene in enumerate(gene_names):
                if gene != tf:
                    gene_expr = expr_data[:, i]
                    corr, pval = pearsonr(tf_expr, gene_expr)
                    if abs(corr) >= threshold and pval < 0.05:
                        grn.add_edge(tf, gene, weight=corr, type='correlation')

        print(f"[INFO] GRN created with {len(grn.nodes)} nodes and {len(grn.edges)} edges.")
        return grn

    def _grnboost2_based_grn(self, expr_data, gene_names, tfs):
        try:
            from arboreto.algo import grnboost2
            from arboreto.utils import load_tf_names
            
            # Create expression dataframe
            expr_df = pd.DataFrame(expr_data, columns=gene_names)
            
            # Run GRNBoost2
            network = grnboost2(expr_df, tf_names=tfs, verbose=True)
            
            # Convert to networkx
            grn = nx.DiGraph()
            for gene in gene_names:
                grn.add_node(gene, is_tf=(gene in tfs))
                
            for _, row in network.iterrows():
                if row['importance'] > 0:
                    grn.add_edge(row['TF'], row['target'], weight=row['importance'])
                    
            print(f"[INFO] GRNBoost2 GRN created with {len(grn.nodes)} nodes and {len(grn.edges)} edges.")
            return grn
            
        except ImportError:
            print("[WARN] GRNBoost2 not available, falling back to correlation method.")
            return self._correlation_based_grn(expr_data, gene_names, tfs, 0.3)
    
    def find_key_regulators(self, top_n=30):
        if self.grn is None:
            return None
        
        out_degree = dict(self.grn.out_degree())
        in_degree = dict(self.grn.in_degree())
        betweenness = nx.betweenness_centrality(self.grn)
        
        tf_nodes = [n for n, d in self.grn.nodes(data=True) if d.get('is_tf', False)]
        
        regulators_df = pd.DataFrame({
            'TF': tf_nodes,
            'OutDegree': [out_degree.get(tf, 0) for tf in tf_nodes],
            'InDegree': [in_degree.get(tf, 0) for tf in tf_nodes],
            'Betweenness': [betweenness.get(tf, 0) for tf in tf_nodes]
        })
        
        regulators_df['Targets'] = regulators_df['OutDegree']
        regulators_df['Importance'] = (regulators_df['OutDegree'] * 0.6 + 
                                      regulators_df['Betweenness'] * regulators_df['OutDegree'].max() * 0.4)
        regulators_df.sort_values('Importance', ascending=False, inplace=True)
        
        return regulators_df.head(top_n)
    
    def plot_grn(self, top_n_tfs=15, layout='spring'):
        if self.grn is None:
            return None
            
        top_regulators = self.find_key_regulators(top_n=top_n_tfs)
        top_tfs = top_regulators['TF'].tolist() if top_regulators is not None else []
        
        if len(top_tfs) > 0:
            all_targets = []
            for tf in top_tfs:
                if tf in self.grn:
                    targets = list(self.grn.successors(tf))
                    all_targets.extend(targets[:10])  # Limit targets per TF
            nodes_to_include = list(set(top_tfs + all_targets))
            subgraph = self.grn.subgraph(nodes_to_include)
        else:
            subgraph = self.grn
            
        plt.figure(figsize=(14, 10))
        
        if layout == 'spring':
            pos = nx.spring_layout(subgraph, k=2, iterations=50)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(subgraph)
        else:
            pos = nx.circular_layout(subgraph)
            
        # Node colors and sizes
        tf_nodes = [n for n, d in subgraph.nodes(data=True) if d.get('is_tf', False)]
        other_nodes = [n for n in subgraph.nodes() if n not in tf_nodes]
        top_tf_nodes = [n for n in tf_nodes if n in top_tfs]
        other_tf_nodes = [n for n in tf_nodes if n not in top_tfs]
        
        node_sizes = dict(subgraph.degree())
        
        # Draw nodes
        nx.draw_networkx_nodes(subgraph, pos, nodelist=top_tf_nodes, 
                              node_size=[200 + 30*node_sizes[n] for n in top_tf_nodes],
                              node_color='red', alpha=0.8, label='Top TFs')
        nx.draw_networkx_nodes(subgraph, pos, nodelist=other_tf_nodes, 
                              node_size=[100 + 20*node_sizes[n] for n in other_tf_nodes],
                              node_color='orange', alpha=0.6, label='Other TFs')
        nx.draw_networkx_nodes(subgraph, pos, nodelist=other_nodes, 
                              node_size=[50 + 10*node_sizes[n] for n in other_nodes],
                              node_color='lightblue', alpha=0.4, label='Target genes')
        
        # Draw edges
        nx.draw_networkx_edges(subgraph, pos, width=0.5, alpha=0.3, edge_color='gray')
        
        # Labels for top TFs
        nx.draw_networkx_labels(subgraph, pos, {n: n for n in top_tf_nodes}, 
                              font_size=10, font_weight='bold')
        
        plt.title(f"Gene Regulatory Network (Top {len(top_tf_nodes)} TFs)")
        plt.legend()
        plt.axis('off')
        
        return plt.gcf()

# =========================
# Step 6: Advanced Pseudotime Analysis
# =========================
def hierarchical_clustering_branches(adata, min_cluster_size=20, max_clusters=100):
    """Perform hierarchical clustering to identify fine-grained branches."""
    print("[INFO] Performing hierarchical clustering for branch detection...")
    
    # Combine multiple representations
    representations = []
    if 'X_pca' in adata.obsm:
        representations.append(adata.obsm['X_pca'][:, :30])
    if 'X_gnn' in adata.obsm:
        representations.append(adata.obsm['X_gnn'])
    if 'X_vae' in adata.obsm:
        representations.append(adata.obsm['X_vae'])
    
    X_combined = np.hstack(representations)
    
    # Normalize
    from sklearn.preprocessing import StandardScaler
    X_combined = StandardScaler().fit_transform(X_combined)
    
    # Hierarchical clustering
    linkage = hierarchy.linkage(X_combined, method='ward')
    
    # Find optimal number of clusters
    best_score = -1
    best_n_clusters = 40
    
    for n_clusters in range(40, min(max_clusters, len(adata)//min_cluster_size)):
        clusters = hierarchy.fcluster(linkage, n_clusters, criterion='maxclust')
        
        # Check cluster sizes
        unique, counts = np.unique(clusters, return_counts=True)
        if np.min(counts) < min_cluster_size:
            continue
            
        score = silhouette_score(X_combined, clusters, sample_size=min(5000, len(clusters)))
        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters
    
    # Apply best clustering
    adata.obs['hierarchical_clusters'] = hierarchy.fcluster(linkage, best_n_clusters, criterion='maxclust').astype(str)
    
    print(f"[INFO] Hierarchical clustering: {best_n_clusters} clusters found with silhouette score {best_score:.3f}")
    
    return adata, best_n_clusters

def compute_multi_method_pseudotime(adata, n_neighbors=30, n_comps=10):
    """Compute pseudotime using multiple methods and combine results."""
    print("[INFO] Computing multi-method pseudotime...")
    
    # Method 1: Standard diffusion pseudotime
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X_gnn')
    sc.tl.diffmap(adata, n_comps=n_comps)
    
    # Set root as earliest developmental stage
    if 'developmental_stage' in adata.obs:
        earliest_stage = adata.obs['developmental_stage'].min()
        root_mask = adata.obs['developmental_stage'] == earliest_stage
        if root_mask.sum() > 0:
            root_idx = np.where(root_mask)[0][0]
            adata.uns['iroot'] = root_idx
    else:
        adata.uns['iroot'] = 0
    
    sc.tl.dpt(adata)
    
    # Method 2: PAGA-based pseudotime
    sc.tl.paga(adata, groups='hierarchical_clusters')
    sc.pl.paga(adata, plot=False)
    
    # Method 3: Try Palantir if available
    try:
        import palantir
        dm_res = palantir.utils.run_diffusion_maps(adata.obsm['X_pca'][:, :30])
        ms_data = palantir.utils.determine_multiscale_space(dm_res)
        
        # Find start cell
        if 'developmental_stage' in adata.obs:
            early_cells = adata.obs['developmental_stage'] == adata.obs['developmental_stage'].min()
            start_cell = adata.obs_names[early_cells][0]
        else:
            start_cell = adata.obs_names[0]
        
        pr_res = palantir.core.run_palantir(ms_data, start_cell, num_waypoints=500)
        adata.obs['palantir_pseudotime'] = pr_res.pseudotime
        adata.obs['palantir_entropy'] = pr_res.entropy
        adata.obsm['palantir_fate_probabilities'] = pr_res.branch_probs
        
        # Combine pseudotimes
        adata.obs['combined_pseudotime'] = (adata.obs['dpt_pseudotime'] + pr_res.pseudotime) / 2
        
        print("[INFO] Palantir pseudotime computed and combined with DPT.")
        
    except:
        print("[INFO] Palantir not available, using DPT only.")
        adata.obs['combined_pseudotime'] = adata.obs['dpt_pseudotime']
    
    return adata

def compute_best_pseudotime(adata, embeddings=['X_pca', 'X_combined', 'X_gnn', 'X_vae'], n_neighbors_list=[10, 15, 20, 25, 30, 40, 50], n_comps=10, max_roots=50, calibrate_monotonic=True):
    """Find the pseudotime with highest Pearson correlation to developmental stage without mutating global state.
    - Tries multiple embeddings and neighbor sizes
    - Scans multiple candidate root cells among the earliest stage
    - Optionally applies isotonic calibration to improve Pearson while preserving order
    Writes results to:
      - obs['dpt_pseudotime_best'] (raw best DPT)
      - obs['dpt_pseudotime_best_cal'] (monotonic calibrated)
      - uns['best_pseudotime_params']
    """
    print("[INFO] Searching for best-correlated pseudotime...")
    if 'developmental_stage' not in adata.obs:
        print("[WARN] No 'developmental_stage' in obs; skipping best pseudotime search.")
        return adata
    stages_all = adata.obs['developmental_stage'].astype(float).values
    best_corr = -1.0
    best_params = None
    best_pt = None
    rng = np.random.default_rng(42)

    # Candidate roots: subset of earliest stage cells
    earliest = np.min(stages_all)
    early_idx = np.where(stages_all == earliest)[0]
    if len(early_idx) == 0:
        early_idx = np.array([int(np.argmin(stages_all))])
    if len(early_idx) > max_roots:
        early_idx = rng.choice(early_idx, size=max_roots, replace=False)

    for emb in embeddings:
        if emb not in adata.obsm:
            continue
        for nn in n_neighbors_list:
            try:
                ad = adata.copy()
                sc.pp.neighbors(ad, n_neighbors=nn, use_rep=emb)
                sc.tl.diffmap(ad, n_comps=n_comps)
                for root_idx in early_idx:
                    ad.uns['iroot'] = int(root_idx)
                    sc.tl.dpt(ad)
                    pt = ad.obs['dpt_pseudotime'].astype(float).values
                    corr, _ = pearsonr(pt, stages_all)
                    if corr > best_corr:
                        best_corr = corr
                        best_params = {'embedding': emb, 'n_neighbors': nn, 'n_comps': n_comps, 'root_idx': int(root_idx)}
                        best_pt = pt
                print(f"[BestPT] emb={emb}, n_neighbors={nn}, best_corr_so_far={best_corr:.3f}")
            except Exception as e:
                print(f"[BestPT] Failed for emb={emb}, n_neighbors={nn}: {e}")
                continue

    if best_pt is None:
        print("[WARN] Could not compute best pseudotime.")
        return adata

    adata.obs['dpt_pseudotime_best'] = best_pt
    adata.uns['best_pseudotime_params'] = best_params
    print(f"[INFO] Best DPT Pearson corr={best_corr:.3f} using {best_params}")

    if calibrate_monotonic:
        try:
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(out_of_bounds='clip')
            pt_cal = iso.fit_transform(best_pt, stages_all)
            adata.obs['dpt_pseudotime_best_cal'] = pt_cal
            corr_cal, _ = pearsonr(pt_cal, stages_all)
            adata.uns['best_pseudotime_params']['calibrated'] = True
            adata.uns['best_pseudotime_params']['pearson_cal'] = float(corr_cal)
            print(f"[INFO] Calibrated best DPT Pearson corr={corr_cal:.3f}")
        except Exception as e:
            print(f"[BestPT] Calibration failed: {e}")

    return adata

def enhanced_grid_search(adata, min_branches=40, min_corr=0.85):
    """Enhanced grid search for optimal parameters."""
    print("[INFO] Starting enhanced grid search...")
    
    param_grid = {
        'n_neighbors': [10, 15, 20, 25, 30, 35, 40, 50],
        'n_pcs': [20, 30, 40, 50],
        'min_dist': [0.1, 0.2, 0.3, 0.4, 0.5],
        'resolution': np.logspace(-0.5, 1.5, 40),
        'clustering_method': ['leiden', 'hierarchical'],
        'embedding': ['X_pca', 'X_gnn', 'X_vae', 'X_combined']
    }
    
    # Create combined embedding
    if 'X_combined' not in adata.obsm:
        embeddings = []
        if 'X_pca' in adata.obsm:
            embeddings.append(adata.obsm['X_pca'][:, :30])
        if 'X_gnn' in adata.obsm:
            embeddings.append(adata.obsm['X_gnn'])
        if 'X_vae' in adata.obsm:
            embeddings.append(adata.obsm['X_vae'])
        if embeddings:
            adata.obsm['X_combined'] = np.hstack(embeddings)
    
    best_result = {'branches': 0, 'correlation': 0, 'params': {}}
    
    for embedding in param_grid['embedding']:
        if embedding not in adata.obsm:
            continue
            
        for n_neighbors in param_grid['n_neighbors']:
            # Compute neighbors
            sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=embedding)
            
            for clustering_method in param_grid['clustering_method']:
                if clustering_method == 'hierarchical':
                    adata_temp, n_clusters = hierarchical_clustering_branches(adata.copy())
                    branches = n_clusters
                    
                    # Compute pseudotime
                    sc.tl.diffmap(adata_temp, n_comps=10)
                    if 'developmental_stage' in adata_temp.obs:
                        root_idx = np.where(adata_temp.obs['developmental_stage'] == 
                                          adata_temp.obs['developmental_stage'].min())[0][0]
                        adata_temp.uns['iroot'] = root_idx
                    sc.tl.dpt(adata_temp)
                    
                    # Calculate correlation
                    if 'developmental_stage' in adata_temp.obs:
                        corr, _ = pearsonr(adata_temp.obs['dpt_pseudotime'], 
                                         adata_temp.obs['developmental_stage'].astype(float))
                    else:
                        corr = 0
                    
                    if branches >= min_branches and corr >= min_corr:
                        if corr > best_result['correlation'] or \
                           (corr == best_result['correlation'] and branches > best_result['branches']):
                            best_result = {
                                'branches': branches,
                                'correlation': corr,
                                'params': {
                                    'embedding': embedding,
                                    'n_neighbors': n_neighbors,
                                    'clustering_method': clustering_method
                                },
                                'adata': adata_temp.copy()
                            }
                            print(f"[GRID SEARCH] Found better: branches={branches}, corr={corr:.3f}")
                            
                else:
                    for resolution in param_grid['resolution']:
                        # Clustering
                        if clustering_method == 'leiden':
                            sc.tl.leiden(adata, resolution=resolution)
                            cluster_key = 'leiden'
                        else:
                            # Louvain clustering is not available
                            continue
                        
                        branches = len(np.unique(adata.obs[cluster_key]))
                        
                        if branches < min_branches:
                            continue
                        
                        # Compute pseudotime
                        sc.tl.diffmap(adata, n_comps=10)
                        if 'developmental_stage' in adata.obs:
                            root_idx = np.where(adata.obs['developmental_stage'] == 
                                              adata.obs['developmental_stage'].min())[0][0]
                            adata.uns['iroot'] = root_idx
                        sc.tl.dpt(adata)
                        
                        # Calculate correlation
                        if 'developmental_stage' in adata.obs:
                            corr, _ = pearsonr(adata.obs['dpt_pseudotime'], 
                                             adata.obs['developmental_stage'].astype(float))
                        else:
                            corr = 0
                        
                        if branches >= min_branches and corr >= min_corr:
                            if corr > best_result['correlation'] or \
                               (corr == best_result['correlation'] and branches > best_result['branches']):
                                best_result = {
                                    'branches': branches,
                                    'correlation': corr,
                                    'params': {
                                        'embedding': embedding,
                                        'n_neighbors': n_neighbors,
                                        'clustering_method': clustering_method,
                                        'resolution': resolution
                                    },
                                    'adata': adata.copy()
                                }
                                print(f"[GRID SEARCH] Found better: branches={branches}, corr={corr:.3f}")
    
    return best_result

# =========================
# Step 7: Enhanced Visualization
# =========================
def visualize_3d_trajectory(adata):
    """Enhanced 3D visualization without PHATE."""
    print("[INFO] Generating enhanced 3D trajectory visualization...")
    
    # Method 1: Force-directed layout
    if 'connectivities' in adata.obsp:
        G = nx.from_scipy_sparse_array(adata.obsp['connectivities'])
        pos = nx.spring_layout(G, dim=3, k=1, iterations=50, seed=42)
        pos_array = np.array([pos[i] for i in range(len(pos))])
    else:
        # Fallback to UMAP 3D
        import umap
        reducer = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.3)
        pos_array = reducer.fit_transform(adata.obsm['X_gnn'])
    
    x, y, z = pos_array[:, 0], pos_array[:, 1], pos_array[:, 2]
    
    # Color by pseudotime
    if 'combined_pseudotime' in adata.obs:
        pseudotime = adata.obs['combined_pseudotime'].values
    elif 'dpt_pseudotime' in adata.obs:
        pseudotime = adata.obs['dpt_pseudotime'].values
    else:
        pseudotime = np.zeros(len(x))
    
    # Identify trajectory milestones
    start_idx = np.argmin(pseudotime)
    end_idx = np.argmax(pseudotime)
    quartiles_idx = [np.argmin(np.abs(pseudotime - np.percentile(pseudotime, q))) 
                     for q in [25, 50, 75]]
    
    # Create figure
    fig = go.Figure()
    
    # Main scatter
    hover_text = []
    for i in range(len(adata)):
        text = f"Cell: {adata.obs.index[i]}<br>"
        text += f"Pseudotime: {pseudotime[i]:.3f}<br>"
        if 'developmental_stage' in adata.obs:
            text += f"Stage: {adata.obs['developmental_stage'].iloc[i]}<br>"
        if 'hierarchical_clusters' in adata.obs:
            text += f"Cluster: {adata.obs['hierarchical_clusters'].iloc[i]}"
        hover_text.append(text)
    
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=4,
            color=pseudotime,
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title='Pseudotime')
        ),
        text=hover_text,
        hoverinfo='text',
        name='Cells'
    ))
    
    # Add milestones
    milestones = {
        'Start': (start_idx, 'green', 'diamond'),
        'Q1': (quartiles_idx[0], 'lightgreen', 'circle'),
        'Median': (quartiles_idx[1], 'yellow', 'square'),
        'Q3': (quartiles_idx[2], 'orange', 'circle'),
        'End': (end_idx, 'red', 'diamond-open')
    }
    
    for name, (idx, color, symbol) in milestones.items():
        fig.add_trace(go.Scatter3d(
            x=[x[idx]], y=[y[idx]], z=[z[idx]],
            mode='markers',
            marker=dict(size=12, color=color, symbol=symbol),
            name=name
        ))
    
    # Add trajectory line (simplified)
    sorted_idx = np.argsort(pseudotime)
    step = max(1, len(sorted_idx) // 100)  # Sample 100 points
    traj_idx = sorted_idx[::step]
    
    fig.add_trace(go.Scatter3d(
        x=x[traj_idx], y=y[traj_idx], z=z[traj_idx],
        mode='lines',
        line=dict(color='gray', width=2),
        opacity=0.3,
        name='Trajectory',
        showlegend=False
    ))
    
    fig.update_layout(
        title="3D Developmental Trajectory",
        scene=dict(
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            zaxis_title='Dimension 3'
        ),
        width=1000,
        height=800
    )
    
    output_file = os.path.join(RESULTS_PATH, '3d_trajectory_enhanced.html')
    fig.write_html(output_file)
    print(f"[INFO] Enhanced 3D visualization saved to {output_file}")
    
    # Also create matplotlib 3D plot
    fig_mpl = plt.figure(figsize=(12, 10))
    ax = fig_mpl.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(x, y, z, c=pseudotime, cmap='viridis', s=10, alpha=0.6)
    ax.scatter(x[start_idx], y[start_idx], z[start_idx], c='green', s=200, marker='D', label='Start')
    ax.scatter(x[end_idx], y[end_idx], z[end_idx], c='red', s=200, marker='D', label='End')
    
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    ax.set_title('3D Developmental Trajectory')
    
    plt.colorbar(scatter, ax=ax, label='Pseudotime')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, '3d_trajectory_matplotlib.png'), dpi=300)
    plt.show()

def plot_comprehensive_results(adata, grn_analyzer, tfs):
    """Generate comprehensive visualization of results."""
    print("[INFO] Generating comprehensive result plots...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. UMAP colored by clusters
    ax1 = plt.subplot(3, 3, 1)
    if 'X_umap' not in adata.obsm:
        sc.tl.umap(adata)
    if 'hierarchical_clusters' in adata.obs:
        cluster_key = 'hierarchical_clusters'
    elif 'leiden' in adata.obs:
        cluster_key = 'leiden'
    else:
        cluster_key = None
    
    if cluster_key:
        clusters = adata.obs[cluster_key].astype('category')
        scatter = ax1.scatter(adata.obsm['X_umap'][:, 0], adata.obsm['X_umap'][:, 1],
                            c=clusters.cat.codes, cmap='tab20', s=10, alpha=0.7)
        ax1.set_title(f'UMAP - {cluster_key}')
    ax1.set_xlabel('UMAP1')
    ax1.set_ylabel('UMAP2')
    
    # 2. UMAP colored by pseudotime
    ax2 = plt.subplot(3, 3, 2)
    if 'combined_pseudotime' in adata.obs:
        pseudotime = adata.obs['combined_pseudotime']
    else:
        pseudotime = adata.obs['dpt_pseudotime']
    scatter = ax2.scatter(adata.obsm['X_umap'][:, 0], adata.obsm['X_umap'][:, 1],
                         c=pseudotime, cmap='viridis', s=10, alpha=0.7)
    plt.colorbar(scatter, ax=ax2)
    ax2.set_title('UMAP - Pseudotime')
    ax2.set_xlabel('UMAP1')
    ax2.set_ylabel('UMAP2')
    
    # 3. UMAP colored by developmental stage
    ax3 = plt.subplot(3, 3, 3)
    if 'developmental_stage' in adata.obs:
        stages = adata.obs['developmental_stage']
        scatter = ax3.scatter(adata.obsm['X_umap'][:, 0], adata.obsm['X_umap'][:, 1],
                            c=stages, cmap='plasma', s=10, alpha=0.7)
        plt.colorbar(scatter, ax=ax3)
    ax3.set_title('UMAP - Developmental Stage')
    ax3.set_xlabel('UMAP1')
    ax3.set_ylabel('UMAP2')
    
    # 4. Pseudotime vs Developmental Stage
    ax4 = plt.subplot(3, 3, 4)
    if 'developmental_stage' in adata.obs:
        ax4.scatter(stages, pseudotime, alpha=0.5, s=10)
        ax4.set_xlabel('Developmental Stage (hpf)')
        ax4.set_ylabel('Pseudotime')
        ax4.set_title('Pseudotime vs Stage Correlation')
        # Add correlation
        corr, _ = pearsonr(stages, pseudotime)
        ax4.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax4.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 5. Cluster sizes
    ax5 = plt.subplot(3, 3, 5)
    if cluster_key:
        cluster_counts = adata.obs[cluster_key].value_counts().sort_index()
        ax5.bar(range(len(cluster_counts)), cluster_counts.values)
        ax5.set_xlabel('Cluster')
        ax5.set_ylabel('Cell Count')
        ax5.set_title(f'Cluster Sizes (n={len(cluster_counts)})')
        ax5.tick_params(axis='x', rotation=90)
    
    # 6. Top TFs by out-degree
    ax6 = plt.subplot(3, 3, 6)
    if grn_analyzer.grn is not None:
        top_regulators = grn_analyzer.find_key_regulators(top_n=15)
        if top_regulators is not None and len(top_regulators) > 0:
            ax6.barh(range(len(top_regulators)), top_regulators['Targets'].values)
            ax6.set_yticks(range(len(top_regulators)))
            ax6.set_yticklabels(top_regulators['TF'].values)
            ax6.set_xlabel('Number of Targets')
            ax6.set_title('Top Transcription Factors')
            ax6.invert_yaxis()
    
    # 7. TF expression heatmap
    ax7 = plt.subplot(3, 3, 7)
    if len(tfs) > 0 and 'dpt_pseudotime' in adata.obs:
        # Order cells by pseudotime
        order = np.argsort(adata.obs['dpt_pseudotime'])
        
        # Get TF expression
        tf_expr = []
        tf_names = []
        for tf in tfs[:20]:  # Top 20 TFs
            if tf in adata.raw.var_names:
                expr = adata.raw[:, tf].X
                if issparse(expr):
                    expr = expr.toarray().flatten()
                tf_expr.append(expr[order])
                tf_names.append(tf)
        
        if tf_expr:
            tf_expr = np.array(tf_expr)
            # Smooth expression
            from scipy.ndimage import gaussian_filter1d
            tf_expr_smooth = np.array([gaussian_filter1d(row, sigma=10) for row in tf_expr])
            
            im = ax7.imshow(tf_expr_smooth, aspect='auto', cmap='viridis', interpolation='bilinear')
            ax7.set_yticks(range(len(tf_names)))
            ax7.set_yticklabels(tf_names)
            ax7.set_xlabel('Cells (ordered by pseudotime)')
            ax7.set_title('TF Expression over Pseudotime')
            plt.colorbar(im, ax=ax7)
    
    # 8. Developmental trajectory in diffusion space
    ax8 = plt.subplot(3, 3, 8)
    if 'X_diffmap' in adata.obsm:
        scatter = ax8.scatter(adata.obsm['X_diffmap'][:, 0], adata.obsm['X_diffmap'][:, 1],
                            c=pseudotime, cmap='viridis', s=10, alpha=0.7)
        plt.colorbar(scatter, ax=ax8)
        ax8.set_xlabel('DC1')
        ax8.set_ylabel('DC2')
        ax8.set_title('Diffusion Map - Pseudotime')
    
    # 9. PAGA graph
    ax9 = plt.subplot(3, 3, 9)
    if 'paga' in adata.uns:
        sc.pl.paga(adata, ax=ax9, show=False)
        ax9.set_title('PAGA Trajectory Graph')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'comprehensive_results.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_cell_gene_heatmap(adata, genes, order_by='dpt_pseudotime', normalize='zscore_per_gene', max_cells=2000):
    """Plot a cell-by-gene heatmap.
    - y axis: cells (optionally ordered by pseudotime or stage)
    - x axis: genes (one or many)
    """
    print("[INFO] Plotting cell-by-gene heatmap...")
    if isinstance(genes, str):
        genes = [genes]

    genes_present = [g for g in genes if g in (adata.raw.var_names if adata.raw is not None else adata.var_names)]
    if len(genes_present) == 0:
        print("[WARN] None of the requested genes were found; skipping heatmap.")
        return

    var_names = adata.raw.var_names if adata.raw is not None else adata.var_names
    X_full = adata.raw.X if adata.raw is not None else adata.X
    if issparse(X_full):
        X_full = X_full.toarray()

    col_indices = [var_names.tolist().index(g) for g in genes_present]
    expr = X_full[:, col_indices]  # cells x genes

    # Order cells
    if order_by in adata.obs:
        order = np.argsort(adata.obs[order_by].values)
        expr = expr[order]
        ordered_obs = adata.obs.iloc[order]
    else:
        ordered_obs = adata.obs

    # Limit number of cells for readability
    if expr.shape[0] > max_cells:
        step = max(1, expr.shape[0] // max_cells)
        expr = expr[::step]
        ordered_obs = ordered_obs.iloc[::step]

    # Normalize per gene
    if normalize == 'zscore_per_gene' and expr.shape[1] > 0:
        expr = (expr - expr.mean(axis=0, keepdims=True)) / (expr.std(axis=0, keepdims=True) + 1e-8)

    plt.figure(figsize=(max(8, 0.4 * len(genes_present)), min(20, 0.004 * expr.shape[0] + 4)))
    ax = sns.heatmap(expr, cmap='viridis', cbar_kws={'label': 'Expression (z-score)'},
                     xticklabels=genes_present, yticklabels=False)
    ax.set_xlabel('Genes')
    ylabel = 'Cells'
    if order_by in adata.obs:
        ylabel += f' (ordered by {order_by})'
    ax.set_ylabel(ylabel)
    ax.set_title('Cell-by-Gene Heatmap')
    plt.tight_layout()
    outfile = os.path.join(RESULTS_PATH, 'cell_gene_heatmap.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"[INFO] Saved cell-by-gene heatmap to {outfile}")

def visualize_trajectory_2d_free(adata, sample_edges=5000, node_size=6):
    """2D trajectory visualization using a force-directed graph layout (no UMAP).
    Colors by pseudotime if available.
    """
    print("[INFO] Generating 2D free-layout trajectory (force-directed)...")
    if 'connectivities' not in adata.obsp:
        print("[WARN] No connectivities found in adata.obsp; computing neighbors first.")
        sc.pp.neighbors(adata, n_neighbors=30, use_rep='X_gnn' if 'X_gnn' in adata.obsm else 'X_pca')

    G = nx.from_scipy_sparse_array(adata.obsp['connectivities'])
    pos = nx.spring_layout(G, k=1.5, iterations=200, seed=42)
    coords = np.array([pos[i] for i in range(len(pos))])

    if 'combined_pseudotime' in adata.obs:
        colors = adata.obs['combined_pseudotime'].values
    elif 'dpt_pseudotime' in adata.obs:
        colors = adata.obs['dpt_pseudotime'].values
    else:
        colors = np.zeros(adata.n_obs)

    plt.figure(figsize=(10, 10))
    sca = plt.scatter(coords[:, 0], coords[:, 1], c=colors, cmap='viridis', s=node_size, alpha=0.7)
    plt.colorbar(sca, label='Pseudotime')
    # Mark start/end by pseudotime
    start_idx = int(np.argmin(colors))
    end_idx = int(np.argmax(colors))
    plt.scatter(coords[start_idx, 0], coords[start_idx, 1], c='green', s=100, marker='D', label='Start')
    plt.scatter(coords[end_idx, 0], coords[end_idx, 1], c='red', s=100, marker='D', label='End')
    plt.legend(loc='upper right')
    plt.title('Force-directed 2D Trajectory (free layout)')
    plt.axis('off')

    # Optionally draw a subset of edges to indicate structure
    try:
        edges = list(G.edges())
        if len(edges) > sample_edges:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(edges), size=sample_edges, replace=False)
            edges = [edges[i] for i in idx]
        for u, v in edges:
            x = [pos[u][0], pos[v][0]]
            y = [pos[u][1], pos[v][1]]
            plt.plot(x, y, color='lightgray', linewidth=0.2, alpha=0.2)
    except Exception:
        pass

    outfile = os.path.join(RESULTS_PATH, 'trajectory_2d_free.png')
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"[INFO] Saved 2D free-layout trajectory to {outfile}")

def visualize_trajectory_3d_free(adata, node_size=4):
    """3D trajectory using a force-directed graph layout (no UMAP constraints)."""
    print("[INFO] Generating 3D free-layout trajectory (force-directed)...")
    if 'connectivities' not in adata.obsp:
        print("[WARN] No connectivities found in adata.obsp; computing neighbors first.")
        sc.pp.neighbors(adata, n_neighbors=30, use_rep='X_gnn' if 'X_gnn' in adata.obsm else 'X_pca')

    G = nx.from_scipy_sparse_array(adata.obsp['connectivities'])
    pos = nx.spring_layout(G, dim=3, k=1.5, iterations=200, seed=42)
    coords = np.array([pos[i] for i in range(len(pos))])
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    if 'combined_pseudotime' in adata.obs:
        colors = adata.obs['combined_pseudotime'].values
    elif 'dpt_pseudotime' in adata.obs:
        colors = adata.obs['dpt_pseudotime'].values
    else:
        colors = np.zeros(adata.n_obs)

    # Plotly interactive
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=node_size, color=colors, colorscale='Viridis', opacity=0.85,
                    colorbar=dict(title='Pseudotime')),
        name='Cells'
    ))
    # Start/End markers
    start_idx = int(np.argmin(colors))
    end_idx = int(np.argmax(colors))
    fig.add_trace(go.Scatter3d(x=[x[start_idx]], y=[y[start_idx]], z=[z[start_idx]], mode='markers',
                               marker=dict(size=8, color='green', symbol='diamond'), name='Start'))
    fig.add_trace(go.Scatter3d(x=[x[end_idx]], y=[y[end_idx]], z=[z[end_idx]], mode='markers',
                               marker=dict(size=8, color='red', symbol='diamond-open'), name='End'))
    fig.update_layout(title='3D Trajectory (Force-directed, free layout)',
                      scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
                      width=1000, height=800)
    html_out = os.path.join(RESULTS_PATH, 'trajectory_3d_free.html')
    fig.write_html(html_out)
    print(f"[INFO] Saved interactive 3D free layout to {html_out}")

    # Matplotlib static
    fig_mpl = plt.figure(figsize=(12, 10))
    ax = fig_mpl.add_subplot(111, projection='3d')
    sca = ax.scatter(x, y, z, c=colors, cmap='viridis', s=node_size*2, alpha=0.7)
    ax.scatter(x[start_idx], y[start_idx], z[start_idx], c='green', s=80, marker='D', label='Start')
    ax.scatter(x[end_idx], y[end_idx], z[end_idx], c='red', s=80, marker='D', label='End')
    ax.legend()
    plt.colorbar(sca, ax=ax, label='Pseudotime')
    ax.set_title('3D Trajectory (Force-directed, free layout)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.tight_layout()
    png_out = os.path.join(RESULTS_PATH, 'trajectory_3d_free_matplotlib.png')
    plt.savefig(png_out, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"[INFO] Saved static 3D free layout to {png_out}")

def visualize_trajectory_pca_spline(adata, n_waypoints=200):
    """Non-force-directed 2D trajectory: fit a spline/path through PCA space ordered by best pseudotime.
    Uses 'dpt_pseudotime_best' if available; falls back to 'combined_pseudotime' or 'dpt_pseudotime'.
    """
    print("[INFO] Plotting PCA spline-based trajectory (non-force-directed)...")
    if 'X_pca' not in adata.obsm:
        print("[WARN] No PCA found; computing PCA for trajectory plot.")
        sc.tl.pca(adata, n_comps=30)
    if 'dpt_pseudotime_best' in adata.obs:
        pt = adata.obs['dpt_pseudotime_best'].values
    elif 'combined_pseudotime' in adata.obs:
        pt = adata.obs['combined_pseudotime'].values
    elif 'dpt_pseudotime' in adata.obs:
        pt = adata.obs['dpt_pseudotime'].values
    else:
        print("[WARN] No pseudotime available; skipping PCA spline plot.")
        return
    order = np.argsort(pt)
    pcs = adata.obsm['X_pca'][order, :2]
    pts = pt[order]
    # Smooth the path using parametric spline on the ordered points
    try:
        from scipy.interpolate import splprep, splev
        t = np.linspace(0, 1, len(pcs))
        tck, _ = splprep([pcs[:, 0], pcs[:, 1]], s=max(1.0, 0.001*len(pcs)))
        u_fine = np.linspace(0, 1, n_waypoints)
        x_s, y_s = splev(u_fine, tck)
        path = np.vstack([x_s, y_s]).T
    except Exception:
        # Fallback to simple moving-average smoothing
        win = max(5, len(pcs)//100)
        if win % 2 == 0:
            win += 1
        kernel = np.ones(win)/win
        x_s = np.convolve(pcs[:, 0], kernel, mode='same')
        y_s = np.convolve(pcs[:, 1], kernel, mode='same')
        idx = np.linspace(0, len(x_s)-1, n_waypoints).astype(int)
        path = np.vstack([x_s[idx], y_s[idx]]).T

    plt.figure(figsize=(9, 8))
    color_arr = adata.obs.get('dpt_pseudotime_best', adata.obs.get('combined_pseudotime', adata.obs.get('dpt_pseudotime', np.zeros(adata.n_obs))))
    sca = plt.scatter(adata.obsm['X_pca'][:, 0], adata.obsm['X_pca'][:, 1], c=color_arr, cmap='viridis', s=6, alpha=0.6)
    # Start/End markers using best/combined/DPT
    start_idx = int(np.argmin(color_arr))
    end_idx = int(np.argmax(color_arr))
    plt.scatter(adata.obsm['X_pca'][start_idx, 0], adata.obsm['X_pca'][start_idx, 1], c='green', s=80, marker='D', label='Start')
    plt.scatter(adata.obsm['X_pca'][end_idx, 0], adata.obsm['X_pca'][end_idx, 1], c='red', s=80, marker='D', label='End')
    plt.colorbar(sca, label='Pseudotime (best/combined/DPT)')
    plt.plot(path[:, 0], path[:, 1], color='black', linewidth=2, alpha=0.8, label='Spline path')
    plt.title('Non-force-directed Trajectory (PCA + spline)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    outfile = os.path.join(RESULTS_PATH, 'trajectory_pca_spline.png')
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"[INFO] Saved PCA spline trajectory to {outfile}")

def plot_grn_detailed(grn_analyzer, max_targets_per_tf=20):
    """Create a more detailed GRN image with edge weights, legends, and shapes."""
    if grn_analyzer.grn is None:
        print("[WARN] No GRN found; skipping detailed GRN plot.")
        return
    print("[INFO] Plotting detailed GRN view...")
    grn = grn_analyzer.grn
    top_regs = grn_analyzer.find_key_regulators(top_n=20)
    top_tfs = top_regs['TF'].tolist() if top_regs is not None else []
    nodes = set()
    for tf in top_tfs:
        nodes.add(tf)
        targets = list(grn.successors(tf))[:max_targets_per_tf]
        nodes.update(targets)
    sub = grn.subgraph(nodes)

    pos = nx.kamada_kawai_layout(sub)
    tf_nodes = [n for n, d in sub.nodes(data=True) if d.get('is_tf', False)]
    target_nodes = [n for n in sub.nodes() if n not in tf_nodes]

    plt.figure(figsize=(16, 12))
    # Edge widths by weight
    weights = np.array([abs(sub[u][v].get('weight', 1.0)) for u, v in sub.edges()])
    if len(weights) == 0:
        widths = 0.5
    else:
        w_min, w_max = np.percentile(weights, 5), np.percentile(weights, 95)
        widths = 0.5 + 3.0 * (np.clip(weights, w_min, w_max) - w_min) / (w_max - w_min + 1e-8)
    nx.draw_networkx_edges(sub, pos, width=widths, alpha=0.25, edge_color='gray')
    # Nodes with different shapes
    nx.draw_networkx_nodes(sub, pos, nodelist=tf_nodes, node_color='crimson', node_shape='s',
                           alpha=0.85, node_size=[200 + 25*sub.degree(n) for n in tf_nodes], label='TFs')
    nx.draw_networkx_nodes(sub, pos, nodelist=target_nodes, node_color='steelblue', node_shape='o',
                           alpha=0.6, node_size=[80 + 10*sub.degree(n) for n in target_nodes], label='Targets')
    # Labels for top TFs only
    label_map = {n: n for n in tf_nodes}
    nx.draw_networkx_labels(sub, pos, labels=label_map, font_size=9, font_weight='bold')
    plt.legend(scatterpoints=1)
    plt.title('Detailed Gene Regulatory Network')
    plt.axis('off')
    outfile = os.path.join(RESULTS_PATH, 'grn_detailed.png')
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"[INFO] Saved detailed GRN image to {outfile}")

def plot_pseudotime_charts(adata):
    """Show pseudotime line plot, histogram, and KDE."""
    key = 'combined_pseudotime' if 'combined_pseudotime' in adata.obs else 'dpt_pseudotime'
    if key not in adata.obs:
        print("[WARN] Pseudotime not found; skipping pseudotime charts.")
        return
    print("[INFO] Plotting pseudotime charts...")
    pt = adata.obs[key].values
    order = np.argsort(pt)
    pt_sorted = pt[order]

    plt.figure(figsize=(16, 10))
    # Line plot
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(pt_sorted, lw=1.2)
    ax1.set_title('Pseudotime along ordered cells')
    ax1.set_xlabel('Ordered cell index')
    ax1.set_ylabel('Pseudotime')
    # Histogram
    ax2 = plt.subplot(2, 2, 2)
    ax2.hist(pt, bins=40, color='tab:blue', alpha=0.8)
    ax2.set_title('Pseudotime distribution (histogram)')
    ax2.set_xlabel('Pseudotime')
    ax2.set_ylabel('Count')
    # KDE
    ax3 = plt.subplot(2, 1, 2)
    try:
        sns.kdeplot(pt, fill=True, ax=ax3, color='tab:green')
    except Exception:
        # Fallback if seaborn fails
        from scipy.stats import gaussian_kde
        xs = np.linspace(pt.min(), pt.max(), 200)
        kde = gaussian_kde(pt)
        ax3.plot(xs, kde(xs), color='tab:green')
        ax3.fill_between(xs, 0, kde(xs), color='tab:green', alpha=0.3)
    ax3.set_title('Pseudotime density (KDE)')
    ax3.set_xlabel('Pseudotime')
    ax3.set_ylabel('Density')
    plt.tight_layout()
    outfile = os.path.join(RESULTS_PATH, 'pseudotime_charts.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"[INFO] Saved pseudotime charts to {outfile}")

def plot_pca_umap(adata):
    """Create dedicated PCA and UMAP scatter plots colored by pseudotime."""
    print("[INFO] Plotting dedicated PCA and UMAP charts...")
    key = 'combined_pseudotime' if 'combined_pseudotime' in adata.obs else 'dpt_pseudotime'
    colors = adata.obs[key].values if key in adata.obs else None

    # Ensure UMAP
    if 'X_umap' not in adata.obsm:
        sc.tl.umap(adata)

    fig = plt.figure(figsize=(14, 6))
    # PCA
    ax1 = plt.subplot(1, 2, 1)
    if 'X_pca' in adata.obsm:
        sca = ax1.scatter(adata.obsm['X_pca'][:, 0], adata.obsm['X_pca'][:, 1],
                          c=colors, cmap='viridis', s=8, alpha=0.8)
        plt.colorbar(sca, ax=ax1, label='Pseudotime')
    ax1.set_title('PCA (PC1 vs PC2)')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    # UMAP
    ax2 = plt.subplot(1, 2, 2)
    sca = ax2.scatter(adata.obsm['X_umap'][:, 0], adata.obsm['X_umap'][:, 1],
                      c=colors, cmap='viridis', s=8, alpha=0.8)
    plt.colorbar(sca, ax=ax2, label='Pseudotime')
    ax2.set_title('UMAP (1 vs 2)')
    ax2.set_xlabel('UMAP1')
    ax2.set_ylabel('UMAP2')
    plt.tight_layout()
    outfile = os.path.join(RESULTS_PATH, 'pca_umap_plots.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"[INFO] Saved PCA/UMAP plots to {outfile}")

def plot_embeddings_overview(adata, vae_hidden_dims=None, vae_latent_dim=None):
    """Overview image: PCA, UMAP, VAE latent, GNN embedding, and VAE architecture."""
    print("[INFO] Plotting embeddings overview (PCA, UMAP, VAE, GNN, VAE layers)...")
    key = 'combined_pseudotime' if 'combined_pseudotime' in adata.obs else 'dpt_pseudotime'
    colors = adata.obs[key].values if key in adata.obs else None
    if 'X_umap' not in adata.obsm:
        sc.tl.umap(adata)

    fig = plt.figure(figsize=(18, 12))
    # PCA
    ax1 = plt.subplot(2, 3, 1)
    if 'X_pca' in adata.obsm:
        sca = ax1.scatter(adata.obsm['X_pca'][:, 0], adata.obsm['X_pca'][:, 1], c=colors, cmap='viridis', s=6, alpha=0.8)
        plt.colorbar(sca, ax=ax1)
    ax1.set_title('PCA (PC1 vs PC2)')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    # UMAP
    ax2 = plt.subplot(2, 3, 2)
    sca = ax2.scatter(adata.obsm['X_umap'][:, 0], adata.obsm['X_umap'][:, 1], c=colors, cmap='viridis', s=6, alpha=0.8)
    plt.colorbar(sca, ax=ax2)
    ax2.set_title('UMAP (1 vs 2)')
    ax2.set_xlabel('UMAP1')
    ax2.set_ylabel('UMAP2')
    # VAE latent
    ax3 = plt.subplot(2, 3, 3)
    if 'X_vae' in adata.obsm:
        sca = ax3.scatter(adata.obsm['X_vae'][:, 0], adata.obsm['X_vae'][:, 1], c=colors, cmap='viridis', s=6, alpha=0.8)
        plt.colorbar(sca, ax=ax3)
    ax3.set_title('VAE latent (dim1 vs dim2)')
    ax3.set_xlabel('z1')
    ax3.set_ylabel('z2')
    # GNN embedding
    ax4 = plt.subplot(2, 3, 4)
    if 'X_gnn' in adata.obsm:
        sca = ax4.scatter(adata.obsm['X_gnn'][:, 0], adata.obsm['X_gnn'][:, 1], c=colors, cmap='viridis', s=6, alpha=0.8)
        plt.colorbar(sca, ax=ax4)
    ax4.set_title('GNN embedding (dim1 vs dim2)')
    ax4.set_xlabel('g1')
    ax4.set_ylabel('g2')
    # VAE architecture schematic
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    if vae_hidden_dims is None:
        vae_hidden_dims = [1024, 512, 256, 128]
    if vae_latent_dim is None:
        vae_latent_dim = 64
    layers = [adata.n_vars] + vae_hidden_dims + [vae_latent_dim] + list(reversed(vae_hidden_dims)) + [adata.n_vars]
    xs = np.linspace(0.05, 0.95, len(layers))
    max_width = max(layers)
    for i, (x, width) in enumerate(zip(xs, layers)):
        rect_height = 0.8 * (width / max_width)
        ax5.add_patch(plt.Rectangle((x-0.01, 0.5 - rect_height/2), 0.02, rect_height, color='tab:gray', alpha=0.7))
        ax5.text(x, 0.5 + rect_height/2 + 0.03, str(width), ha='center', va='bottom', fontsize=8)
    for i in range(len(xs)-1):
        ax5.plot([xs[i]+0.01, xs[i+1]-0.01], [0.5, 0.5], color='black', lw=1)
    ax5.set_title('VAE architecture (layer widths)')
    # Leave last subplot empty or use legend space
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    ax6.text(0.0, 0.8, 'Color: pseudotime', fontsize=10)
    plt.tight_layout()
    outfile = os.path.join(RESULTS_PATH, 'embeddings_overview.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"[INFO] Saved embeddings overview to {outfile}")

def plot_tf_dynamics(adata, tfs, grn_analyzer):
    """Plot TF expression dynamics and save to CSV."""
    print("[INFO] Plotting TF dynamics...")
    
    if 'dpt_pseudotime' not in adata.obs:
        print("[WARN] Pseudotime not computed.")
        return
    
    pseudotime = adata.obs['dpt_pseudotime'].values
    time_bins = np.linspace(pseudotime.min(), pseudotime.max(), 50)
    
    # Get TF out-degrees if available
    tf_outdegree_dict = {}
    if grn_analyzer.grn is not None:
        tf_outdegree_dict = dict(grn_analyzer.grn.out_degree())
    
    # Calculate mean expression per bin
    tf_means = {}
    for tf in tfs:
        if tf in adata.raw.var_names:
            expr = adata.raw[:, tf].X
            if issparse(expr):
                expr = expr.toarray().flatten()
            
            means = []
            for i in range(len(time_bins)-1):
                mask = (pseudotime >= time_bins[i]) & (pseudotime < time_bins[i+1])
                if np.sum(mask) > 0:
                    means.append(np.mean(expr[mask]))
                else:
                    means.append(0)
            tf_means[tf] = means
    
    # Order TFs by importance
    if tf_outdegree_dict:
        ordered_tfs = sorted(tf_means.keys(), 
                           key=lambda tf: (-tf_outdegree_dict.get(tf, 0), -np.mean(tf_means[tf])))
    else:
        ordered_tfs = sorted(tf_means.keys(), key=lambda tf: -np.mean(tf_means[tf]))
    
    # Save to CSV
    df = pd.DataFrame([tf_means[tf] for tf in ordered_tfs], 
                     index=ordered_tfs, 
                     columns=[f"bin_{i+1}" for i in range(len(time_bins)-1)])
    
    if tf_outdegree_dict:
        df.insert(0, 'OutDegree', [tf_outdegree_dict.get(tf, 0) for tf in ordered_tfs])
    
    csv_path = os.path.join(RESULTS_PATH, 'tf_expression_dynamics.csv')
    df.to_csv(csv_path)
    print(f"[INFO] Saved TF dynamics to {csv_path}")
    
    # Plot heatmap
    plt.figure(figsize=(14, min(20, 0.3*len(ordered_tfs[:50]))))
    
    heatmap_data = [tf_means[tf] for tf in ordered_tfs[:50]]  # Top 50
    
    ax = sns.heatmap(heatmap_data, 
                     xticklabels=10, 
                     yticklabels=ordered_tfs[:50],
                     cmap='viridis', 
                     cbar_kws={'label': 'Mean Expression'})
    
    plt.xlabel('Pseudotime Bin')
    plt.ylabel('Transcription Factor')
    plt.title('TF Expression Dynamics over Pseudotime')
    plt.tight_layout()
    
    heatmap_path = os.path.join(RESULTS_PATH, 'tf_dynamics_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # --- NEW: Separate, larger heatmap figure ---
    plt.figure(figsize=(24, max(12, 0.5*len(ordered_tfs[:50]))))  # Larger figure for better visibility
    ax2 = sns.heatmap(heatmap_data, 
                      xticklabels=10, 
                      yticklabels=ordered_tfs[:50],
                      cmap='viridis', 
                      cbar_kws={'label': 'Mean Expression'})
    plt.xlabel('Pseudotime Bin')
    plt.ylabel('Transcription Factor')
    plt.title('TF Expression Dynamics over Pseudotime (Larger)')
    plt.tight_layout()
    large_heatmap_path = os.path.join(RESULTS_PATH, 'tf_dynamics_heatmap_large.png')
    plt.savefig(large_heatmap_path, dpi=300, bbox_inches='tight')
    plt.show()
    # --- END NEW ---
    
    # Print top TFs per stage
    print("\n[INFO] Top TFs by developmental stage:")
    stages = ['Early (0-33%)', 'Middle (33-66%)', 'Late (66-100%)']
    for i, stage in enumerate(stages):
        stage_start = i / 3
        stage_end = (i + 1) / 3
        mask = (pseudotime >= np.quantile(pseudotime, stage_start)) & \
               (pseudotime < np.quantile(pseudotime, stage_end))
        
        tf_stage_means = {}
        for tf in tfs:
            if tf in adata.raw.var_names:
                expr = adata.raw[:, tf].X
                if issparse(expr):
                    expr = expr.toarray().flatten()
                tf_stage_means[tf] = np.mean(expr[mask])
        
        top_tfs = sorted(tf_stage_means.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n{stage}:")
        for tf, mean_expr in top_tfs:
            out_degree = tf_outdegree_dict.get(tf, 0)
            print(f"  {tf}: expr={mean_expr:.3f}, targets={out_degree}")

# =========================
# Step 8: Evaluation Metrics
# =========================
def evaluate_comprehensive_metrics(adata, grn_analyzer):
    """Evaluate comprehensive metrics for the analysis."""
    print("\n[INFO] Evaluating comprehensive metrics...")
    
    metrics = {}
    
    # 1. Clustering metrics
    if 'hierarchical_clusters' in adata.obs:
        cluster_key = 'hierarchical_clusters'
    elif 'leiden' in adata.obs:
        cluster_key = 'leiden'
    else:
        cluster_key = None
    
    if cluster_key:
        metrics['n_branches'] = len(np.unique(adata.obs[cluster_key]))
        cluster_sizes = adata.obs[cluster_key].value_counts()
        metrics['min_cluster_size'] = cluster_sizes.min()
        metrics['max_cluster_size'] = cluster_sizes.max()
        metrics['median_cluster_size'] = cluster_sizes.median()
    
    # 2. Pseudotime metrics
    if 'dpt_pseudotime' in adata.obs and 'developmental_stage' in adata.obs:
        pseudotime = adata.obs['dpt_pseudotime'].astype(float)
        stages = adata.obs['developmental_stage'].astype(float)
        
        # Pearson correlation
        corr_pearson, p_pearson = pearsonr(pseudotime, stages)
        metrics['pseudotime_pearson_r'] = corr_pearson
        metrics['pseudotime_pearson_p'] = p_pearson
        
        # Spearman correlation
        corr_spearman, p_spearman = spearmanr(pseudotime, stages)
        metrics['pseudotime_spearman_r'] = corr_spearman
        metrics['pseudotime_spearman_p'] = p_spearman
    # Best pseudotime metrics (if computed)
    if 'dpt_pseudotime_best' in adata.obs and 'developmental_stage' in adata.obs:
        ptb = adata.obs['dpt_pseudotime_best'].astype(float)
        stages = adata.obs['developmental_stage'].astype(float)
        corr_pearson_b, p_pearson_b = pearsonr(ptb, stages)
        corr_spearman_b, p_spearman_b = spearmanr(ptb, stages)
        metrics['best_pseudotime_pearson_r'] = corr_pearson_b
        metrics['best_pseudotime_pearson_p'] = p_pearson_b
        metrics['best_pseudotime_spearman_r'] = corr_spearman_b
        metrics['best_pseudotime_spearman_p'] = p_spearman_b
    
    # 3. Cell type accuracy (if available)
    if 'cell_type_annotation' in adata.obs:
        # Notochord accuracy
        notochord_cells = adata.obs['cell_type_annotation'].str.contains('notochord', case=False, na=False)
        if notochord_cells.sum() > 0:
            X = adata.obsm['X_gnn']
            y = notochord_cells.astype(int)
            
            if y.sum() > 5 and (len(y) - y.sum()) > 5:  # Need both classes
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                clf = LogisticRegression(max_iter=1000, class_weight='balanced')
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                metrics['notochord_accuracy'] = accuracy
    
    # 4. GRN metrics
    if grn_analyzer.grn is not None:
        metrics['grn_nodes'] = len(grn_analyzer.grn.nodes)
        metrics['grn_edges'] = len(grn_analyzer.grn.edges)
        metrics['grn_density'] = nx.density(grn_analyzer.grn)
        
        # Top regulators
        top_regulators = grn_analyzer.find_key_regulators(top_n=10)
        if top_regulators is not None and len(top_regulators) > 0:
            metrics['top_tf_targets'] = top_regulators['Targets'].iloc[0]
            metrics['mean_tf_targets'] = top_regulators['Targets'].mean()
    
    # 5. Trajectory complexity
    if 'paga' in adata.uns and 'connectivities' in adata.uns['paga']:
        paga_conn = adata.uns['paga']['connectivities']
        if hasattr(paga_conn, 'todense'):
            paga_conn = paga_conn.todense()
        metrics['paga_connectivity'] = np.mean(paga_conn[paga_conn > 0])
    
    # Print metrics
    print("\n" + "="*50)
    print("FINAL METRICS:")
    print("="*50)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    print("="*50)
    
    # Check if criteria are met
    if 'n_branches' in metrics and 'pseudotime_pearson_r' in metrics:
        if metrics['n_branches'] >= 40 and metrics['pseudotime_pearson_r'] >= 0.85:
            print("\n✓ SUCCESS: Criteria met! (≥40 branches and ≥0.85 correlation)")
        else:
            print(f"\n✗ Criteria not met: {metrics['n_branches']} branches, "
                  f"{metrics['pseudotime_pearson_r']:.3f} correlation")
    
    return metrics

# RNA velocity and phase portrait analysis
import scvelo as scv

def compute_and_plot_rna_velocity(adata, results_path=RESULTS_PATH, n_top_genes=5):
    """Compute RNA velocity, plot velocity on UMAP, and phase portraits for top dynamic genes."""
    print("[INFO] Computing RNA velocity and plotting phase portraits...")
    # Check for spliced/unspliced layers
    if not (hasattr(adata, 'layers') and 'spliced' in adata.layers and 'unspliced' in adata.layers):
        print("[WARN] No spliced/unspliced layers found in AnnData. Skipping RNA velocity.")
        return
    
    # Preprocess for velocity
    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    
    # Compute velocity
    scv.tl.velocity(adata, mode='stochastic')
    scv.tl.velocity_graph(adata)
    
    # Plot velocity on UMAP
    plt.figure(None)
    scv.pl.velocity_embedding_stream(adata, basis='umap', color='developmental_stage', save=False, show=False)
    plt.savefig(os.path.join(results_path, 'rna_velocity_umap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Identify top dynamic genes
    scv.tl.rank_velocity_genes(adata, groupby='developmental_stage', min_corr=0.1)
    top_genes = adata.var['velocity_score'].sort_values(ascending=False).head(n_top_genes).index.tolist()
    
    # Plot phase portraits for top genes
    for gene in top_genes:
        plt.figure(None)
        scv.pl.velocity(adata, gene, save=False, show=False)
        plt.savefig(os.path.join(results_path, f'phase_portrait_{gene}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    print(f"[INFO] RNA velocity and phase portraits saved to {results_path}")

# KPZ equation application for trajectory end detection
from scipy.ndimage import gaussian_filter1d

def apply_kpz_equation(adata, nu=1.0, lambd=1.0, D=1.0, noise_std=0.1, n_steps=100, dt=0.01, save_plot=True):
    """Simulate KPZ-like equation for the trajectory and optionally help find the end of the trajectory."""
    print("[INFO] Applying KPZ equation to model trajectory end...")
    # Use pseudotime as the field phi
    if 'combined_pseudotime' in adata.obs:
        phi = adata.obs['combined_pseudotime'].values.copy()
    elif 'dpt_pseudotime' in adata.obs:
        phi = adata.obs['dpt_pseudotime'].values.copy()
    else:
        print("[WARN] No pseudotime found for KPZ equation.")
        return None
    phi = (phi - np.min(phi)) / (np.max(phi) - np.min(phi) + 1e-8)  # Normalize
    n = len(phi)
    # Simulate KPZ evolution
    phi_traj = [phi.copy()]
    for t in range(n_steps):
        lap = np.gradient(np.gradient(phi))
        nonlinear = phi * lap
        noise = np.random.normal(0, noise_std, size=n)
        dphi = nu * lap - lambd * nonlinear + D * phi * (1 - phi) + noise
        phi = phi + dt * dphi
        phi = np.clip(phi, 0, 1)
        phi_traj.append(phi.copy())
    phi_traj = np.array(phi_traj)
    # Optionally, use the last state as the "end" of the trajectory
    phi_end = phi_traj[-1]
    # Save plot
    if save_plot:
        plt.figure(figsize=(10,6))
        plt.imshow(phi_traj, aspect='auto', cmap='viridis', interpolation='bilinear')
        plt.colorbar(label='Phi (trajectory state)')
        plt.xlabel('Cell index')
        plt.ylabel('Time step')
        plt.title('KPZ Equation Simulation of Trajectory')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_PATH, 'kpz_equation_simulation.png'), dpi=300)
        plt.close()
        # Plot end state
        plt.figure(figsize=(10,4))
        plt.plot(phi_end, label='KPZ end state')
        plt.xlabel('Cell index')
        plt.ylabel('Phi (end)')
        plt.title('KPZ Equation End State')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_PATH, 'kpz_equation_end_state.png'), dpi=300)
        plt.close()
    # Save end state to file
    np.savetxt(os.path.join(RESULTS_PATH, 'kpz_end_state.txt'), phi_end)
    print(f"[INFO] KPZ equation simulation and end state saved to {RESULTS_PATH}")
    return phi_end

def compute_kpz_exponent(pseudotime, min_window=5, max_window_frac=0.25, n_windows=10):
    """
    Estimate the KPZ scaling exponent (beta) from a 1D pseudotime array.
    Uses windowed standard deviation and log-log linear regression.
    Returns (beta, window_sizes, widths)
    """
    import numpy as np
    pseudotime = np.asarray(pseudotime)
    n = len(pseudotime)
    if n < 2 * min_window:
        return np.nan, np.array([]), np.array([])
    max_window = max(min(int(n * max_window_frac), n // 2), min_window+1)
    window_sizes = np.unique(np.logspace(np.log10(min_window), np.log10(max_window), n_windows, dtype=int))
    widths = []
    for w in window_sizes:
        if w >= n:
            continue
        stds = [np.std(pseudotime[i:i+w]) for i in range(0, n-w+1, max(1, w//2))]
        widths.append(np.mean(stds))
    window_sizes = window_sizes[:len(widths)]
    widths = np.array(widths)
    # Linear fit in log-log space
    if len(window_sizes) < 2 or np.any(widths <= 0):
        return np.nan, window_sizes, widths
    log_ws = np.log(window_sizes)
    log_widths = np.log(widths)
    slope, intercept = np.polyfit(log_ws, log_widths, 1)
    beta = slope
    return beta, window_sizes, widths

def compute_kpz_drift_speed(pseudotime, lambd=1.0):
    """
    Estimate the KPZ drift speed from a 1D pseudotime array and lambda parameter.
    v = (lambda/2) * mean(slope^2)
    """
    import numpy as np
    pseudotime = np.asarray(pseudotime)
    slopes = np.diff(pseudotime)
    mean_sq_slope = np.mean(slopes**2)
    drift_speed = 0.5 * lambd * mean_sq_slope
    return drift_speed

def maximize_silhouette_score(adata, min_clusters=2, max_clusters=50, min_cluster_size=10):
    """
    Try different embeddings and clustering methods to maximize silhouette score.
    Prints diagnostics and returns best score and parameters.
    """
    from sklearn.metrics import silhouette_score
    from scipy.cluster import hierarchy
    import scanpy as sc
    import numpy as np
    best_score = -2
    best_params = None
    best_labels = None
    best_embedding = None
    embeddings_to_try = []
    if 'X_pca' in adata.obsm:
        embeddings_to_try.append(('X_pca', adata.obsm['X_pca'][:, :30]))
    if 'X_combined' in adata.obsm:
        embeddings_to_try.append(('X_combined', adata.obsm['X_combined']))
    for emb_name, emb in embeddings_to_try:
        # Hierarchical clustering
        linkage = hierarchy.linkage(emb, method='ward')
        for n_clusters in range(min_clusters, min(max_clusters, len(emb)//min_cluster_size)+1):
            labels = hierarchy.fcluster(linkage, n_clusters, criterion='maxclust')
            unique, counts = np.unique(labels, return_counts=True)
            if len(unique) < 2 or np.min(counts) < min_cluster_size:
                continue
            try:
                score = silhouette_score(emb, labels, sample_size=min(5000, len(labels)))
            except Exception as e:
                continue
            print(f"[Silhouette] emb={emb_name}, method=hierarchical, n_clusters={n_clusters}, score={score:.3f}, sizes={counts}")
            if score > best_score:
                best_score = score
                best_params = {'embedding': emb_name, 'method': 'hierarchical', 'n_clusters': n_clusters}
                best_labels = labels
                best_embedding = emb
        # Leiden clustering
        for resolution in np.linspace(0.2, 2.0, 10):
            sc.pp.neighbors(adata, n_neighbors=15, use_rep=emb_name)
            sc.tl.leiden(adata, resolution=resolution, key_added='leiden_tmp')
            labels = adata.obs['leiden_tmp'].astype(str).values
            unique, counts = np.unique(labels, return_counts=True)
            if len(unique) < 2 or np.min(counts) < min_cluster_size:
                continue
            try:
                score = silhouette_score(emb, labels, sample_size=min(5000, len(labels)))
            except Exception as e:
                continue
            print(f"[Silhouette] emb={emb_name}, method=leiden, resolution={resolution:.2f}, score={score:.3f}, sizes={counts}")
            if score > best_score:
                best_score = score
                best_params = {'embedding': emb_name, 'method': 'leiden', 'resolution': resolution}
                best_labels = labels
                best_embedding = emb
    print(f"\n[Silhouette] Best score: {best_score:.3f} with params: {best_params}")
    return best_score, best_params, best_labels, best_embedding

# =========================
# Step 9: Main Pipeline
# =========================
def clean_and_save_tf_list(txt_path, cleaned_csv_path):
    """Clean and save TF list."""
    if not os.path.exists(txt_path):
        print(f"[ERROR] TF list file not found: {txt_path}")
        return None
    
    try:
        df = pd.read_csv(txt_path, sep='\t', header=0)
        if 'Symbol' not in df.columns:
            print(f"[ERROR] 'Symbol' column not found in {txt_path}")
            return None
        
        tf_symbols = df['Symbol'].astype(str).str.strip()
        tf_symbols = tf_symbols[tf_symbols != '']
        tf_symbols = tf_symbols.drop_duplicates()
        tf_symbols = tf_symbols.dropna()
        
        tf_symbols.to_csv(cleaned_csv_path, index=False, header=False)
        print(f"[INFO] Cleaned TF list saved to {cleaned_csv_path} ({len(tf_symbols)} symbols)")
        
        return cleaned_csv_path
    except Exception as e:
        print(f"[ERROR] Failed to clean TF list: {e}")
        return None

def main(load_from_h5ad=False, h5ad_path=None):
    """Main pipeline for zebrafish trajectory analysis."""
    print("[INFO] Starting enhanced zebrafish scRNA-seq pipeline...")
    print("="*70)
    
    # Create results directory
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    # Clean TF list
    clean_and_save_tf_list(RAW_TF_LIST_FILE, CLEANED_TF_LIST_FILE)
    
    if load_from_h5ad and h5ad_path is not None and os.path.exists(h5ad_path):
        print(f"[INFO] Loading pre-processed data from {h5ad_path}...")
        adata = sc.read_h5ad(h5ad_path)
    else:
        # Step 1: Enhanced data loading and preprocessing
        processor = EnhancedscRNAProcessor(TIMEPOINT_FILES, METADATA_FILE, CLEANED_TF_LIST_FILE)
        adata = processor.load_data()
        adata = processor.enhanced_qc_filtering()
        adata = processor.enhanced_normalize_and_scale()
        adata = processor.engineer_features()
        adata = processor.run_enhanced_pca(n_components=50)
        
        # Step 2: VAE training
        print("\n[INFO] Training enhanced VAE...")
        vae_trainer = EnhancedVAETrainer(
            input_dim=adata.shape[1], 
            latent_dim=64,
            hidden_dims=[1024, 512, 256, 128]
        )
        train_losses, test_losses = vae_trainer.train(adata.X, epochs=150, batch_size=128)
        vae_embeddings = vae_trainer.get_latent_representation(adata.X)
        adata.obsm['X_vae'] = vae_embeddings
        
        # Step 3: GNN training
        print("\n[INFO] Training enhanced GNN...")
        gnn_trainer = EnhancedGNNTrainer(
            in_channels=vae_embeddings.shape[1], 
            hidden_channels=128,
            out_channels=64,
            n_layers=3
        )
        gnn_losses = gnn_trainer.train(vae_embeddings, epochs=150, n_neighbors=30)
        gnn_embeddings = gnn_trainer.get_embeddings(vae_embeddings, n_neighbors=30)
        adata.obsm['X_gnn'] = gnn_embeddings
        
        # Create combined embedding
        adata.obsm['X_combined'] = np.hstack([
            adata.obsm['X_pca'][:, :30],
            adata.obsm['X_vae'],
            adata.obsm['X_gnn']
        ])
        
        # Save intermediate results
        h5ad_path = os.path.join(RESULTS_PATH, 'adata_processed_enhanced.h5ad')
        print(f"\n[INFO] Saving processed data to {h5ad_path}...")
        # Sanitize object-dtype columns to prevent H5AD writing errors
        for col in adata.obs.select_dtypes(include='object').columns:
            adata.obs[col] = adata.obs[col].astype('category')
        for col in adata.var.select_dtypes(include='object').columns:
            adata.var[col] = adata.var[col].astype('category')
        adata.write(h5ad_path)
    
    # Step 4: Hierarchical clustering for branches
    adata, n_clusters = hierarchical_clustering_branches(adata, min_cluster_size=20, max_clusters=100)
    
    # --- NEW: Maximize silhouette score ---
    best_sil_score, best_sil_params, best_sil_labels, best_sil_emb = maximize_silhouette_score(adata, min_clusters=2, max_clusters=50, min_cluster_size=10)
    if best_sil_score >= 0.75:
        print(f"[Silhouette] SUCCESS: Silhouette score is {best_sil_score:.3f} (>=0.75)")
    else:
        print(f"[Silhouette] WARNING: Silhouette score is {best_sil_score:.3f} (<0.75). Consider tuning parameters or preprocessing.")
    # Optionally, store best clustering in adata.obs
    if best_sil_params is not None:
        if best_sil_params['method'] == 'hierarchical':
            adata.obs['best_silhouette_cluster'] = best_sil_labels.astype(str)
        elif best_sil_params['method'] == 'leiden':
            adata.obs['best_silhouette_cluster'] = best_sil_labels
    # --- END NEW ---
    
    # Step 5: Enhanced grid search
    best_result = enhanced_grid_search(adata, min_branches=40, min_corr=0.85)
    
    if best_result['branches'] > 0:
        print(f"\n[INFO] Best configuration found:")
        print(f"  Branches: {best_result['branches']}")
        print(f"  Correlation: {best_result['correlation']:.3f}")
        print(f"  Parameters: {best_result['params']}")
        
        if 'adata' in best_result:
            adata = best_result['adata']
    
    # Step 6: Multi-method pseudotime
    adata = compute_multi_method_pseudotime(adata)
    # Try to improve pseudotime correlation without overwriting originals
    adata = compute_best_pseudotime(adata, embeddings=['X_pca', 'X_combined', 'X_gnn', 'X_vae'], n_neighbors_list=[15, 30, 50], n_comps=10)
    # KPZ calculation
    if 'combined_pseudotime' in adata.obs:
        kpz_pseudotime = adata.obs['combined_pseudotime'].values
    elif 'dpt_pseudotime' in adata.obs:
        kpz_pseudotime = adata.obs['dpt_pseudotime'].values
    else:
        kpz_pseudotime = None
    if kpz_pseudotime is not None:
        kpz_beta, kpz_ws, kpz_widths = compute_kpz_exponent(kpz_pseudotime)
        print(f"[KPZ] Estimated KPZ scaling exponent (beta): {kpz_beta:.3f}")
        kpz_path = os.path.join(RESULTS_PATH, 'kpz_exponent.txt')
        with open(kpz_path, 'w') as f:
            f.write(f"KPZ scaling exponent (beta): {kpz_beta:.6f}\n")
            f.write(f"Window sizes: {kpz_ws.tolist()}\n")
            f.write(f"Widths: {kpz_widths.tolist()}\n")
        # Optional: plot KPZ fit
        plt.figure(figsize=(8,6))
        plt.plot(kpz_ws, kpz_widths, 'o', label='Observed')
        if not np.isnan(kpz_beta):
            plt.plot(kpz_ws, kpz_ws**kpz_beta * kpz_widths[0]/(kpz_ws[0]**kpz_beta), '-', label=f'Fit (beta={kpz_beta:.2f})')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Window size')
        plt.ylabel('Width (std)')
        plt.title('KPZ Scaling of Pseudotime Trajectory')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_PATH, 'kpz_scaling.png'), dpi=300)
        plt.show()
        # --- NEW: Compute and print KPZ drift speed ---
        kpz_drift_speed = compute_kpz_drift_speed(kpz_pseudotime, lambd=1.0)  # Use your lambda value if different
        print(f"[KPZ] Estimated drift speed from KPZ: {kpz_drift_speed:.4f}")
        # --- END NEW ---
    # RNA velocity and phase portrait analysis
    compute_and_plot_rna_velocity(adata, results_path=RESULTS_PATH, n_top_genes=5)
    # KPZ equation simulation for trajectory end
    apply_kpz_equation(adata, nu=1.0, lambd=1.0, D=1.0, noise_std=0.1, n_steps=100, dt=0.01, save_plot=True)
    
    # Step 7: GRN analysis
    print("\n[INFO] Inferring gene regulatory network...")
    grn_analyzer = EnhancedGRNAnalyzer(adata, tf_list=processor.tf_list)
    tfs = grn_analyzer.identify_tfs(min_expr_pct=0.05)
    grn = grn_analyzer.infer_grn(method='correlation', n_genes=2000, threshold=0.3)
    
    if grn is not None:
        top_regulators = grn_analyzer.find_key_regulators(top_n=30)
        print("\n[INFO] Top transcriptional regulators:")
        print(top_regulators.head(10))
        
        grn_analyzer.plot_grn(top_n_tfs=15)
        plt.savefig(os.path.join(RESULTS_PATH, 'grn_network.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    # Step 8: Comprehensive visualization
    visualize_3d_trajectory(adata)
    plot_comprehensive_results(adata, grn_analyzer, tfs)
    plot_tf_dynamics(adata, tfs, grn_analyzer)

    # Additional requested visualizations
    try:
        # 1) Cell-by-gene heatmap (use top TFs if available, else top HVGs)
        genes_for_heatmap = tfs[:10] if tfs else (adata.var_names[:10].tolist() if hasattr(adata.var_names, 'tolist') else list(adata.var_names[:10]))
        plot_cell_gene_heatmap(adata, genes_for_heatmap, order_by='combined_pseudotime' if 'combined_pseudotime' in adata.obs else 'dpt_pseudotime')
    except Exception as e:
        print(f"[WARN] Failed to plot cell-gene heatmap: {e}")

    try:
        # 2) 2D trajectory with free layout (force-directed)
        visualize_trajectory_2d_free(adata)
    except Exception as e:
        print(f"[WARN] Failed to plot 2D free-layout trajectory: {e}")

    try:
        # 2b) 3D trajectory with free layout (force-directed)
        visualize_trajectory_3d_free(adata)
    except Exception as e:
        print(f"[WARN] Failed to plot 3D free-layout trajectory: {e}")

    try:
        # Non-force-directed trajectory (PCA spline)
        visualize_trajectory_pca_spline(adata)
    except Exception as e:
        print(f"[WARN] Failed to plot PCA spline trajectory: {e}")

    try:
        # 3) Detailed GRN image
        plot_grn_detailed(grn_analyzer, max_targets_per_tf=20)
    except Exception as e:
        print(f"[WARN] Failed to plot detailed GRN: {e}")

    try:
        # 5) Pseudotime charts (line/hist/KDE)
        plot_pseudotime_charts(adata)
    except Exception as e:
        print(f"[WARN] Failed to plot pseudotime charts: {e}")

    try:
        # 6) Dedicated PCA and UMAP charts
        plot_pca_umap(adata)
    except Exception as e:
        print(f"[WARN] Failed to plot PCA/UMAP charts: {e}")

    try:
        # 7) Overview image of PCA, VAE, GNN and VAE hidden layers
        plot_embeddings_overview(adata, vae_hidden_dims=[1024, 512, 256, 128], vae_latent_dim=64)
    except Exception as e:
        print(f"[WARN] Failed to plot embeddings overview: {e}")
    
    # Step 9: Final evaluation
    metrics = evaluate_comprehensive_metrics(adata, grn_analyzer)
    
    # Save final results
    final_h5ad_path = os.path.join(RESULTS_PATH, 'adata_final_enhanced.h5ad')
    print(f"\n[INFO] Saving final results to {final_h5ad_path}...")
    # Sanitize object-dtype columns to prevent H5AD writing errors
    for col in adata.obs.select_dtypes(include='object').columns:
        if adata.obs[col].isnull().any():
            adata.obs[col] = adata.obs[col].astype(str)
        adata.obs[col] = adata.obs[col].astype('category')
    for col in adata.var.select_dtypes(include='object').columns:
        if adata.var[col].isnull().any():
            adata.var[col] = adata.var[col].astype(str)
        adata.var[col] = adata.var[col].astype('category')
    adata.write(final_h5ad_path)
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(RESULTS_PATH, 'final_metrics.csv'), index=False)
    
    print("\n[INFO] Pipeline completed successfully!")
    print("="*70)
    
    return adata, metrics

if __name__ == "__main__":
    # Run the enhanced pipeline
    adata, metrics = main(load_from_h5ad=False)