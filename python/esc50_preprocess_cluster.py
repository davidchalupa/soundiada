#!/usr/bin/env python3
"""
ESC-50 preprocessing + embedding + clustering prototype (single-file script).

Usage examples:
  # default (chunks summarizer, 5 chunks)
  python esc50_preprocess_cluster.py

  # sample 5 categories, 10 samples each, chunked features
  python esc50_preprocess_cluster.py --n-categories 5 --samples-per-category 10 --feat-mode chunks --n-chunks 5

  # use original mean+std summarizer
  python esc50_preprocess_cluster.py --feat-mode meanstd
"""
import os
import sys
import argparse
import zipfile
import requests
import io
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
import umap
import hdbscan
import plotly.express as px
import webbrowser
import shutil
import random

# ---------------------------
# Configurable defaults
# ---------------------------
DATA_DIR = "data/ESC-50-master"   # expected repo layout: data/ESC-50-master/audio + meta/esc50.csv
WAV_SUBDIR = "audio"             # inside repo
META_CSV = os.path.join(DATA_DIR, "meta", "esc50.csv")
OUT_DIR = "out"
FEAT_DIR = os.path.join(OUT_DIR, "feats")
EMB_PATH = os.path.join(OUT_DIR, "embeddings.npy")
META_OUT = os.path.join(OUT_DIR, "meta_with_feats.csv")
UMAP2_HTML = os.path.join(OUT_DIR, "umap_2d.html")
UMAP3_HTML = os.path.join(OUT_DIR, "umap_3d.html")

# audio / feature params
SR_TARGET = 22050
TARGET_SECONDS = 5.0
N_MELS = 64
N_FFT = 2048
HOP_LENGTH = 512

# embedding / dim-reduction params
PCA_DIM = 64
# UMAP2_PARAMS = dict(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
UMAP3_PARAMS = dict(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)

def make_umap_params(n_samples):
    """
    Adaptive UMAP params.
    :param n_samples:
    :return:
    """
    # Keep UMAP local when dataset is small
    if n_samples <= 100:
        n_neighbors = max(3, min(10, n_samples // 5))
    else:
        n_neighbors = 15

    print(f"UMAP using n_neighbors={n_neighbors} for n_samples={n_samples}")

    return dict(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        random_state=42
    )

# clustering / outlier params
HDB_MIN_CLUSTER_SIZE = 6
HDB_MIN_SAMPLES = 2
KMEANS_K = 50   # default KMeans clusters used for coloring (overridden when sampling)

# PANNs weights local path (if you downloaded manually)
CNN10_LOCAL_PATH = os.path.join("models", "Cnn10_mAP=0.380.pth")

# ---------------------------
# Helper functions
# ---------------------------
def download_and_extract_esc50(dest_dir="data"):
    print("Downloading ESC-50 (GitHub zip)...")
    url = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        print("Download failed:", e)
        print("Please download ESC-50 manually from https://github.com/karolpiczak/ESC-50 and place it under data/")
        sys.exit(1)

    z = zipfile.ZipFile(io.BytesIO(resp.content))
    print("Extracting...")
    z.extractall(dest_dir)
    print("Extracted to", dest_dir)
    return True

def list_wavs_from_meta(meta_csv):
    df = pd.read_csv(meta_csv)
    # meta CSV has 'filename' relative to audio folder and 'category' etc.
    df['file_path'] = df['filename'].apply(lambda fn: os.path.join(DATA_DIR, WAV_SUBDIR, fn))
    return df

def ensure_outdirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FEAT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CNN10_LOCAL_PATH), exist_ok=True)

def load_audio(path, sr=SR_TARGET, target_seconds=TARGET_SECONDS):
    y, sr0 = sf.read(path, dtype='float32')
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr0 != sr:
        # use keyword args to be robust to librosa API changes
        y = librosa.resample(y=y, orig_sr=sr0, target_sr=sr)
        sr = sr
    # pad/trim
    target_len = int(sr * target_seconds)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y, sr

def compute_logmel(y, sr=SR_TARGET, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    return logmel  # shape: (n_mels, time_frames)

def summarize_logmel_to_embedding(logmel, mode="chunks", n_chunks=5):
    """
    Summarize a log-mel spectrogram to a fixed-size embedding.

    Modes:
      - "meanstd": original mean+std across time -> dim 2*n_mels
      - "chunks": split into `n_chunks` consecutive time windows, compute mean per mel for each chunk,
                  then append global std per mel.
                  final dim = n_mels * (n_chunks + 1)
    """
    n_mels, n_frames = logmel.shape

    if mode == "meanstd":
        mean = np.mean(logmel, axis=1)
        std = np.std(logmel, axis=1)
        emb = np.concatenate([mean, std])
        return emb.astype(np.float32)

    if mode == "chunks":
        if n_chunks <= 0:
            raise ValueError("n_chunks must be >= 1")
        # chunking: ceil-based to cover all frames
        chunk_size = int(np.ceil(n_frames / float(n_chunks)))
        chunk_means = []
        for i in range(n_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, n_frames)
            if end <= start:
                seg_mean = np.zeros((n_mels,), dtype=np.float32)
            else:
                seg = logmel[:, start:end]
                seg_mean = np.mean(seg, axis=1)
            chunk_means.append(seg_mean)
        chunk_means_flat = np.concatenate(chunk_means, axis=0)  # n_mels * n_chunks
        global_std = np.std(logmel, axis=1)                    # n_mels
        emb = np.concatenate([chunk_means_flat, global_std])
        return emb.astype(np.float32)

    raise ValueError(f"Unknown summarize mode: {mode}")

def open_in_browser(paths, delay_between=0.5):
    for p in paths:
        try:
            pth = Path(p).resolve()
            if not pth.exists():
                print(f"[open_in_browser] Warning: {pth} does not exist, skipping.")
                continue
            uri = pth.as_uri()   # file://...
            webbrowser.open_new_tab(uri)
            time.sleep(delay_between)
        except Exception as e:
            print(f"[open_in_browser] Failed to open {p}: {e}")

# ---------------------------
# PANNs CNN10 loader (local checkpoint)
# ---------------------------
def try_load_panns_cnn10():
    """
    Load CNN10 checkpoint from local file. If the checkpoint is missing or torch isn't available, return None.
    The script intentionally does NOT attempt scripted download from Zenodo (that often gets rate-limited).
    """
    try:
        import torch
        import torch.nn as nn
    except Exception as e:
        print("PyTorch not available in this interpreter:", e)
        return None

    if not os.path.isfile(CNN10_LOCAL_PATH):
        print(f"CNN10 checkpoint not found at {CNN10_LOCAL_PATH}. Please download manually and place it there.")
        return None

    try:
        class SimpleCnn10(nn.Module):
            def __init__(self, n_mels=64, emb_dim=512):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(1, 64, kernel_size=(3,3), padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d((2,2)),
                    nn.Conv2d(64, 128, kernel_size=(3,3), padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d((2,2)),
                    nn.Conv2d(128, 256, kernel_size=(3,3), padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1,1)),
                )
                self.fc = nn.Linear(256, emb_dim)
            def forward(self, x):
                h = self.net(x)
                h = h.view(h.size(0), -1)
                emb = self.fc(h)
                return emb

        model = SimpleCnn10(n_mels=N_MELS, emb_dim=512)
        model.eval()

        import torch
        ckpt = torch.load(CNN10_LOCAL_PATH, map_location="cpu")

        state = None
        if isinstance(ckpt, dict):
            for key in ("model", "state_dict", "model_state_dict", "state"):
                if key in ckpt:
                    state = ckpt[key]
                    break
            if state is None:
                state = ckpt
        else:
            state = ckpt

        try:
            model.load_state_dict(state, strict=False)
            print("Loaded CNN10 weights into local backbone (strict=False).")
        except Exception as e:
            print("Warning when loading state_dict (strict=False):", e)

        return model

    except Exception as e:
        print("CNN10 local loader failed:", e)
        return None

def extract_embedding_from_panns_model_local(model, y, sr):
    try:
        import torch
    except Exception:
        raise RuntimeError("PyTorch is required to extract embeddings from the local PANNs model.")

    logmel = compute_logmel(y, sr=sr, n_mels=N_MELS)
    x = torch.tensor(logmel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,n_mels,frames)
    x = x.to("cpu")
    with torch.no_grad():
        out = model(x)
    if isinstance(out, (list, tuple)):
        out = out[0]
    if hasattr(out, "detach"):
        emb = out.detach().cpu().squeeze()
        if emb.ndim > 1:
            emb = emb.mean(axis=0)
        emb_np = emb.numpy().astype(np.float32)
    else:
        emb_np = np.array(out).squeeze().astype(np.float32)
    return emb_np

# ---------------------------
# Main pipeline
# ---------------------------
def main(args):
    if args.download:
        download_and_extract_esc50(dest_dir="data")

    if not os.path.exists(META_CSV):
        print(f"Cannot find ESC-50 metadata CSV at {META_CSV}")
        print("If you downloaded ESC-50 manually, please ensure the repo is extracted under data/ESC-50-master/")
        sys.exit(1)

    ensure_outdirs()

    meta = list_wavs_from_meta(META_CSV)
    print(f"Found {len(meta)} entries in ESC-50 meta.")

    # --- Sampling: optionally subsample by categories & samples per category ---
    if args.n_categories is not None:
        seed = int(args.seed or 42)
        rng = random.Random(seed)
        unique_cats = sorted(meta['category'].unique().tolist())
        total_cats = len(unique_cats)
        if args.n_categories > total_cats:
            print(f"Requested n_categories={args.n_categories} but only {total_cats} available. Using all categories.")
            chosen = unique_cats
        else:
            chosen = rng.sample(unique_cats, int(args.n_categories))
        print(f"Selected categories ({len(chosen)}): {chosen}")

        # collect samples_per_category using independent per-category random_state
        rows = []
        per_cat_preview = {}
        for i, c in enumerate(chosen):
            rows_c = meta[meta['category'] == c]
            if args.samples_per_category is not None:
                k = int(args.samples_per_category)
                if k < len(rows_c):
                    rs = seed + i
                    sampled = rows_c.sample(n=k, random_state=rs)
                else:
                    sampled = rows_c
            else:
                sampled = rows_c
            rows.append(sampled)
            per_cat_preview[c] = sampled['filename'].tolist()[:5]
        meta = pd.concat(rows).reset_index(drop=True)
        print("Per-category sample counts:\n", meta['category'].value_counts().to_dict())
        print("Per-category sample preview (first up to 5 filenames each):")
        for c, preview in per_cat_preview.items():
            print(f"  - {c}: {preview}")

    # whether to use local PANNs CNN10 (default off unless requested)
    use_panns = args.cnn10
    pann_model = None
    if use_panns:
        pann_model = try_load_panns_cnn10()
        if pann_model is None:
            print("PANNs CNN10 local loader failed; falling back to mean+std.")
            use_panns = False
        else:
            print("Using local PANNs CNN10 for embeddings.")

    embeddings = []
    rows_out = []

    # create emb_tag including features to avoid cache collisions
    if use_panns:
        emb_tag = "cnn10"
    else:
        emb_tag = f"{args.feat_mode}"
        if args.feat_mode == "chunks":
            emb_tag += f".{int(args.n_chunks)}"

    # iterate and compute features / embeddings (cache per-file .npy)
    for _, r in tqdm(meta.iterrows(), total=len(meta), desc="Processing files"):
        wavp = r['file_path']
        base = os.path.basename(wavp)
        feat_name_tagged = base.replace(".wav", f".{emb_tag}.npy")
        feat_path_tagged = os.path.join(FEAT_DIR, feat_name_tagged)
        feat_path_legacy = os.path.join(FEAT_DIR, base.replace(".wav", ".npy"))

        try:
            if os.path.exists(feat_path_tagged):
                emb = np.load(feat_path_tagged)
            elif (emb_tag.startswith("meanstd") or emb_tag.startswith("chunks")) and os.path.exists(feat_path_legacy):
                # legacy support: reuse old untagged cache only for meanstd/chunks old runs
                emb = np.load(feat_path_legacy)
                np.save(feat_path_tagged, emb)
            else:
                if use_panns and (pann_model is not None):
                    try:
                        y, sr = load_audio(wavp)
                        emb = extract_embedding_from_panns_model_local(pann_model, y, sr)
                        np.save(feat_path_tagged, emb)
                    except Exception as e:
                        print(f"PANNs local extraction failed for {wavp}: {e}. Falling back to summarizer.")
                        y, sr = load_audio(wavp)
                        logmel = compute_logmel(y, sr=sr)
                        emb = summarize_logmel_to_embedding(logmel, mode=args.feat_mode, n_chunks=int(args.n_chunks))
                        np.save(feat_path_tagged, emb)
                else:
                    y, sr = load_audio(wavp)
                    logmel = compute_logmel(y, sr=sr)
                    emb = summarize_logmel_to_embedding(logmel, mode=args.feat_mode, n_chunks=int(args.n_chunks))
                    np.save(feat_path_tagged, emb)
        except Exception as e:
            print("Error with", wavp, e)
            continue

        embeddings.append(emb)
        row = {
            'file_path': wavp,
            'category': r.get('category', ''),
            'fold': r.get('fold', ''),
            'base': base
        }
        rows_out.append(row)


    if len(embeddings) == 0:
        print("No embeddings were computed. Exiting.")
        sys.exit(1)

    X = np.stack(embeddings, axis=0)
    meta_out = pd.DataFrame(rows_out)
    print("Computed embeddings shape:", X.shape)

    # Decide whether to use StandardScaler + PCA based on dataset size.
    # For very small datasets PCA (and the standard scaler fitted on few samples)
    # can destroy class-separating directions; in that case feed raw embeddings to UMAP.
    PCA_SKIP_THRESHOLD = 200
    if X.shape[0] < PCA_SKIP_THRESHOLD:
        print(
            f"Small dataset detected (n_samples={X.shape[0]}) — skipping StandardScaler + PCA and using raw embeddings for UMAP.")
        Xp = X.astype(np.float32)
        scaler = None
        pca = None
    else:
        # standardize -> PCA -> embeddings (safe cap on PCA dim)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        pca_dim_eff = min(PCA_DIM, Xs.shape[0], Xs.shape[1])
        print(f"Running PCA -> reduce to {pca_dim_eff}")
        pca = PCA(n_components=pca_dim_eff, random_state=42)
        Xp = pca.fit_transform(Xs)
        try:
            print("PCA explained var (first 5):", pca.explained_variance_ratio_[:5])
        except Exception:
            pass
    # save PCA embeddings (canonical embedding used for clustering & indexing)
    np.save(EMB_PATH, Xp)
    print("Saved embeddings to", EMB_PATH)

    # UMAP 2D + 3D for visualization
    print("Computing UMAP 2D...")
    reducer2 = umap.UMAP(**make_umap_params(Xp.shape[0]))
    XY2 = reducer2.fit_transform(Xp)
    meta_out['umap2_x'] = XY2[:,0]
    meta_out['umap2_y'] = XY2[:,1]

    # print("Computing UMAP 3D...")
    # rreducer3 = umap.UMAP(metric="cosine", **UMAP3_PARAMS)
    # XY3 = reducer3.fit_transform(Xp)
    # meta_out['umap3_x'] = XY3[:,0]
    # meta_out['umap3_y'] = XY3[:,1]
    # meta_out['umap3_z'] = XY3[:,2]

    # Clustering: HDBSCAN (kept for analysis / tooltip)
    print("Clustering with HDBSCAN (for tooltip / analysis)...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=HDB_MIN_CLUSTER_SIZE, min_samples=HDB_MIN_SAMPLES, metric='euclidean')
    hdb_labels = clusterer.fit_predict(Xp)
    meta_out['hdbscan_label'] = hdb_labels  # -1 = noise

    labels = meta_out['hdbscan_label']
    n_clusters = len(set(labels)) - (1 if -1 in labels.values else 0)
    n_noise = (labels == -1).sum()

    print("HDBSCAN found", n_clusters, "clusters")
    print("Number of noise points:", n_noise)
    print("Unique HDBSCAN labels:", sorted(labels.unique()))

    # determine K for KMeans coloring: if user sampled categories, use that number; otherwise global
    if args.n_categories is not None:
        k_for_plot = int(args.n_categories)
    else:
        k_for_plot = KMEANS_K

    print(f"Clustering with KMeans K={k_for_plot} (used for plot coloring)...")
    km = KMeans(n_clusters=k_for_plot, random_state=42)
    km_labels = km.fit_predict(Xp)
    meta_out['kmeans_label'] = km_labels

    # Outlier detection using LOF (Local Outlier Factor)
    print("Computing LOF outlier scores...")
    lof = LocalOutlierFactor(n_neighbors=20, contamination='auto')
    lof_pred = lof.fit_predict(Xp)  # -1 = outlier
    lof_scores = -lof.negative_outlier_factor_
    meta_out['lof_outlier'] = (lof_pred == -1)
    meta_out['lof_score'] = lof_scores

    # Save metadata CSV
    meta_out.to_csv(META_OUT, index=False)
    print("Saved meta + cluster labels to", META_OUT)


    # --- BEGIN: export simple JS module for React (filename, category, umap2 coords) ---
    import json
    JS_OUT = os.path.join(OUT_DIR, "embeddings_for_react.js")

    # Build a small list of dicts we want to export; round floats for compactness
    js_rows = []
    for _, r in meta_out.iterrows():
        # ensure required fields exist
        if 'base' in r and 'category' in r and 'umap2_x' in r and 'umap2_y' in r:
            js_rows.append({
                "file": r['base'],
                "category": r['category'],
                # cast and round to 6 decimals to keep file small & readable
                "x": round(float(r['umap2_x']), 6),
                "y": round(float(r['umap2_y']), 6)
            })

    # Dump as compact JSON and wrap as an ES module export
    try:
        with open(JS_OUT, "w", encoding="utf-8") as jf:
            jf.write("export const esc50Embeddings = ")
            # use separators to minimize whitespace
            jf.write(json.dumps(js_rows, separators=(",", ":"), ensure_ascii=False))
            jf.write(";\n\nexport default esc50Embeddings;\n")
        print("Saved JS embeddings for React to", JS_OUT)
    except Exception as e:
        print("Failed to write JS embeddings file:", e)
    # --- END: export simple JS module ---

    # ensure audio folder in out/ with accessible relative paths (kept for compatibility)
    AUDIO_OUT_DIR = os.path.join(OUT_DIR, "audio")
    os.makedirs(AUDIO_OUT_DIR, exist_ok=True)

    # build raw-GitHub audio URLs so HTML can be uploaded and still fetch audio
    meta_out['audio_rel_path'] = meta_out['base'].apply(
        lambda fn: f"https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/{fn}"
    )

    # --- Fixed 2D plotting block (use custom_data at creation time) ---
    # small jitter to reduce exact overlap
    jitter = 1e-3 * max(1.0, (meta_out['umap2_x'].max() - meta_out['umap2_x'].min()))
    rng = np.random.RandomState(42)
    meta_out['umap2_x_jit'] = meta_out['umap2_x'] + rng.normal(scale=jitter, size=len(meta_out))
    meta_out['umap2_y_jit'] = meta_out['umap2_y'] + rng.normal(scale=jitter, size=len(meta_out))

    # columns to attach as customdata (order matters)
    custom_cols = ['audio_rel_path', 'base', 'category', 'kmeans_label', 'hdbscan_label', 'lof_score']

    # Create the Plotly scatter with custom_data passed in (ensures row-aligned customdata)
    fig2 = px.scatter(
        meta_out,
        x='umap2_x_jit',
        y='umap2_y_jit',
        color=meta_out['kmeans_label'].astype(str),
        color_discrete_sequence=px.colors.qualitative.Safe,
        hover_data=['file_path', 'category', 'kmeans_label', 'hdbscan_label', 'lof_score'],
        custom_data=custom_cols,
        title="ESC-50 UMAP 2D colored by KMeans cluster"
    )

    # marker styling and hovertemplate referencing customdata indices
    fig2.update_traces(marker=dict(size=6, opacity=0.8))
    fig2.update_traces(
        hovertemplate=
        "<b>%{customdata[1]}</b><br>"
        "Category: %{customdata[2]}<br>"
        "KMeans cluster: %{customdata[3]}<br>"
        "HDBSCAN label: %{customdata[4]}<br>"
        "LOF score: %{customdata[5]:.2f}<br>"
        "<extra></extra>"
    )

    # produce HTML string (include plotly.js via CDN for portability)
    html_str = fig2.to_html(full_html=True, include_plotlyjs='cdn')

    # Append a small JS + <audio> player to the HTML so clicks play the corresponding audio.
    player_js = r"""
    <script>
    var audioPlayer = document.createElement('audio');
    audioPlayer.id = 'audioPlayer';
    audioPlayer.controls = true;
    audioPlayer.style.position = 'fixed';
    audioPlayer.style.bottom = '10px';
    audioPlayer.style.left = '10px';
    audioPlayer.style.zIndex = 9999;
    audioPlayer.style.width = '320px';
    document.body.appendChild(audioPlayer);

    window.addEventListener('load', function() {
        var plots = document.getElementsByClassName('plotly-graph-div');
        if (plots.length === 0) {
            console.warn('No plotly graph found on page.');
            return;
        }
        var gd = plots[0];

        gd.on('plotly_click', function(eventData) {
            try {
                var pt = eventData.points[0];
                var audioPath = pt.customdata[0];
                if (!audioPath) {
                    console.warn('No audio path for clicked point.');
                    return;
                }
                var ap = document.getElementById('audioPlayer');
                ap.src = audioPath;
                ap.play().catch(function(e){ console.warn('Audio play failed:', e); });
            } catch (err) {
                console.error('Error handling plotly_click:', err);
            }
        });
    });
    </script>
    """

    final_html = html_str.replace("</body>", player_js + "\n</body>")
    with open(UMAP2_HTML, "w", encoding="utf-8") as f:
        f.write(final_html)

    print("Saved interactive 2D plot with click-to-play to", UMAP2_HTML)

    # # optional 3D scatter
    # print("Building interactive 3D scatter (Plotly)...")
    # fig3 = px.scatter_3d(meta_out, x='umap3_x', y='umap3_y', z='umap3_z',
    #                     color=meta_out['kmeans_label'].astype(str),
    #                     hover_data=['file_path','category','kmeans_label','hdbscan_label','lof_score'],
    #                     title="ESC-50 UMAP 3D colored by KMeans cluster")
    # fig3.update_traces(marker=dict(size=3))
    # fig3.write_html(UMAP3_HTML)
    # print("Saved 3D interactive plot to", UMAP3_HTML)

    print("Done. Outputs saved in", OUT_DIR)

    # open HTMLs unless user requested otherwise
    if not args.no_browser:
        print("Opening interactive plots in your default browser...")
        open_in_browser([UMAP2_HTML])
    else:
        print("Skipping automatic browser open (use --no-browser to suppress).")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true", help="try to download ESC-50 (GitHub zip) into data/")
    p.add_argument("--no-browser", action="store_true", help="do not open the generated HTML files in the browser")
    p.add_argument("--cnn10", action="store_true", help="use local PANNs CNN10 instead of mean+std embeddings")
    p.add_argument("--n-categories", dest="n_categories", type=int, default=None,
                   help="if set, randomly pick this many categories (deterministic via --seed)")
    p.add_argument("--samples-per-category", dest="samples_per_category", type=int, default=None,
                   help="if set with --n-categories, pick this many samples per chosen category")
    p.add_argument("--seed", dest="seed", type=int, default=42, help="random seed for sampling")
    p.add_argument("--feat-mode", choices=["meanstd", "chunks"], default="chunks",
                   help="feature summarization mode (default: chunks)")
    p.add_argument("--n-chunks", dest="n_chunks", type=int, default=5,
                   help="number of time chunks when using feat-mode=chunks (default: 5)")
    args = p.parse_args()
    main(args)
