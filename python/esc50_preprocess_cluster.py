#!/usr/bin/env python3
"""
ESC-50 preprocessing + embedding + clustering prototype. A version used for generation of soundiada embeddings.

Usage example:
  # sample 5 categories, 10 samples each, chunked features
  python esc50_preprocess_cluster.py --n-categories 5 --samples-per-category 10 --feat-mode chunks --n-chunks 5
"""

import os
import sys
import argparse
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
import random
import json

# defaults / paths
DATA_DIR = "data/ESC-50-master"
WAV_SUBDIR = "audio"
META_CSV = os.path.join(DATA_DIR, "meta", "esc50.csv")

OUT_DIR = "out"
FEAT_DIR = os.path.join(OUT_DIR, "feats")
EMB_PATH = os.path.join(OUT_DIR, "embeddings.npy")
META_OUT = os.path.join(OUT_DIR, "meta_with_feats.csv")
UMAP2_HTML = os.path.join(OUT_DIR, "umap_2d.html")
JS_OUT = os.path.join(OUT_DIR, "embeddings_for_react.js")

# audio / feature params
SR_TARGET = 22050
TARGET_SECONDS = 5.0
N_MELS = 64
N_FFT = 2048
HOP_LENGTH = 512

# embedding / dim reduction
PCA_DIM = 64
PCA_SKIP_THRESHOLD = 200

# clustering / outlier
HDB_MIN_CLUSTER_SIZE = 6
HDB_MIN_SAMPLES = 2
KMEANS_K = 50

# helpers
def list_wavs_from_meta(meta_csv):
    df = pd.read_csv(meta_csv)
    df['file_path'] = df['filename'].apply(lambda fn: os.path.join(DATA_DIR, WAV_SUBDIR, fn))
    return df

def ensure_outdirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FEAT_DIR, exist_ok=True)

def load_audio(path, sr=SR_TARGET, target_seconds=TARGET_SECONDS):
    y, sr0 = sf.read(path, dtype='float32')
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr0 != sr:
        y = librosa.resample(y=y, orig_sr=sr0, target_sr=sr)
    target_len = int(sr * target_seconds)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y, sr

def compute_logmel(y, sr=SR_TARGET, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    return logmel

def summarize_logmel_to_embedding(logmel, mode="chunks", n_chunks=5):
    n_mels, n_frames = logmel.shape
    if mode == "meanstd":
        mean = np.mean(logmel, axis=1)
        std = np.std(logmel, axis=1)
        emb = np.concatenate([mean, std])
        return emb.astype(np.float32)
    if mode == "chunks":
        if n_chunks <= 0:
            raise ValueError("n_chunks must be >= 1")
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
        chunk_means_flat = np.concatenate(chunk_means, axis=0)
        global_std = np.std(logmel, axis=1)
        emb = np.concatenate([chunk_means_flat, global_std])
        return emb.astype(np.float32)
    raise ValueError(f"Unknown summarize mode: {mode}")

def make_umap_params(n_samples):
    if n_samples <= 100:
        n_neighbors = max(3, min(10, n_samples // 5))
    else:
        n_neighbors = 15
    print(f"UMAP using n_neighbors={n_neighbors} for n_samples={n_samples}")
    return dict(n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=42)

def open_in_browser(paths, delay_between=0.5):
    for p in paths:
        try:
            pth = Path(p).resolve()
            if not pth.exists():
                print(f"[open_in_browser] Warning: {pth} does not exist, skipping.")
                continue
            webbrowser.open_new_tab(pth.as_uri())
            time.sleep(delay_between)
        except Exception as e:
            print(f"[open_in_browser] Failed to open {p}: {e}")

def main(args):
    if not os.path.exists(META_CSV):
        print(f"Cannot find ESC-50 metadata CSV at {META_CSV}")
        print("Extract ESC-50 under data/ESC-50-master/ with meta/esc50.csv present.")
        sys.exit(1)

    ensure_outdirs()
    meta = list_wavs_from_meta(META_CSV)
    print(f"Found {len(meta)} entries in ESC-50 meta.")

    # sampling logic (deterministic behavior)
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

    # embeddings + rows_out
    embeddings = []
    rows_out = []

    emb_tag = f"{args.feat_mode}"
    if args.feat_mode == "chunks":
        emb_tag += f".{int(args.n_chunks)}"

    # process files (with caching behavior)
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
                emb = np.load(feat_path_legacy)
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
        rows_out.append({
            'file_path': wavp,
            'category': r.get('category', ''),
            'fold': r.get('fold', ''),
            'base': base
        })

    if len(embeddings) == 0:
        print("No embeddings were computed. Exiting.")
        sys.exit(1)

    X = np.stack(embeddings, axis=0)
    meta_out = pd.DataFrame(rows_out)
    print("Computed embeddings shape:", X.shape)

    # PCA / scaler decision (preserve threshold)
    if X.shape[0] < PCA_SKIP_THRESHOLD:
        print(f"Small dataset detected (n_samples={X.shape[0]}) — skipping StandardScaler + PCA and using raw embeddings for UMAP.")
        Xp = X.astype(np.float32)
    else:
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

    np.save(EMB_PATH, Xp)
    print("Saved embeddings to", EMB_PATH)

    # UMAP 2D
    print("Computing UMAP 2D...")
    reducer2 = umap.UMAP(**make_umap_params(Xp.shape[0]))
    XY2 = reducer2.fit_transform(Xp)
    meta_out['umap2_x'] = XY2[:, 0]
    meta_out['umap2_y'] = XY2[:, 1]

    # HDBSCAN clustering (for tooltip / analysis)
    print("Clustering with HDBSCAN (for tooltip / analysis)...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=HDB_MIN_CLUSTER_SIZE, min_samples=HDB_MIN_SAMPLES, metric='euclidean')
    hdb_labels = clusterer.fit_predict(Xp)
    meta_out['hdbscan_label'] = hdb_labels

    labels = meta_out['hdbscan_label']
    n_clusters = len(set(labels)) - (1 if -1 in labels.values else 0)
    n_noise = (labels == -1).sum()
    print("HDBSCAN found", n_clusters, "clusters")
    print("Number of noise points:", n_noise)
    print("Unique HDBSCAN labels:", sorted(labels.unique()))

    # k-means used for coloring: if user sampled categories, use that value; else default
    if args.n_categories is not None:
        k_for_plot = int(args.n_categories)
    else:
        k_for_plot = KMEANS_K

    print(f"Clustering with KMeans K={k_for_plot} (used for plot coloring)...")
    km = KMeans(n_clusters=k_for_plot, random_state=42)
    km_labels = km.fit_predict(Xp)
    meta_out['kmeans_label'] = km_labels

    # LOF outlier detection
    print("Computing LOF outlier scores...")
    lof = LocalOutlierFactor(n_neighbors=20, contamination='auto')
    lof_pred = lof.fit_predict(Xp)
    lof_scores = -lof.negative_outlier_factor_
    meta_out['lof_outlier'] = (lof_pred == -1)
    meta_out['lof_score'] = lof_scores

    # save metadata CSV
    meta_out.to_csv(META_OUT, index=False)
    print("Saved meta + cluster labels to", META_OUT)

    # export JS module for soundiada layout
    js_rows = []
    for _, r in meta_out.iterrows():
        if 'base' in r and 'category' in r and 'umap2_x' in r and 'umap2_y' in r:
            js_rows.append({
                "file": r['base'],
                "category": r['category'],
                "x": round(float(r['umap2_x']), 6),
                "y": round(float(r['umap2_y']), 6)
            })
    try:
        with open(JS_OUT, "w", encoding="utf-8") as jf:
            jf.write("export const esc50Embeddings = ")
            jf.write(json.dumps(js_rows, separators=(",", ":"), ensure_ascii=False))
            jf.write(";\n\nexport default esc50Embeddings;\n")
        print("Saved JS embeddings for React to", JS_OUT)
    except Exception as e:
        print("Failed to write JS embeddings file:", e)

    # audio URL column (same form used by original HTML)
    meta_out['audio_rel_path'] = meta_out['base'].apply(
        lambda fn: f"https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/{fn}"
    )

    # jitter to reduce overlap (same RNG seed used in original)
    jitter = 1e-3 * max(1.0, (meta_out['umap2_x'].max() - meta_out['umap2_x'].min()))
    rng = np.random.RandomState(42)
    meta_out['umap2_x_jit'] = meta_out['umap2_x'] + rng.normal(scale=jitter, size=len(meta_out))
    meta_out['umap2_y_jit'] = meta_out['umap2_y'] + rng.normal(scale=jitter, size=len(meta_out))

    # plotly scatter (same hovertemplate and customdata order)
    custom_cols = ['audio_rel_path', 'base', 'category', 'kmeans_label', 'hdbscan_label', 'lof_score']
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

    html_str = fig2.to_html(full_html=True, include_plotlyjs='cdn')
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

    print("Done. Outputs saved in", OUT_DIR)
    if not args.no_browser:
        print("Opening interactive plots in your default browser...")
        open_in_browser([UMAP2_HTML])
    else:
        print("Skipping automatic browser open (use --no-browser to suppress).")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--no-browser", action="store_true", help="do not open the generated HTML files in the browser")
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
