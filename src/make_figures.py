import json
import sys
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

PROJECT     = Path(__file__).parent.parent
RESULTS_DIR = PROJECT / "results"
FIGS_DIR    = PROJECT / "figures"
FIGS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))
from nmf   import EuclideanNMF, ISNMF, SparseNMF, ConvolutiveNMF
from audio import load_stems, compute_stft, compute_istft, SOURCE_NAMES

SR = 22050; N_FFT = 2048; HOP = 512; DURATION = 20.0

SRC_LABELS = {"drums": "Drums", "bass": "Bass", "other": "Other", "vocals": "Vocals"}
SRC_COLORS = {"drums": "#4C72B0", "bass": "#55A868", "other": "#DD8452", "vocals": "#C44E52"}


def _db(X):
    return 20 * np.log10(np.maximum(X, 1e-10))


def _plot_spec(ax, mag, title, color=None, vmin=-70, vmax=0):
    times = np.arange(mag.shape[1]) * HOP / SR
    freqs = np.linspace(0, SR / 2 / 1000, mag.shape[0])
    img = ax.imshow(_db(mag), origin="lower", aspect="auto", cmap="magma",
                    extent=[times[0], times[-1], freqs[0], freqs[-1]],
                    vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11, fontweight="bold",
                 color=color if color else "black", pad=4)
    ax.set_ylabel("Freq (kHz)", fontsize=8)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.tick_params(labelsize=7)
    return img


def plot_sdrvsk(results, out_stem):
    markers = {"Euclidean NMF": "o", "IS-NMF": "s", "Sparse NMF": "^", "Convolutive NMF": "D"}
    fig, ax = plt.subplots(figsize=(6, 4))
    for method, kv in results.items():
        ks   = sorted(kv.keys())
        vals = [kv[k] for k in ks]
        if any(not np.isnan(v) for v in vals):
            ax.plot(ks, vals, marker=markers.get(method, "o"), label=method, linewidth=1.5)
    ax.set_xlabel("Number of NMF components $K$", fontsize=11)
    ax.set_ylabel("Mean SDR (dB)", fontsize=11)
    ax.set_title("Separation Quality vs. Number of Components", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(str(out_stem.with_suffix(f".{ext}")), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved sdrvsk figures")


def plot_runtime(summary, out_stem):
    methods = ["euclidean", "isnmf", "sparse", "cnmf"]
    labels  = ["Euclidean\nNMF", "IS-NMF", "Sparse\nNMF", "Conv.\nNMF"]
    times   = [summary.get(m, {}).get("_runtime", {}).get("mean_s", 0) for m in methods]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(labels, times, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
    ax.bar_label(bars, fmt="%.1fs", padding=3, fontsize=9)
    ax.set_ylim(0, max(times) * 1.18 if max(times) > 0 else 1)
    ax.set_ylabel("Mean runtime per track (s)", fontsize=10)
    ax.set_title("Computational Cost Comparison", fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(str(out_stem.with_suffix(f".{ext}")), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved runtime figures")


def plot_poster_spectrograms(track_path, method="sparse"):
    stems   = load_stems(track_path, sr=SR, duration=DURATION)
    mix_wav = stems["mixture"]
    X_mag, X_phase = compute_stft(mix_wav, n_fft=N_FFT, hop_length=HOP)
    X = np.maximum(X_mag, 1e-10)

    npz     = np.load(RESULTS_DIR / f"weights_{method}.npz")
    W       = npz["W_combined"].astype(np.float64)
    K_ps    = int(npz["K_per_source"])
    K_total = W.shape[1]

    model = SparseNMF(n_components=K_total, lam=0.1, max_iter=50)
    model.fit(X, W_init=W, fix_W=True)
    H = model.H

    EPS = 1e-10
    WH_full = W @ H + EPS
    estimates = {}
    for i, src in enumerate(SOURCE_NAMES):
        idx = list(range(i * K_ps, (i + 1) * K_ps))
        estimates[src] = (W[:, idx] @ H[idx, :] + EPS) / WH_full * X_mag

    fig = plt.figure(figsize=(16, 9), facecolor="white")
    gs  = gridspec.GridSpec(3, 5, figure=fig,
                            hspace=0.55, wspace=0.35,
                            left=0.06, right=0.97, top=0.90, bottom=0.08)

    ax_mix = fig.add_subplot(gs[0, :])
    img = _plot_spec(ax_mix, X_mag, "Mixture Spectrogram", vmin=-70, vmax=0)
    cbar = fig.colorbar(img, ax=ax_mix, pad=0.01, fraction=0.015)
    cbar.set_label("dB", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    for col, src in enumerate(SOURCE_NAMES):
        ax = fig.add_subplot(gs[1, col])
        idx    = list(range(col * K_ps, (col + 1) * K_ps))
        W_mean = W[:, idx].mean(axis=1)
        freqs  = np.linspace(0, SR / 2 / 1000, len(W_mean))
        W_norm = (W_mean - W_mean.min()) / (W_mean.max() - W_mean.min() + EPS)
        ax.fill_betweenx(freqs, 0, W_norm, color=SRC_COLORS[src], alpha=0.7)
        ax.plot(W_norm, freqs, color=SRC_COLORS[src], linewidth=0.8)
        ax.set_title(SRC_LABELS[src], fontsize=9, fontweight="bold",
                     color=SRC_COLORS[src], pad=3)
        ax.set_xlabel("Norm. magnitude", fontsize=7)
        ax.set_ylabel("Freq (kHz)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_ylim(0, SR / 2 / 1000)
        ax.set_xlim(0, 1)
        ax.grid(True, linestyle="--", alpha=0.3)

    ax_w = fig.add_subplot(gs[1, 4])
    W_db = _db(W)
    ax_w.imshow(W_db, origin="lower", aspect="auto", cmap="viridis",
                vmin=np.percentile(W_db, 5), vmax=np.percentile(W_db, 99))
    ax_w.set_title("Full W Matrix\n(F × K, all sources)", fontsize=9, fontweight="bold", pad=3)
    ax_w.set_xlabel("Component index K", fontsize=7)
    ax_w.set_ylabel("Freq bin", fontsize=7)
    ax_w.tick_params(labelsize=6)
    for j in range(1, 4):
        ax_w.axvline(j * K_ps - 0.5, color="white", linewidth=1.2, alpha=0.8)
    for j, src in enumerate(SOURCE_NAMES):
        ax_w.text(K_ps * j + K_ps / 2, W_db.shape[0] + 5,
                  SRC_LABELS[src][:3], ha="center", va="bottom", fontsize=6)

    for col, src in enumerate(SOURCE_NAMES):
        ax = fig.add_subplot(gs[2, col])
        _plot_spec(ax, estimates[src], SRC_LABELS[src], color=SRC_COLORS[src], vmin=-70, vmax=0)

    _plot_spec(fig.add_subplot(gs[2, 4]), X_mag, "Mixture\n(reference)", vmin=-70, vmax=0)

    fig.suptitle(
        f"Spectral Decomposition — {track_path.stem.replace('.stem', '')}"
        f"\n(Sparse NMF, K=48, supervised W)",
        fontsize=13, fontweight="bold", y=0.97,
    )
    for ext, dpi in (("pdf", 150), ("png", 200)):
        plt.savefig(str(FIGS_DIR / f"poster_spectrograms.{ext}"), dpi=dpi)
    plt.close()
    print(f"Saved poster spectrograms")


if __name__ == "__main__":
    with open(RESULTS_DIR / "results.json") as f:
        data = json.load(f)
    all_results = data["results"]
    runtimes    = data.get("runtimes", {})

    summary = {m: {"_runtime": {"mean_s":  float(np.mean(rt)) if rt else 0.0,
                                "total_s": float(np.sum(rt))  if rt else 0.0}}
               for m, rt in ((m, runtimes.get(m, [])) for m in all_results)}

    valid_names = set(all_results["euclidean"].keys())
    all_test    = sorted((PROJECT / "musdb18" / "test").glob("*.stem.mp4"))
    valid_paths = [p for p in all_test if p.stem in valid_names]
    print(f"Valid tracks: {len(valid_paths)}")

    import museval
    K_values = [8, 16, 24, 32, 40, 48]
    method_factories = {
        "Euclidean NMF":   lambda K: EuclideanNMF(n_components=K, max_iter=30),
        "IS-NMF":          lambda K: ISNMF(n_components=K, max_iter=10),
        "Sparse NMF":      lambda K: SparseNMF(n_components=K, lam=0.1, max_iter=30),
        "Convolutive NMF": lambda K: ConvolutiveNMF(n_components=K, n_frames=5, max_iter=5),
    }

    track_path = valid_paths[0]
    print(f"Track: {track_path.name}")

    stems   = load_stems(track_path, sr=SR, duration=DURATION)
    mix_wav = stems["mixture"]
    X_mag, X_phase = compute_stft(mix_wav, n_fft=N_FFT, hop_length=HOP)
    X       = np.maximum(X_mag, 1e-10)
    ref_arr = np.stack([stems[s] for s in SOURCE_NAMES], axis=0)

    sdrvsk = {m: {} for m in method_factories}
    for K in K_values:
        print(f"  K = {K}", flush=True)
        for method_name, factory in method_factories.items():
            try:
                model   = factory(K)
                model.fit(X.copy(), fix_W=False)
                K_ps    = K // 4
                W_fit   = model.W_conv[0] if hasattr(model, "W_conv") else model.W
                WH_full = W_fit @ model.H + 1e-10
                est_list = []
                for i, src in enumerate(SOURCE_NAMES):
                    idx  = list(range(i * K_ps, (i + 1) * K_ps))
                    mask = (W_fit[:, idx] @ model.H[idx, :] + 1e-10) / WH_full
                    est_list.append(compute_istft(mask * X_mag, X_phase,
                                                  hop_length=HOP, length=len(mix_wav)))
                est_arr = np.stack(est_list, axis=0)
                T = min(ref_arr.shape[1], est_arr.shape[1])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sdr, _, _, _, _ = museval.metrics.bss_eval(ref_arr[:, :T], est_arr[:, :T])
                sdrvsk[method_name][K] = float(np.nanmean(sdr))
            except Exception as e:
                print(f"    Warning ({method_name}, K={K}): {e}")
                sdrvsk[method_name][K] = float("nan")
            print(f"    {method_name}: {sdrvsk[method_name].get(K, float('nan')):.2f} dB", flush=True)

    with open(RESULTS_DIR / "sdrvsk.json", "w") as f:
        json.dump(sdrvsk, f, indent=2)

    plot_sdrvsk(sdrvsk, FIGS_DIR / "sdrvsk")
    plot_runtime(summary, FIGS_DIR / "runtime")
    plot_poster_spectrograms(track_path)
    print("Done.")
