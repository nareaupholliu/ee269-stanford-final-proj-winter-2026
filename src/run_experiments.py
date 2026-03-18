import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from train    import train_all, SR, N_FFT, HOP, DURATION
from evaluate import evaluate_all, compute_summary, print_latex_table, SOURCE_NAMES, RESULTS_DIR
from nmf      import EuclideanNMF, ISNMF, SparseNMF, ConvolutiveNMF
from audio    import load_stems, compute_stft

PROJECT  = Path(__file__).parent.parent
FIGS_DIR = PROJECT / "figures"
FIGS_DIR.mkdir(exist_ok=True)


def sdrvsk_experiment(test_paths, K_values=(10, 20, 30, 40, 50), max_iter=100, n_tracks=3):
    import warnings, museval
    from audio import compute_istft

    test_paths = list(test_paths)[:n_tracks]
    method_factories = {
        "Euclidean NMF":   lambda K: EuclideanNMF(n_components=K, max_iter=max_iter),
        "IS-NMF":          lambda K: ISNMF(n_components=K, max_iter=max_iter),
        "Sparse NMF":      lambda K: SparseNMF(n_components=K, lam=0.1, max_iter=max_iter),
        "Convolutive NMF": lambda K: ConvolutiveNMF(n_components=K, n_frames=5, max_iter=max_iter),
    }

    results = {m: {} for m in method_factories}
    for K in K_values:
        print(f"  K = {K}", flush=True)
        for method_name, factory in method_factories.items():
            sdrs = []
            for track_path in test_paths:
                try:
                    stems   = load_stems(track_path, sr=SR, duration=DURATION)
                    mix_wav = stems["mixture"]
                    X_mag, X_phase = compute_stft(mix_wav, n_fft=N_FFT, hop_length=HOP)
                    X = np.maximum(X_mag, 1e-10)
                    model = factory(K)
                    model.fit(X, fix_W=False)
                    K_ps    = K // 4
                    EPS     = 1e-10
                    W_fit   = model.W_conv[0] if hasattr(model, "W_conv") else model.W
                    WH_full = W_fit @ model.H + EPS
                    ref_arr = np.stack([stems[s] for s in SOURCE_NAMES], axis=0)
                    est_list = []
                    for i, src in enumerate(SOURCE_NAMES):
                        idx  = list(range(i * K_ps, (i + 1) * K_ps))
                        mask = (W_fit[:, idx] @ model.H[idx, :] + EPS) / WH_full
                        est_list.append(compute_istft(mask * X_mag, X_phase,
                                                      hop_length=HOP, length=len(mix_wav)))
                    est_arr = np.stack(est_list, axis=0)
                    T = min(ref_arr.shape[1], est_arr.shape[1])
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        sdr, _, _, _, _ = museval.metrics.bss_eval(ref_arr[:, :T], est_arr[:, :T])
                    sdrs.append(float(np.nanmean(sdr)))
                except Exception as e:
                    print(f"    Warning: {e}")
            results[method_name][K] = float(np.nanmean(sdrs)) if sdrs else float("nan")
            print(f"    {method_name}: {results[method_name][K]:.2f} dB", flush=True)

    return results


def plot_sdrvsk(sdrvsk_data, out_path):
    markers = {"Euclidean NMF": "o", "IS-NMF": "s", "Sparse NMF": "^", "Convolutive NMF": "D"}
    fig, ax = plt.subplots(figsize=(6, 4))
    for method, kv in sdrvsk_data.items():
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
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()


def plot_runtime(summary, out_path):
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
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test",  action="store_true")
    parser.add_argument("--skip-train",  action="store_true")
    parser.add_argument("--skip-sdrvsk", action="store_true")
    parser.add_argument("--methods",     nargs="+", default=None,
                        choices=["euclidean", "isnmf", "sparse", "cnmf"])
    args = parser.parse_args()

    if args.smoke_test:
        methods, n_train, K_per_src, max_iter, max_tracks = \
            args.methods or ["euclidean", "isnmf", "sparse"], 3, 6, 50, 2
    else:
        methods, n_train, K_per_src, max_iter, max_tracks = \
            args.methods or ["euclidean", "isnmf", "sparse", "cnmf"], 20, 12, 100, None

    if not args.skip_train:
        train_all(n_train=n_train, K_per_source=K_per_src, methods_to_run=methods)

    all_results, summary = evaluate_all(
        methods_to_run=methods,
        max_tracks=max_tracks,
        max_iter=max_iter if args.smoke_test else None,
    )

    if not args.skip_sdrvsk and not args.smoke_test:
        test_paths = sorted((PROJECT / "musdb18" / "test").glob("*.stem.mp4"))
        sdrvsk = sdrvsk_experiment(test_paths, n_tracks=5)
        with open(RESULTS_DIR / "sdrvsk.json", "w") as f:
            json.dump(sdrvsk, f, indent=2)
        plot_sdrvsk(sdrvsk, FIGS_DIR / "sdrvsk.pdf")
        plot_sdrvsk(sdrvsk, FIGS_DIR / "sdrvsk.png")

    plot_runtime(summary, FIGS_DIR / "runtime.pdf")
    plot_runtime(summary, FIGS_DIR / "runtime.png")
    print_latex_table(summary)


if __name__ == "__main__":
    main()
