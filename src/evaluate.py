import argparse
import csv
import json
import sys
import time
import warnings
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from nmf import EuclideanNMF, ISNMF, SparseNMF, ConvolutiveNMF
from audio import load_stems, compute_stft, compute_istft, SOURCE_NAMES

import museval

PROJECT     = Path(__file__).parent.parent
MUSDB_TEST  = PROJECT / "musdb18" / "test"
RESULTS_DIR = PROJECT / "results"

SR       = 22050
N_FFT    = 2048
HOP      = 512
DURATION = 20.0

METHOD_FACTORIES = {
    "euclidean": lambda K, **kw: EuclideanNMF(n_components=K, **kw),
    "isnmf":     lambda K, **kw: ISNMF(n_components=K, **kw),
    "sparse":    lambda K, **kw: SparseNMF(n_components=K, lam=0.1, **kw),
    "cnmf":      lambda K, **kw: ConvolutiveNMF(n_components=K, n_frames=5, **kw),
}

METHOD_MAX_ITER = {
    "euclidean": 50,
    "isnmf":     10,
    "sparse":    50,
    "cnmf":      5,
}


def load_weights(method_name):
    path = RESULTS_DIR / f"weights_{method_name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Weights not found: {path}. Run train.py first.")
    data = np.load(path)
    return data["W_combined"].astype(np.float64), int(data["K_per_source"])


def evaluate_track(track_path, method_name, W, K_per_source, max_iter=300):
    stems       = load_stems(track_path, sr=SR, duration=DURATION)
    mix_wav     = stems["mixture"]
    X_mag, X_phase = compute_stft(mix_wav, n_fft=N_FFT, hop_length=HOP)
    X = np.maximum(X_mag, 1e-10)

    comp_splits = {src: list(range(i * K_per_source, (i + 1) * K_per_source))
                   for i, src in enumerate(SOURCE_NAMES)}

    t0 = time.time()
    model = METHOD_FACTORIES[method_name](W.shape[1], max_iter=max_iter)
    model.fit(X, W_init=W, fix_W=True)
    runtime = time.time() - t0

    W_fit   = model.W_conv[0] if method_name == "cnmf" else model.W
    H_fit   = model.H
    EPS     = 1e-10
    WH_full = W_fit @ H_fit + EPS

    estimates = {}
    for src, idx in comp_splits.items():
        mask = (W_fit[:, idx] @ H_fit[idx, :] + EPS) / WH_full
        estimates[src] = compute_istft(mask * X_mag, X_phase,
                                       hop_length=HOP, length=len(mix_wav))

    ref_arr = np.stack([stems[s] for s in SOURCE_NAMES], axis=0)
    est_arr = np.stack([estimates[s] for s in SOURCE_NAMES], axis=0)
    T = min(ref_arr.shape[1], est_arr.shape[1])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sdr, isr, sir, sar, _ = museval.metrics.bss_eval(ref_arr[:, :T], est_arr[:, :T])

    scores = {src: {"SDR": float(np.nanmedian(sdr[i])),
                    "SIR": float(np.nanmedian(sir[i])),
                    "SAR": float(np.nanmedian(sar[i]))}
              for i, src in enumerate(SOURCE_NAMES)}

    return scores, runtime


def evaluate_all(methods_to_run=None, max_tracks=None, max_iter=None):
    test_paths = sorted(MUSDB_TEST.glob("*.stem.mp4"))
    if not test_paths:
        raise FileNotFoundError(f"No test tracks in {MUSDB_TEST}")
    if max_tracks:
        test_paths = test_paths[:max_tracks]

    methods_to_run = methods_to_run or list(METHOD_FACTORIES.keys())
    print(f"Evaluating {len(methods_to_run)} methods on {len(test_paths)} tracks")

    weights = {m: load_weights(m) for m in methods_to_run}

    out_json = RESULTS_DIR / "results.json"
    if out_json.exists():
        with open(out_json) as f:
            saved = json.load(f)
        all_results = saved.get("results", {})
        runtimes    = saved.get("runtimes", {})
    else:
        all_results = {}
        runtimes    = {}

    for method_name in methods_to_run:
        if method_name in all_results and len(all_results[method_name]) >= len(test_paths):
            print(f"[{method_name}] already evaluated, skipping")
            continue

        W, K_ps = weights[method_name]
        if method_name not in all_results:
            all_results[method_name] = {}
            runtimes[method_name]    = []

        iters = max_iter if max_iter is not None else METHOD_MAX_ITER.get(method_name, 50)
        print(f"\n=== {method_name.upper()} (max_iter={iters}) ===")

        for track_path in tqdm(test_paths, desc=method_name):
            track_name = track_path.stem
            if track_name in all_results[method_name]:
                continue
            try:
                scores, rt = evaluate_track(track_path, method_name, W, K_ps, max_iter=iters)
                all_results[method_name][track_name] = scores
                runtimes[method_name].append(rt)
            except Exception as e:
                print(f"  Error on {track_name}: {e}")

        on_disk = json.load(open(out_json)) if out_json.exists() else {"results": {}, "runtimes": {}}
        on_disk["results"][method_name]  = all_results[method_name]
        on_disk["runtimes"][method_name] = runtimes[method_name]
        with open(out_json, "w") as f:
            json.dump(on_disk, f, indent=2)

    summary = compute_summary(all_results, runtimes)
    out_csv = RESULTS_DIR / "summary.csv"
    save_summary_csv(summary, out_csv)
    print(f"Summary saved to {out_csv}")

    return all_results, summary


def compute_summary(all_results, runtimes):
    summary = {}
    for method, tracks in all_results.items():
        summary[method] = {}
        for src in SOURCE_NAMES:
            sdrs = [t[src]["SDR"] for t in tracks.values() if src in t]
            sirs = [t[src]["SIR"] for t in tracks.values() if src in t]
            sars = [t[src]["SAR"] for t in tracks.values() if src in t]
            summary[method][src] = {
                "SDR_mean": float(np.nanmean(sdrs)), "SDR_std": float(np.nanstd(sdrs)),
                "SIR_mean": float(np.nanmean(sirs)), "SIR_std": float(np.nanstd(sirs)),
                "SAR_mean": float(np.nanmean(sars)), "SAR_std": float(np.nanstd(sars)),
            }
        rt = runtimes.get(method, [])
        summary[method]["_runtime"] = {
            "mean_s":  float(np.mean(rt)) if rt else 0.0,
            "total_s": float(np.sum(rt))  if rt else 0.0,
        }
    return summary


def save_summary_csv(summary, path):
    header = ["method", "source", "SDR_mean", "SDR_std", "SIR_mean", "SIR_std", "SAR_mean", "SAR_std"]
    rows = []
    for method, srcs in summary.items():
        for src in SOURCE_NAMES:
            if src in srcs:
                d = srcs[src]
                rows.append([method, src,
                              f"{d['SDR_mean']:.2f}", f"{d['SDR_std']:.2f}",
                              f"{d['SIR_mean']:.2f}", f"{d['SIR_std']:.2f}",
                              f"{d['SAR_mean']:.2f}", f"{d['SAR_std']:.2f}"])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def print_latex_table(summary):
    labels = {"euclidean": "Euclidean NMF", "isnmf": "IS-NMF",
              "sparse": "Sparse NMF", "cnmf": "Convolutive NMF"}
    src_order = ["vocals", "bass", "drums", "other"]
    print("\n% Table 1 — SDR / SIR / SAR per source")
    for method in ["euclidean", "isnmf", "sparse", "cnmf"]:
        if method not in summary:
            continue
        parts = [labels[method], "48"]
        for src in src_order:
            d = summary[method].get(src, {})
            parts += [f"{d.get('SDR_mean', float('nan')):.1f}",
                      f"{d.get('SIR_mean', float('nan')):.1f}",
                      f"{d.get('SAR_mean', float('nan')):.1f}"]
        print(" & ".join(parts) + " \\\\")

    print("\n% Table 2 — runtime")
    for method in ["euclidean", "isnmf", "sparse", "cnmf"]:
        if method not in summary:
            continue
        rt = summary[method].get("_runtime", {})
        print(f"{labels[method]} & {rt.get('mean_s', 0):.1f} \\\\")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods",    nargs="+", default=None,
                        choices=list(METHOD_FACTORIES.keys()))
    parser.add_argument("--max_tracks", type=int, default=None)
    parser.add_argument("--max_iter",   type=int, default=300)
    args = parser.parse_args()

    all_results, summary = evaluate_all(
        methods_to_run=args.methods,
        max_tracks=args.max_tracks,
        max_iter=args.max_iter,
    )
    print_latex_table(summary)
