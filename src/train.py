import argparse
import sys
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from nmf import EuclideanNMF, ISNMF, SparseNMF, ConvolutiveNMF
from audio import load_stems, compute_stft, SOURCE_NAMES

PROJECT     = Path(__file__).parent.parent
MUSDB_TRAIN = PROJECT / "musdb18" / "train"
RESULTS_DIR = PROJECT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SR       = 22050
N_FFT    = 2048
HOP      = 512
DURATION = 20.0

METHODS = {
    "euclidean": lambda K: EuclideanNMF(n_components=K, max_iter=200),
    "isnmf":     lambda K: ISNMF(n_components=K, max_iter=200),
    "sparse":    lambda K: SparseNMF(n_components=K, lam=0.1, max_iter=200),
    "cnmf":      lambda K: ConvolutiveNMF(n_components=K, n_frames=5, max_iter=100),
}


def learn_source_dictionary(method_name, method_factory, source_name, train_paths, K_per_source):
    Xs = []
    for p in train_paths:
        try:
            mag, _ = compute_stft(load_stems(p, sr=SR, duration=DURATION)[source_name],
                                  n_fft=N_FFT, hop_length=HOP)
            Xs.append(mag)
        except Exception as e:
            print(f"  Warning: could not load {p.name}: {e}")

    if not Xs:
        raise RuntimeError(f"No training data for {source_name}")

    X = np.concatenate(Xs, axis=1)
    if X.shape[1] > 3000:
        idx = np.random.choice(X.shape[1], 3000, replace=False)
        X = X[:, np.sort(idx)]

    model = method_factory(K_per_source)
    model.fit(X, fix_W=False)
    return model.W


def train_all(n_train=20, K_per_source=12, methods_to_run=None):
    train_paths = sorted(MUSDB_TRAIN.glob("*.stem.mp4"))[:n_train]
    if not train_paths:
        raise FileNotFoundError(f"No .stem.mp4 files in {MUSDB_TRAIN}")

    methods_to_run = methods_to_run or list(METHODS.keys())
    print(f"Training on {len(train_paths)} tracks, {K_per_source} components/source")

    for method_name in methods_to_run:
        out_path = RESULTS_DIR / f"weights_{method_name}.npz"
        if out_path.exists():
            print(f"[{method_name}] weights exist, skipping (delete to retrain)")
            continue

        print(f"\n=== {method_name.upper()} ===")
        t0 = time.time()
        W_dict = {}
        for src in SOURCE_NAMES:
            print(f"  {src}...")
            W_dict[src] = learn_source_dictionary(
                method_name, METHODS[method_name], src, train_paths, K_per_source
            )

        W_combined = np.concatenate([W_dict[s] for s in SOURCE_NAMES], axis=1)
        print(f"  done in {time.time() - t0:.1f}s, W: {W_combined.shape}")
        np.savez(out_path, **{s: W_dict[s] for s in SOURCE_NAMES},
                 W_combined=W_combined, K_per_source=np.array(K_per_source))

    print("\nWeights saved to results/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train",      type=int, default=20)
    parser.add_argument("--K_per_source", type=int, default=12)
    parser.add_argument("--methods",      nargs="+", default=None,
                        choices=list(METHODS.keys()))
    args = parser.parse_args()
    train_all(n_train=args.n_train, K_per_source=args.K_per_source,
              methods_to_run=args.methods)
