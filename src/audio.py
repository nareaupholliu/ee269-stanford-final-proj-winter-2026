import numpy as np
import stempeg
import librosa

STEM_NAMES = ["mixture", "drums", "bass", "other", "vocals"]
SOURCE_NAMES = ["drums", "bass", "other", "vocals"]


def load_stems(path, sr=22050, duration=30.0):
    audio, rate = stempeg.read_stems(str(path))
    if duration is not None:
        audio = audio[:, :int(duration * rate), :]
    stems = {}
    for i, name in enumerate(STEM_NAMES):
        mono = audio[i].mean(axis=1)
        if rate != sr:
            mono = librosa.resample(mono, orig_sr=rate, target_sr=sr)
        stems[name] = mono.astype(np.float32)
    return stems


def compute_stft(y, n_fft=2048, hop_length=512):
    D = librosa.stft(y.astype(np.float32), n_fft=n_fft, hop_length=hop_length,
                     window='hann', center=True)
    magnitude = np.abs(D).astype(np.float64)
    phase = np.exp(1j * np.angle(D))
    return magnitude, phase


def compute_istft(magnitude, phase, hop_length=512, length=None):
    D = magnitude * phase
    y = librosa.istft(D.astype(np.complex64), hop_length=hop_length,
                      window='hann', center=True, length=length)
    return y.astype(np.float32)


def reconstruct_sources(X_mag, X_phase, W, H, component_splits, hop_length=512, ref_length=None):
    from nmf import EPS
    WH_full = W @ H + EPS
    waveforms = {}
    for name, idx in component_splits.items():
        mask = (W[:, idx] @ H[idx, :] + EPS) / WH_full
        waveforms[name] = compute_istft(mask * X_mag, X_phase,
                                        hop_length=hop_length, length=ref_length)
    return waveforms
