"""
Feature extraction for vibration fault detection.

Computes 35 features from 3-axis acceleration data, matching firmware formulas.
Handles 1-axis and 2-axis data by producing NaN for missing axes.
"""

import numpy as np
from scipy import signal as sig
from scipy.fft import rfft, rfftfreq
import json
import os


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------

def tukey_window(n, taper_size=48):
    """Firmware-style Tukey window matching __hanning() with wn=48.

    Hanning tapers on the first and last *taper_size // 2* samples,
    flat 1.0 in the middle.
    """
    if n <= 0:
        return np.array([], dtype=np.float64)
    w = np.ones(n, dtype=np.float64)
    half = taper_size // 2  # 24 samples each side
    if n <= taper_size:
        return np.hanning(n).astype(np.float64)
    taper = 0.5 * (1.0 - np.cos(np.pi * np.arange(half) / half))
    w[:half] = taper
    w[-half:] = taper[::-1]
    return w


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------

def compute_rms(x):
    """Time-domain RMS: sqrt(mean(x**2))."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return 0.0
    return float(np.sqrt(np.mean(x ** 2)))


def compute_fft(x, fs, taper_size=48):
    """Tukey-windowed one-sided FFT with zero-padding to next power of 2.

    Returns
    -------
    spectrum : complex ndarray  – one-sided rfft coefficients
    freqs    : ndarray          – frequency bins (Hz)
    n_fft    : int              – FFT length (after zero-padding)
    n_signal : int              – original signal length
    """
    x = np.asarray(x, dtype=np.float64)
    n_signal = len(x)
    if n_signal == 0:
        return np.array([0j]), np.array([0.0]), 1, 0
    w = tukey_window(n_signal, taper_size)
    x_w = x * w
    n_fft = int(2 ** np.ceil(np.log2(n_signal)))
    spectrum = rfft(x_w, n=n_fft)
    freqs = rfftfreq(n_fft, d=1.0 / fs)
    return spectrum, freqs, n_fft, n_signal


def compute_velocity_spectrum(acc_spectrum, freqs):
    """Convert acceleration spectrum to velocity via omega integration.

    velocity(f) = acc(f) / (j * 2 * pi * f) * 9806.65
    DC bin is zeroed.  Units: g -> mm/s.
    """
    vel = np.zeros_like(acc_spectrum)
    nz = freqs > 0
    vel[nz] = acc_spectrum[nz] / (1j * 2.0 * np.pi * freqs[nz]) * 9806.65
    return vel


def compute_parseval_rms(spectrum, n_signal, n_fft):
    """Parseval RMS from one-sided rfft output.

    rms = sqrt(2 * sum(|X[k]|^2) / n_signal / n_fft)  for k = 0 .. n_fft/2 - 1
    """
    if n_signal == 0 or n_fft == 0:
        return 0.0
    mag_sq = np.abs(spectrum) ** 2
    total = np.sum(mag_sq[: n_fft // 2])
    return float(np.sqrt(2.0 * total / n_signal / n_fft))


def compute_envelope_rms(x, fs):
    """Envelope RMS: 4th-order Butterworth HPF at fs/4, then time-domain RMS."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 13:
        return compute_rms(x)
    nyq = fs / 2.0
    cutoff = fs / 4.0
    wn = np.clip(cutoff / nyq, 0.01, 0.99)
    b, a = sig.butter(4, wn, btype='high')
    try:
        filtered = sig.filtfilt(b, a, x)
    except ValueError:
        return compute_rms(x)
    return compute_rms(filtered)


def compute_kurtosis(x):
    """Kurtosis: (1/N * sum((x - mu)^4)) / sigma^4."""
    x = np.asarray(x, dtype=np.float64)
    mu = np.mean(x)
    var = np.mean((x - mu) ** 2)
    if var == 0:
        return 0.0
    return float(np.mean((x - mu) ** 4) / (var ** 2))


def compute_peak_to_peak(x):
    """Peak-to-peak: max(x) - min(x)."""
    x = np.asarray(x)
    return float(np.max(x) - np.min(x))


def compute_composite_crest_factor(axes_data):
    """Composite crest factor across available axes.

    maxCf = sqrt(sum((peak_i - mean_i)^2)) / sqrt(sum(rms_i^2))

    peak_i = max(|x_i|), mean_i = mean(x_i).
    Only uses axes that are not None.
    """
    num_sq = 0.0
    den_sq = 0.0
    for a in axes_data:
        if a is None:
            continue
        a = np.asarray(a, dtype=np.float64)
        peak = float(np.max(np.abs(a)))
        mean_val = float(np.mean(a))
        rms = compute_rms(a)
        num_sq += (peak - mean_val) ** 2
        den_sq += rms ** 2
    if den_sq == 0:
        return 0.0
    return float(np.sqrt(num_sq) / np.sqrt(den_sq))


# ---------------------------------------------------------------------------
# RPM estimation
# ---------------------------------------------------------------------------

def estimate_rpm(x, fs, min_rpm=300, max_rpm=6000):
    """Estimate RPM from dominant FFT peak in the 5-100 Hz range.

    Falls back to 1800 RPM if no clear peak is found.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return 1800.0

    min_freq = min_rpm / 60.0
    max_freq = max_rpm / 60.0

    n = len(x)
    n_fft = int(2 ** np.ceil(np.log2(n)))
    spectrum = rfft(x, n=n_fft)
    freqs = rfftfreq(n_fft, d=1.0 / fs)
    magnitudes = np.abs(spectrum)

    mask = (freqs >= min_freq) & (freqs <= max_freq)
    if not np.any(mask):
        return 1800.0

    masked_mags = np.zeros_like(magnitudes)
    masked_mags[mask] = magnitudes[mask]

    peak_idx = int(np.argmax(masked_mags))
    peak_freq = freqs[peak_idx]
    if peak_freq == 0:
        return 1800.0

    return float(peak_freq * 60.0)


# ---------------------------------------------------------------------------
# Peak ratios
# ---------------------------------------------------------------------------

def find_peak_in_band(magnitude_spectrum, freqs, f_low, f_high):
    """Highest peak (via scipy find_peaks) in [f_low, f_high].

    Returns peak magnitude, or 0.0 if band is empty.
    """
    mask = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(mask):
        return 0.0

    band_mags = magnitude_spectrum[mask]
    if len(band_mags) < 3:
        return float(np.max(band_mags)) if len(band_mags) > 0 else 0.0

    peaks, _ = sig.find_peaks(band_mags)
    if len(peaks) == 0:
        return float(np.max(band_mags))

    return float(np.max(band_mags[peaks]))


def compute_peak_ratios(acc_spectrum, vel_spectrum, freqs, rpm,
                        acc_rms, vel_rms, n_signal):
    """Per-axis peak ratios in RPM-relative bands.

    Bands (X = rotation freq = rpm / 60):
        Low  :  0.5X –  5.5X
        Mid  :  5.5X – 10.5X
        High : 10.5X – 30.5X

    Returns dict with 6 keys: acc{Low,Mid,High}PeakRatio, vel{Low,Mid,High}PeakRatio
    """
    rot_freq = rpm / 60.0
    bands = {
        'Low':  (0.5  * rot_freq,  5.5 * rot_freq),
        'Mid':  (5.5  * rot_freq, 10.5 * rot_freq),
        'High': (10.5 * rot_freq, 30.5 * rot_freq),
    }

    # Normalize FFT magnitudes to physical amplitude: 2 * |X[k]| / n_signal
    acc_mag = np.abs(acc_spectrum) * 2.0 / max(n_signal, 1)
    vel_mag = np.abs(vel_spectrum) * 2.0 / max(n_signal, 1)

    result = {}
    for band_name, (f_low, f_high) in bands.items():
        acc_peak = find_peak_in_band(acc_mag, freqs, f_low, f_high)
        vel_peak = find_peak_in_band(vel_mag, freqs, f_low, f_high)
        result[f'acc{band_name}PeakRatio'] = acc_peak / acc_rms if acc_rms > 0 else 0.0
        result[f'vel{band_name}PeakRatio'] = vel_peak / vel_rms if vel_rms > 0 else 0.0

    return result


# ---------------------------------------------------------------------------
# Per-axis feature helper
# ---------------------------------------------------------------------------

def _axis_features(data, fs, rpm, axis_label):
    """Compute all per-axis features for one axis.

    Returns dict with keys like xRMS, xVRMS, xEnvRMS, xKU, xP2P,
    and acc/vel peak ratio keys suffixed with axis_label.
    """
    prefix = axis_label  # 'x', 'y', or 'z'
    label = axis_label.upper()  # 'X', 'Y', 'Z'

    if data is None:
        features = {
            f'{prefix}RMS': np.nan,
            f'{prefix}VRMS': np.nan,
            f'{prefix}EnvRMS': np.nan,
            f'{prefix}KU': np.nan,
            f'{prefix}P2P': np.nan,
        }
        for band in ('Low', 'Mid', 'High'):
            features[f'acc{band}PeakRatio{label}'] = np.nan
            features[f'vel{band}PeakRatio{label}'] = np.nan
        return features

    arr = np.asarray(data, dtype=np.float64)

    # Time-domain
    rms = compute_rms(arr)
    env_rms = compute_envelope_rms(arr, fs)
    kurtosis = compute_kurtosis(arr)
    p2p = compute_peak_to_peak(arr)

    # Frequency-domain
    acc_spec, freqs, n_fft, n_signal = compute_fft(arr, fs)
    vel_spec = compute_velocity_spectrum(acc_spec, freqs)
    vrms = compute_parseval_rms(vel_spec, n_signal, n_fft)

    # Peak ratios
    pr = compute_peak_ratios(acc_spec, vel_spec, freqs, rpm,
                             rms, vrms, n_signal)

    features = {
        f'{prefix}RMS': rms,
        f'{prefix}VRMS': vrms,
        f'{prefix}EnvRMS': env_rms,
        f'{prefix}KU': kurtosis,
        f'{prefix}P2P': p2p,
    }
    for band in ('Low', 'Mid', 'High'):
        features[f'acc{band}PeakRatio{label}'] = pr[f'acc{band}PeakRatio']
        features[f'vel{band}PeakRatio{label}'] = pr[f'vel{band}PeakRatio']

    return features


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

# Canonical order of all 35 features
FEATURE_NAMES = [
    'temp',
    'xRMS', 'yRMS', 'zRMS',
    'xVRMS', 'yVRMS', 'zVRMS',
    'xEnvRMS', 'yEnvRMS', 'zEnvRMS',
    'xKU', 'yKU', 'zKU',
    'xP2P', 'yP2P', 'zP2P',
    'maxCf',
    'accLowPeakRatioX', 'accLowPeakRatioY', 'accLowPeakRatioZ',
    'accMidPeakRatioX', 'accMidPeakRatioY', 'accMidPeakRatioZ',
    'accHighPeakRatioX', 'accHighPeakRatioY', 'accHighPeakRatioZ',
    'velLowPeakRatioX', 'velLowPeakRatioY', 'velLowPeakRatioZ',
    'velMidPeakRatioX', 'velMidPeakRatioY', 'velMidPeakRatioZ',
    'velHighPeakRatioX', 'velHighPeakRatioY', 'velHighPeakRatioZ',
]


def extract_features(x, y=None, z=None, fs=48000, rpm=None, temp=None):
    """Extract 35 features from up to 3-axis acceleration data.

    Parameters
    ----------
    x : array-like
        X-axis acceleration (required).
    y, z : array-like or None
        Y / Z-axis acceleration.  None → NaN for those features.
    fs : float
        Sample rate in Hz.
    rpm : float or None
        Shaft RPM.  If None, estimated from x-axis data.
    temp : float or None
        Temperature reading (pass-through).

    Returns
    -------
    dict  – 35 features keyed by FEATURE_NAMES.
    """
    x = np.asarray(x, dtype=np.float64)
    if y is not None:
        y = np.asarray(y, dtype=np.float64)
    if z is not None:
        z = np.asarray(z, dtype=np.float64)

    # RPM estimation
    if rpm is None or rpm <= 0:
        rpm = estimate_rpm(x, fs)

    # Per-axis features
    fx = _axis_features(x, fs, rpm, 'x')
    fy = _axis_features(y, fs, rpm, 'y')
    fz = _axis_features(z, fs, rpm, 'z')

    # Composite crest factor (uses available axes)
    axes_data = [a for a in [x, y, z] if a is not None]
    max_cf = compute_composite_crest_factor(axes_data)

    # Assemble output
    features = {'temp': temp if temp is not None else np.nan}
    features.update(fx)
    features.update(fy)
    features.update(fz)
    features['maxCf'] = max_cf

    # Return in canonical order
    return {name: features[name] for name in FEATURE_NAMES}


def extract_features_from_json(filepath):
    """Load a segment JSON and extract all 35 features plus metadata.

    Returns
    -------
    dict  – features + metadata keys (filename, dataset, fault_category, etc.)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # --- signal data (handle JSON null → None) ---
    original = data.get('original_data', {})
    x_raw = original.get('x')
    y_raw = original.get('y')
    z_raw = original.get('z')
    x = np.asarray(x_raw, dtype=np.float64) if x_raw is not None else None
    y = np.asarray(y_raw, dtype=np.float64) if y_raw is not None else None
    z = np.asarray(z_raw, dtype=np.float64) if z_raw is not None else None

    if x is None:
        raise ValueError(f"No x-axis data in {filepath}")

    # --- sample rate ---
    sensor = data.get('sensor', {})
    signal_info = data.get('signal', {})
    fs = sensor.get('sample_rate_hz') or signal_info.get('sample_rate_hz')
    if fs is None:
        raise ValueError(f"No sample_rate_hz in {filepath}")
    fs = float(fs)

    # --- unit conversion (mg → g) ---
    units = sensor.get('units', 'g')
    if units == 'mg':
        x = x / 1000.0
        if y is not None:
            y = y / 1000.0
        if z is not None:
            z = z / 1000.0

    # --- RPM (may be top-level or nested in operating_conditions) ---
    condition = data.get('condition', {})
    rpm = condition.get('rpm')
    if rpm is None:
        oc = condition.get('operating_conditions', {})
        if isinstance(oc, dict):
            rpm = oc.get('rpm')
    if rpm is not None:
        rpm = float(rpm)

    # --- temperature (pass-through if present) ---
    temp = condition.get('temperature')

    # --- estimate RPM if still missing ---
    rpm_estimated = False
    if rpm is None or rpm <= 0:
        rpm = estimate_rpm(x, fs)
        rpm_estimated = True

    # --- extract features ---
    features = extract_features(x, y, z, fs, rpm=rpm, temp=temp)

    # --- attach metadata ---
    meta = data.get('metadata', {})
    features['filename'] = os.path.basename(filepath)
    features['dataset'] = meta.get('dataset', '')
    features['fault_category'] = condition.get('fault_category', '')
    features['fault_type'] = condition.get('fault_type', condition.get('condition_label', ''))
    features['rpm_used'] = rpm
    features['rpm_estimated'] = rpm_estimated
    features['sample_rate_hz'] = fs

    return features


# ---------------------------------------------------------------------------
# Batch extraction helper
# ---------------------------------------------------------------------------

def extract_all(data_dir, output_csv=None, verbose=True):
    """Extract features from every JSON in *data_dir* and return a DataFrame.

    Optionally saves to *output_csv*.
    """
    import pandas as pd
    import glob

    files = sorted(glob.glob(os.path.join(data_dir, '**', '*.json'), recursive=True))
    if verbose:
        print(f"Found {len(files)} JSON files in {data_dir}")

    rows = []
    for i, fp in enumerate(files):
        try:
            row = extract_features_from_json(fp)
            rows.append(row)
        except Exception as e:
            if verbose:
                print(f"  SKIP {os.path.basename(fp)}: {e}")
        if verbose and (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(files)}")

    df = pd.DataFrame(rows)
    if verbose:
        print(f"Extracted features from {len(df)} segments.")

    if output_csv:
        df.to_csv(output_csv, index=False)
        if verbose:
            print(f"Saved to {output_csv}")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python feature_extraction.py <data_dir> [output.csv]")
        sys.exit(1)
    data_dir = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else 'features.csv'
    extract_all(data_dir, output_csv)
