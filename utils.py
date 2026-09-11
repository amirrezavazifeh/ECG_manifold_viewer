"""
utils.py
--------
Utility functions for loading and preprocessing multi-lead ECG signals.

Designed around a generic 12-lead layout:
    I, II, III, AVF, AVL, AVR, V1, V2, V3, V4, V5, V6

Also works transparently with 2-lead databases such as MIT-BIH (leads
MLII + V1): leads that are not physically recorded in a given record are
simply filled with NaN and left unused, so the exact same downstream code
(R-peak detection, beat segmentation, model input) works regardless of how
many leads a record actually has.

Modular steps:
    1. load_ecg_signal        -> raw (12, T) signal, access via signal["V1"]
    2. load_or_compute_rpeaks -> R-peak indices (+ MIT-BIH symbols if present)
    3. extract_heartbeats     -> (12, N, window_len) segmented beats,
                                  access via heartbeats["II"], + AAMI labels
    4. compute_embeddings     -> UMAP / t-SNE / PCA 2D embeddings, one lead at a time
                                  (features are standardized to zero mean / unit
                                  variance before any of the three methods run)
    5. save_results / load_results -> persist/restore the full pipeline output
                                        (signal, heartbeats, embeddings, AAMI
                                        labels, metadata, fs) as a single .npz.
                                        Embeddings are stored/restored as one
                                        (12, N, 2) array per method (umap/tsne/pca),
                                        same (12, ...) layout as signal/heartbeats.
    6. main                   -> runs steps 1-5 end to end for one record
    7. export_atlas_json      -> converts a main() results dict into the JSON
                                   schema consumed by the ECG Atlas Visualization HTML
"""

import numpy as np
import scipy.signal
import wfdb
import neurokit2 as nk
from skimage.transform import resize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from umap import UMAP


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STANDARD_12_LEAD_ORDER = ['I', 'II', 'III', 'AVF', 'AVL', 'AVR',
                           'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Maps a lead name as it might appear in a wfdb record header to our
# canonical 12-lead slot name.
LEAD_NAME_ALIASES = {
    'I': 'I', 'II': 'II', 'III': 'III',
    'AVR': 'AVR', 'AVL': 'AVL', 'AVF': 'AVF',
    'aVR': 'AVR', 'aVL': 'AVL', 'aVF': 'AVF',
    'V1': 'V1', 'V2': 'V2', 'V3': 'V3',
    'V4': 'V4', 'V5': 'V5', 'V6': 'V6',
    # MIT-BIH Arrhythmia Database specific leads
    'MLII': 'II',    # modified limb lead II -> standard lead II slot
    'MLIII': 'III',
}

AAMI_MAP = {
    'N': 0, '.': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,   # Normal
    'V': 1, 'E': 1,                                    # Ventricular ectopic
    'A': 2, 'S': 2, 'J': 2, 'a': 2,                     # Supraventricular ectopic
    'F': 3,                                             # Fusion
    'Q': 4, '/': 4, 'f': 4,                              # Unknown / paced
}
AAMI_CLASS_NAMES = {0: 'N', 1: 'V', 2: 'S', 3: 'F', 4: 'Q', 5: 'U'}


# ---------------------------------------------------------------------------
# Container: lets you index a (12, ...) array by lead name
# ---------------------------------------------------------------------------

class LeadArray:
    """
    Thin wrapper around a numpy array that lets you index leads by name.

    Works for:
        - raw signals:    shape (12, T)
        - segmented beats: shape (12, N, window_len)

    Example
    -------
    >>> ecg_signals["V1"]        # shape (T,)
    >>> heartbeats["II"]         # shape (N, window_len)
    >>> ecg_signals.data         # underlying raw numpy array
    """

    def __init__(self, data, lead_order=STANDARD_12_LEAD_ORDER):
        self.data = np.asarray(data)
        self.lead_order = list(lead_order)
        self._lead_idx = {lead: i for i, lead in enumerate(self.lead_order)}

    def __getitem__(self, key):
        if isinstance(key, str):
            key = key.upper()
            if key not in self._lead_idx:
                raise KeyError(f"Unknown lead '{key}'. Available slots: {self.lead_order}")
            return self.data[self._lead_idx[key]]
        return self.data[key]

    def __setitem__(self, key, value):
        if isinstance(key, str):
            key = key.upper()
            self.data[self._lead_idx[key]] = value
        else:
            self.data[key] = value

    @property
    def shape(self):
        return self.data.shape

    @property
    def available_leads(self):
        """Leads that actually contain real (non-NaN) data for this record."""
        return [lead for lead in self.lead_order
                if not np.all(np.isnan(self.data[self._lead_idx[lead]]))]

    def __repr__(self):
        return (f"LeadArray(shape={self.data.shape}, "
                f"leads={self.lead_order}, available={self.available_leads})")


# ---------------------------------------------------------------------------
# 1. Loading raw signals
# ---------------------------------------------------------------------------

def load_ecg_signal(record_name, record_path_base, lead_order=STANDARD_12_LEAD_ORDER,
                     lead_aliases=LEAD_NAME_ALIASES, verbose=True):
    """
    Load a raw multi-lead ECG record into a fixed 12-lead layout.

    Any lead not physically present in the record is filled with NaN, so a
    2-lead database (MIT-BIH) and a true 12-lead database can be consumed by
    exactly the same downstream code.

    Args:
        record_name (str or int): Record identifier (e.g. 100 for MIT-BIH).
        record_path_base (str): Directory containing the record files
            (record path is built as f"{record_path_base}{record_name}").
        lead_order (list): Canonical lead ordering to output.
        lead_aliases (dict): Maps a header sig_name -> canonical lead name.
        verbose (bool): Print a short summary after loading.

    Returns:
        ecg_signals (LeadArray): shape (12, T). Access via ecg_signals["V1"].
        fs (float): Sampling frequency in Hz.
        metadata (dict): Record-level metadata (age, sex, comments, ...).
    """

    record_path = f"{record_path_base}{record_name}"
    signals, fields = wfdb.rdsamp(record_path)  # signals: (T, n_sig)

    fs = fields['fs']
    sig_names = fields['sig_name']

    T = signals.shape[0]
    data = np.full((len(lead_order), T), np.nan, dtype=float)
    lead_idx = {lead: i for i, lead in enumerate(lead_order)}

    matched_leads = {}
    for col, raw_name in enumerate(sig_names):
        canonical = lead_aliases.get(raw_name, raw_name.upper())
        if canonical in lead_idx:
            data[lead_idx[canonical]] = signals[:, col]
            matched_leads[raw_name] = canonical

    ecg_signals = LeadArray(data, lead_order=lead_order)

    # Optional demographic metadata (MIT-BIH style header comments: "AGE SEX ...")
    comments = fields.get('comments', [])
    age, sex, sex_text = None, None, None
    if comments:
        try:
            tokens = comments[0].split()
            age = int(tokens[0])
            sex_text = tokens[1]
            sex = 0 if sex_text == 'M' else 1
        except (ValueError, IndexError):
            pass

    metadata = {
        'record_name': record_name,
        'fs': fs,
        'raw_sig_names': sig_names,
        'matched_leads': matched_leads,
        'age': age,
        'sex': sex,
        'sex_text': sex_text,
        'comments': comments,
    }

    if verbose:
        print(f"Loaded record {record_name}: {T} samples @ {fs} Hz")
        print(f"  Raw leads in file : {sig_names}")
        print(f"  Mapped to slots   : {matched_leads}")
        print(f"  Signal shape      : {ecg_signals.shape}")

    return ecg_signals, fs, metadata


# ---------------------------------------------------------------------------
# 2. R-peak detection / loading
# ---------------------------------------------------------------------------

def load_or_compute_rpeaks(ecg_signals, fs, record_name=None, record_path_base=None,
                            annotation_extension='atr', detection_lead='II', verbose=True):
    """
    Load R-peaks from an existing wfdb annotation file if available,
    otherwise detect them from scratch using NeuroKit2.

    Args:
        ecg_signals (LeadArray): shape (12, T), as returned by load_ecg_signal.
        fs (float): Sampling frequency in Hz.
        record_name (str or int, optional): Needed to look up an annotation file.
        record_path_base (str, optional): Directory containing the record/annotation
            files (only needed if record_name is given).
        annotation_extension (str): wfdb annotation extension, e.g. 'atr'.
        detection_lead (str): Which lead to run NeuroKit2 on if no annotation
            file is found (must be an available lead for this record).
        verbose (bool): Print a short summary.

    Returns:
        r_peaks (np.ndarray): Sample indices of the R-peaks.
        beat_symbols (np.ndarray or None): MIT-BIH style annotation symbols
            aligned with r_peaks, or None if no annotation file was found.
        source (str): 'annotation' or 'neurokit'.
    """

    beat_symbols = None
    r_peaks = None
    source = None

    if record_name is not None and record_path_base is not None:
        record_path = f"{record_path_base}{record_name}"
        try:
            annotation = wfdb.rdann(record_path, annotation_extension)
            r_peaks = np.array(annotation.sample)
            beat_symbols = np.array(annotation.symbol)
            source = 'annotation'
        except FileNotFoundError:
            source = None  # fall through to NeuroKit2 detection

    if r_peaks is None:
        lead_signal = ecg_signals[detection_lead]
        if np.all(np.isnan(lead_signal)):
            raise ValueError(
                f"Lead '{detection_lead}' is not available in this record "
                f"(available: {ecg_signals.available_leads}); pass a different detection_lead."
            )
        _, rpeaks_dict = nk.ecg_peaks(lead_signal, sampling_rate=fs)
        r_peaks = rpeaks_dict['ECG_R_Peaks']
        source = 'neurokit'

    if verbose:
        print(f"R-peaks source: {source}")
        print(f"  Number of R-peaks: {len(r_peaks)}")

    return r_peaks, beat_symbols, source


# ---------------------------------------------------------------------------
# 3. Heartbeat segmentation
# ---------------------------------------------------------------------------

def extract_heartbeats(ecg_signals, r_peaks, window_len=256, beat_symbols=None,
                        baseline_kernel=127, verbose=True):
    """
    Segment individual heartbeats around each R-peak, for every lead, using
    1/3 of the previous RR interval before the peak and 2/3 of the next RR
    interval after the peak, then resample to a fixed window length and
    remove baseline wander with a median filter.

    The first and last R-peaks are dropped since they lack, respectively, a
    previous and next R-peak to define their segment (matching the original
    single-lead implementation).

    Args:
        ecg_signals (LeadArray): shape (12, T).
        r_peaks (np.ndarray): R-peak sample indices, sorted ascending.
        window_len (int): Number of samples per beat after resampling.
        beat_symbols (np.ndarray, optional): MIT-BIH style symbols aligned
            with r_peaks. If given, AAMI superclass labels are returned too.
        baseline_kernel (int): Median filter kernel size for baseline removal
            (must be odd).
        verbose (bool): Print a short summary.

    Returns:
        heartbeats (LeadArray): shape (12, N, window_len).
            Access via heartbeats["II"] -> shape (N, window_len).
        labels (dict or None): {'symbols': ..., 'AAMI': ...} if beat_symbols
            was provided, otherwise None.
    """

    r_peaks = np.asarray(r_peaks)
    valid_idx = np.arange(1, len(r_peaks) - 1)

    prev_peaks = r_peaks[valid_idx - 1]
    curr_peaks = r_peaks[valid_idx]
    next_peaks = r_peaks[valid_idx + 1]

    segment_starts = curr_peaks - np.round((curr_peaks - prev_peaks) / 3).astype(int)
    segment_ends = curr_peaks + np.round(2 * (next_peaks - curr_peaks) / 3).astype(int)

    n_leads = ecg_signals.data.shape[0]
    n_beats = len(valid_idx)
    beats_data = np.full((n_leads, n_beats, window_len), np.nan, dtype=float)

    available_lead_names = ecg_signals.available_leads
    available_idx = [ecg_signals._lead_idx[l] for l in available_lead_names]

    for i in range(n_beats):
        start, end = segment_starts[i], segment_ends[i]
        if end <= start:
            continue  # degenerate segment, leave as NaN
        for lead_i in available_idx:
            raw_seg = ecg_signals.data[lead_i, start:end]
            seg = resize(raw_seg, (window_len,), anti_aliasing=True)
            baseline = scipy.signal.medfilt(seg, kernel_size=baseline_kernel)
            beats_data[lead_i, i] = seg - baseline

    heartbeats = LeadArray(beats_data, lead_order=ecg_signals.lead_order)

    labels = None
    if beat_symbols is not None:
        symbols_trimmed = np.asarray(beat_symbols)[valid_idx]
        aami_labels = np.array([AAMI_MAP.get(sym, 5) for sym in symbols_trimmed])
        labels = {'symbols': symbols_trimmed, 'AAMI': aami_labels}

    if verbose:
        print(f"Extracted heartbeats shape: {heartbeats.shape}")
        if labels is not None:
            print(f"  Symbols shape: {labels['symbols'].shape}")
            print(f"  AAMI labels shape: {labels['AAMI'].shape}")

    return heartbeats, labels


# ---------------------------------------------------------------------------
# 4. Embeddings (UMAP / t-SNE / PCA)
# ---------------------------------------------------------------------------

def compute_embeddings(modality, standardize=False, min_dist=0.1, n_neighbors=15,
                        random_state=42, perplexity=30.0, verbose=True):
    """
    Compute UMAP, t-SNE, and PCA 2D embeddings for a single modality
    (e.g. one lead's heartbeats, or one feature representation).

    Standardization (zero mean, unit variance per feature/column, via
    sklearn's StandardScaler) before UMAP/t-SNE/PCA is OPTIONAL and OFF by
    default. All three methods rely on distances/variance in feature space,
    so turning this on puts every sample on a comparable scale, which can
    keep a handful of high-amplitude samples (e.g. a QRS peak) from
    dominating the embedding purely because of scale rather than shape --
    but it also discards the original amplitude information, which may
    matter for ECG beats. Enable it explicitly if you want that behavior.

    Call this once per modality/lead; to compare multiple leads, call it
    in a loop from the notebook/script and collect the results yourself.

    Args:
        modality (np.ndarray): shape (num_samples, num_features). Must not
            contain NaNs; drop/impute rows with missing leads before calling.
        standardize (bool): If True, z-score each feature/column with
            StandardScaler before running UMAP/t-SNE/PCA. Default is False
            (embeddings run directly on the raw beat amplitudes).
        min_dist (float): UMAP min_dist parameter.
        n_neighbors (int): UMAP n_neighbors parameter.
        random_state (int): Random seed shared across UMAP / t-SNE / PCA.
        perplexity (float): t-SNE perplexity parameter.
        verbose (bool): Print a short summary.

    Returns:
        dict: {'umap', 'tsne', 'pca'}, each of shape (num_samples, 2).
    """

    if np.any(np.isnan(modality)):
        raise ValueError(
            "modality contains NaN values. This typically happens when a lead "
            "isn't available for every record. Filter/impute rows first, "
            "e.g. modality[~np.isnan(modality).any(axis=1)]."
        )

    if standardize:
        scaler = StandardScaler()
        modality = scaler.fit_transform(modality)

    ump = UMAP(n_components=2, min_dist=min_dist, n_neighbors=n_neighbors,
               random_state=random_state, init='pca')
    tsne = TSNE(n_components=2, perplexity=perplexity,
                random_state=random_state, init='pca')
    pca = PCA(n_components=2, random_state=random_state)

    embeddings = {
        'umap': ump.fit_transform(modality),
        'tsne': tsne.fit_transform(modality),
        'pca': pca.fit_transform(modality),
    }

    if verbose:
        print(f"Standardization: {'on' if standardize else 'off'}")
        for key, value in embeddings.items():
            print(f"{key}: shape {value.shape}")

    return embeddings


# ---------------------------------------------------------------------------
# 5. Saving / loading full pipeline output
# ---------------------------------------------------------------------------

def save_results(filepath, ecg_signals, heartbeats, embeddings_per_lead,
                  aami_labels, metadata, fs, r_peaks=None, verbose=True):
    """
    Save the full pipeline output (signal, heartbeats, embeddings, AAMI
    labels, metadata, sampling frequency, R-peaks) into a single .npz file.

    Embeddings are stored in the same (12, N, ...) layout as signal/heartbeats:
    for each method ('umap', 'tsne', 'pca') a single array of shape
    (12, N, 2) is saved, where N is the number of heartbeats the embeddings
    were computed on. Leads that embeddings weren't computed for are filled
    with NaN, exactly like unused leads in signal_data/heartbeats_data.

    Args:
        filepath (str): Output path, e.g. "record_100.npz".
        ecg_signals (LeadArray): shape (12, T), from load_ecg_signal.
        heartbeats (LeadArray): shape (12, N, window_len), from extract_heartbeats.
        embeddings_per_lead (dict): {lead_name: {'umap', 'tsne', 'pca'}}, one
            entry per lead that compute_embeddings was called on. All leads
            must have embeddings of the same N (e.g. computed on beats[valid_mask]
            with a shared valid_mask across leads).
        aami_labels (np.ndarray or None): AAMI superclass label per heartbeat,
            aligned with heartbeats (labels['AAMI'] from extract_heartbeats).
        metadata (dict): Record-level metadata, from load_ecg_signal.
        fs (float): Sampling frequency in Hz.
        r_peaks (np.ndarray, optional): R-peak sample index per heartbeat,
            aligned with heartbeats (i.e. r_peaks[1:-1] from load_or_compute_rpeaks).
        verbose (bool): Print a short summary.

    Returns:
        str: The filepath that was written to.
    """

    lead_order = ecg_signals.lead_order
    lead_idx = {lead: i for i, lead in enumerate(lead_order)}
    methods = ('umap', 'tsne', 'pca')

    save_dict = {
        'signal_data': ecg_signals.data,
        'signal_lead_order': np.array(ecg_signals.lead_order),
        'heartbeats_data': heartbeats.data,
        'heartbeats_lead_order': np.array(heartbeats.lead_order),
        'aami_labels': aami_labels if aami_labels is not None else np.array([]),
        'r_peaks': np.asarray(r_peaks) if r_peaks is not None else np.array([]),
        'metadata': np.array(metadata, dtype=object),
        'fs': np.array(fs),
        'embeddings_lead_order': np.array(lead_order),
    }

    if embeddings_per_lead:
        n_embed = next(iter(embeddings_per_lead.values()))['umap'].shape[0]
        for method in methods:
            emb_arr = np.full((len(lead_order), n_embed, 2), np.nan)
            for lead_name, lead_embeddings in embeddings_per_lead.items():
                emb_arr[lead_idx[lead_name]] = lead_embeddings[method]
            save_dict[f'embeddings_{method}'] = emb_arr

    np.savez(filepath, **save_dict)

    if verbose:
        print(f"Saved pipeline results to: {filepath}")
        for key, value in save_dict.items():
            shape = getattr(value, 'shape', None)
            print(f"  {key}: shape {shape}" if shape is not None else f"  {key}")

    return filepath


def load_results(filepath, verbose=True):
    """
    Load a .npz file written by save_results back into usable objects.

    Args:
        filepath (str): Path to the .npz file.
        verbose (bool): Print a short summary.

    Returns:
        dict: {
            'ecg_signals'  : LeadArray, shape (12, T),
            'heartbeats'   : LeadArray, shape (12, N, window_len),
            'aami_labels'  : np.ndarray or None,
            'r_peaks'      : np.ndarray or None, aligned with heartbeats,
            'embeddings'   : dict {'umap', 'tsne', 'pca'} -> LeadArray, shape (12, N, 2)
                              (missing leads are NaN). Access e.g.
                              embeddings['umap']['II'] -> shape (N, 2).
            'metadata'     : dict,
            'fs'           : float,
        }
    """

    npz = np.load(filepath, allow_pickle=True)

    ecg_signals = LeadArray(npz['signal_data'], lead_order=list(npz['signal_lead_order']))
    heartbeats = LeadArray(npz['heartbeats_data'], lead_order=list(npz['heartbeats_lead_order']))

    aami_labels = npz['aami_labels']
    aami_labels = None if aami_labels.size == 0 else aami_labels

    r_peaks = npz['r_peaks'] if 'r_peaks' in npz.files else np.array([])
    r_peaks = None if r_peaks.size == 0 else r_peaks

    metadata = npz['metadata'].item()
    fs = npz['fs'].item()

    embeddings_lead_order = list(npz['embeddings_lead_order'])
    embeddings = {}
    for method in ('umap', 'tsne', 'pca'):
        key = f'embeddings_{method}'
        if key in npz.files:
            embeddings[method] = LeadArray(npz[key], lead_order=embeddings_lead_order)

    if verbose:
        print(f"Loaded pipeline results from: {filepath}")
        print(f"  Signal shape    : {ecg_signals.shape}")
        print(f"  Heartbeats shape: {heartbeats.shape}")
        if embeddings:
            print(f"  Embeddings shape: {next(iter(embeddings.values())).shape} "
                  f"(methods: {list(embeddings.keys())})")

    return {
        'ecg_signals': ecg_signals,
        'heartbeats': heartbeats,
        'aami_labels': aami_labels,
        'r_peaks': r_peaks,
        'embeddings': embeddings,
        'metadata': metadata,
        'fs': fs,
    }


# ---------------------------------------------------------------------------
# 6. End-to-end pipeline
# ---------------------------------------------------------------------------

def main(record_number, record_path_base, output_dir=None,
         leads_for_embeddings=None, window_len=256,
         detection_lead="II", standardize=False, plot=True, verbose=True):
    """
    Run the full ECG pipeline for a single record:
        1. load the raw signal
        2. load (or compute) R-peaks
        3. extract heartbeats (+ AAMI labels, if annotations are available)
        4. compute UMAP/t-SNE/PCA embeddings, one lead at a time (features
           are standardized before each method by default, see compute_embeddings)
        5. optionally save everything to a single .npz file

    Args:
        record_number (int): MIT-BIH record number.
        record_path_base (str): Directory containing the record files.
        output_dir (str, optional): If given, results are saved to
            f"{output_dir}/record_{record_number}.npz".
        leads_for_embeddings (tuple, optional): Which leads to compute
            embeddings for. Default is None, which auto-detects the leads
            actually present in this record (via ecg_signals.available_leads)
            and uses all of them -- e.g. ('II', 'V1') for a record with
            MLII+V1, ('II', 'V5') for a record with MLII+V5. Pass an explicit
            tuple to override (e.g. to embed only one lead, or a specific
            subset of a true 12-lead record).
        window_len (int): Heartbeat window length after resampling.
        detection_lead (str): Lead used for NeuroKit2 R-peak detection if no
            annotation file is found.
        standardize (bool): Passed through to compute_embeddings. Default is
            False (embeddings run on raw beat amplitudes). Set True to
            z-score each lead's beats before UMAP/t-SNE/PCA.
        plot (bool): If True, show a few sanity-check plots (requires matplotlib).
        verbose (bool): Print progress from each pipeline step.

    Returns:
        dict: {
            'ecg_signals'  : LeadArray, shape (12, T),
            'heartbeats'   : LeadArray, shape (12, N, window_len),
            'labels'       : dict {'symbols', 'AAMI'} or None,
            'r_peaks_heartbeats' : np.ndarray, shape (N,), R-peak sample index
                                    per heartbeat (aligned with heartbeats/labels).
            'embeddings'   : dict {lead_name: {'umap', 'tsne', 'pca'}},
            'embedding_aami_labels' : np.ndarray or None, aligned with embeddings,
            'embedding_r_peaks' : np.ndarray or None, aligned with embeddings,
            'embedding_valid_mask' : np.ndarray, boolean mask into heartbeats
                                      selecting the rows used for embeddings,
            'metadata'     : dict,
            'fs'           : float,
            'output_path'  : str or None,
        }
    """

    # 1. Load raw signal
    ecg_signals, fs, metadata = load_ecg_signal(
        record_number, record_path_base, verbose=verbose
    )

    # Auto-detect which leads to embed if the caller didn't specify any.
    # MIT-BIH records vary in their second lead (V1, V5, V2, ...), so a
    # fixed default like ("II", "V1") silently breaks on any record that
    # isn't II+V1. available_leads is already in STANDARD_12_LEAD_ORDER,
    # so e.g. II sorts before V1/V5/etc. automatically.
    if leads_for_embeddings is None:
        leads_for_embeddings = tuple(ecg_signals.available_leads)
        if verbose:
            print(f"leads_for_embeddings not specified -> using this record's "
                  f"available leads: {leads_for_embeddings}")

    # 2. Load or compute R-peaks
    r_peaks, beat_symbols, source = load_or_compute_rpeaks(
        ecg_signals, fs,
        record_name=record_number,
        record_path_base=record_path_base,
        detection_lead=detection_lead,
        verbose=verbose,
    )

    # 3. Extract heartbeats (+ AAMI labels if annotations were found)
    heartbeats, labels = extract_heartbeats(
        ecg_signals, r_peaks,
        window_len=window_len,
        beat_symbols=beat_symbols,
        verbose=verbose,
    )
    aami_labels = labels["AAMI"] if labels is not None else None

    # R-peak sample index per heartbeat, aligned with heartbeats/labels
    # (extract_heartbeats drops the first/last R-peak, see its docstring).
    r_peaks_heartbeats = np.asarray(r_peaks)[1:-1]

    # 4. Compute embeddings, one lead at a time
    #    Rows with NaN in ANY of the requested leads are dropped so the
    #    per-lead embeddings stay row-aligned with each other.
    #
    #    Fail fast if a requested lead isn't physically present in THIS
    #    record: leads_for_embeddings has a fixed default (e.g. "V1"), but
    #    not every MIT-BIH record has the same second lead (e.g. record 100
    #    has MLII+V5, not MLII+V1). Silently proceeding would make every row
    #    NaN for that lead, collapsing valid_mask to all-False and 0 beats,
    #    which previously surfaced as a confusing StandardScaler error deep
    #    in compute_embeddings instead of this clear one.
    available = ecg_signals.available_leads
    missing = [lead for lead in leads_for_embeddings if lead not in available]
    if missing:
        raise ValueError(
            f"leads_for_embeddings {leads_for_embeddings} requested lead(s) "
            f"{missing} not present in record {record_number} "
            f"(available leads: {available}). Pass a leads_for_embeddings "
            f"tuple using only available leads, e.g. "
            f"leads_for_embeddings=('II', '{available[-1] if available else '...'}')."
        )

    lead_beats = {lead: heartbeats[lead] for lead in leads_for_embeddings}
    valid_mask = np.ones(heartbeats.shape[1], dtype=bool)
    for beats in lead_beats.values():
        valid_mask &= ~np.isnan(beats).any(axis=1)

    if not valid_mask.any():
        raise ValueError(
            f"0 heartbeats have non-NaN data for all of leads_for_embeddings="
            f"{leads_for_embeddings} in record {record_number}, so there's "
            f"nothing to embed. This usually means segmentation produced "
            f"degenerate (NaN) beats for one of these leads -- check the "
            f"heartbeats shape/NaN count printed above."
        )

    embeddings_per_lead = {
        lead: compute_embeddings(beats[valid_mask], standardize=standardize, verbose=verbose)
        for lead, beats in lead_beats.items()
    }
    embedding_aami_labels = aami_labels[valid_mask] if aami_labels is not None else None
    embedding_r_peaks = r_peaks_heartbeats[valid_mask]

    # 5. Optionally save everything to a single .npz file
    output_path = None
    if output_dir is not None:
        output_path = f"{output_dir}/record_{record_number}.npz"
        save_results(
            output_path, ecg_signals, heartbeats, embeddings_per_lead,
            aami_labels, metadata, fs, r_peaks=r_peaks_heartbeats, verbose=verbose,
        )

    if plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 1 + len(leads_for_embeddings), figsize=(15, 3))
        axes[0].plot(ecg_signals[detection_lead][:3000])
        axes[0].set_title(f"Record {record_number} - Lead {detection_lead}")
        for ax, lead in zip(axes[1:], leads_for_embeddings):
            emb = embeddings_per_lead[lead]["umap"]
            if embedding_aami_labels is not None:
                ax.scatter(emb[:, 0], emb[:, 1], c=embedding_aami_labels, cmap="tab10", s=5)
            else:
                ax.scatter(emb[:, 0], emb[:, 1], s=5)
            ax.set_title(f"UMAP - Lead {lead}")
        plt.tight_layout()
        plt.show()

    return {
        "ecg_signals": ecg_signals,
        "heartbeats": heartbeats,
        "labels": labels,
        "r_peaks_heartbeats": r_peaks_heartbeats,
        "embeddings": embeddings_per_lead,
        "embedding_aami_labels": embedding_aami_labels,
        "embedding_r_peaks": embedding_r_peaks,
        "embedding_valid_mask": valid_mask,
        "metadata": metadata,
        "fs": fs,
        "output_path": output_path,
    }


# ---------------------------------------------------------------------------
# 7. Export for the ECG Atlas Visualization HTML
# ---------------------------------------------------------------------------

def export_atlas_json(results, output_path, leads=None, verbose=True):
    """
    Convert a main() results dict into the JSON schema expected by the
    ECG Atlas Visualization HTML (dict-of-lead access, mirroring LeadArray):

        {
          "fs": <float>,
          "available_leads": [...],
          "lead_display_names": {lead: "Lead II (from MLII)", ...},
          "signal_data": {lead: [T floats], ...},
          "heartbeats_data": {lead: [[window_len floats], ...], ...},
          "r_peaks": [N ints],           # aligned with heartbeats_data/aami_labels
          "aami_labels": [N ints],
          "embeddings": {"umap": {lead: [[x, y], ...]}, "tsne": {...}, "pca": {...}},
          "metadata": {...}
        }

    Only the leads in `leads` (default: whichever leads embeddings were
    computed for) are included, so file size stays proportional to the
    number of real leads rather than the full 12-lead layout.

    Args:
        results (dict): Output of main(...).
        output_path (str): Where to write the .json file, e.g. "ecg_data.json".
        leads (list, optional): Which leads to include. Defaults to the keys
            of results['embeddings'] (i.e. leads_for_embeddings passed to main()).
        verbose (bool): Print a short summary.

    Returns:
        str: The output_path that was written to.
    """
    import json

    ecg_signals = results["ecg_signals"]
    heartbeats = results["heartbeats"]
    embeddings_per_lead = results["embeddings"]
    aami_labels = results["embedding_aami_labels"]
    r_peaks = results["embedding_r_peaks"]
    valid_mask = results["embedding_valid_mask"]
    metadata = results["metadata"]
    fs = results["fs"]

    leads = list(leads) if leads is not None else list(embeddings_per_lead.keys())

    matched_leads = metadata.get("matched_leads", {}) or {}
    lead_display_names = {}
    for lead in leads:
        raw_name = next((raw for raw, canon in matched_leads.items() if canon == lead), None)
        lead_display_names[lead] = f"{lead} (from {raw_name})" if raw_name and raw_name != lead else lead

    payload = {
        "fs": float(fs),
        "available_leads": leads,
        "lead_display_names": lead_display_names,
        "signal_data": {lead: ecg_signals[lead].tolist() for lead in leads},
        "heartbeats_data": {lead: heartbeats[lead][valid_mask].tolist() for lead in leads},
        "r_peaks": np.asarray(r_peaks).tolist(),
        "aami_labels": np.asarray(aami_labels).tolist() if aami_labels is not None else [],
        "embeddings": {
            method: {lead: embeddings_per_lead[lead][method].tolist() for lead in leads}
            for method in ("umap", "tsne", "pca")
        },
        "metadata": {k: v for k, v in metadata.items() if k != "comments"},
    }

    with open(output_path, "w") as f:
        json.dump(payload, f)

    if verbose:
        n_beats = len(payload["r_peaks"])
        print(f"Exported atlas JSON to: {output_path}")
        print(f"  Leads    : {leads}")
        print(f"  N beats  : {n_beats}")

    return output_path
