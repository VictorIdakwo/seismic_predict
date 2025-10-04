import os
import glob
import math
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from obspy import read, Stream, UTCDateTime
from obspy.signal.filter import bandpass
from obspy.signal.filter import envelope
from obspy.taup import TauPyModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------
# Config
# ---------------------------
DATA_DIR = r"C:\Users\victor.idakwo\Documents\ehealth Africa\ehealth Africa\eHA GitHub\seismic Analytics\Jan"
STATION_FILE = r"C:\Users\victor.idakwo\Documents\ehealth Africa\ehealth Africa\eHA GitHub\seismic Analytics\stations.csv"
TRAINING_CSV = r"C:\Users\victor.idakwo\Documents\ehealth Africa\ehealth Africa\eHA GitHub\seismic Analytics\training_events.csv"
MODEL_OUT = r"magnitude_model.joblib"
MODEL_FEATURES_OUT = r"magnitude_features.csv"    # optional: saved features for inspection

# TauP model for distance estimation
taup_model = TauPyModel(model="iasp91")

# ---------------------------
# Helpers: station metadata
# ---------------------------
stations_df = pd.read_csv(STATION_FILE)
stations_df.columns = (
    stations_df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("/", "_")
)

def station_coords(station_code):
    row = stations_df[stations_df["station"] == station_code]
    if row.empty:
        return None
    return float(row["latitude"].values[0]), float(row["longitude"].values[0])

# ---------------------------
# Low-level feature extraction
# ---------------------------
def compute_dominant_frequency(data, sampling_rate):
    """Estimate dominant frequency via FFT peak in power spectrum."""
    n = len(data)
    if n == 0:
        return np.nan
    # window and detrend
    data = data - np.mean(data)
    freqs = np.fft.rfftfreq(n, d=1.0/sampling_rate)
    ps = np.abs(np.fft.rfft(data))**2
    idx = np.argmax(ps)
    return float(freqs[idx]) if idx < len(freqs) else np.nan

def compute_snr(tr, pick_time, pre_sec=2.0, win_sec=2.0):
    """Estimate simple SNR as ratio of RMS(signal) / RMS(noise)."""
    sr = tr.stats.sampling_rate
    start_noise = pick_time - pre_sec
    end_noise = pick_time - 0.5  # keep a gap before pick
    start_sig = pick_time
    end_sig = pick_time + win_sec

    try:
        noise = tr.slice(starttime=start_noise, endtime=end_noise).data
        sig = tr.slice(starttime=start_sig, endtime=end_sig).data
    except Exception:
        return np.nan

    if noise.size == 0 or sig.size == 0:
        return np.nan

    rms_noise = np.sqrt(np.mean(noise**2))
    rms_sig = np.sqrt(np.mean(sig**2))
    if rms_noise == 0:
        return 1000.0  # Cap at a large but finite value instead of infinity
    snr = rms_sig / rms_noise
    return float(min(snr, 1000.0))  # Cap maximum SNR at 1000

def extract_features_from_trace(tr, pick_time, event_origin_time, event_lat=None, event_lon=None):
    """Given a Trace, compute features around pick_time."""
    sr = tr.stats.sampling_rate
    # pick window: -1 to +5 seconds by default (adjustable)
    t0 = pick_time - 1.0
    t1 = pick_time + 5.0
    try:
        seg = tr.slice(starttime=t0, endtime=t1)
        data = seg.data.astype(float)
    except Exception:
        return None

    # basic amplitude features
    peak_abs = float(np.max(np.abs(data))) if data.size > 0 else np.nan
    rms = float(np.sqrt(np.mean(data**2))) if data.size > 0 else np.nan
    envelope_peak = float(np.max(envelope(data))) if data.size > 0 else np.nan
    dominant_freq = compute_dominant_frequency(data, sr)

    # SNR
    snr = compute_snr(tr, pick_time)

    # duration: time index when envelope decays below e.g., 10% of peak
    try:
        env = envelope(data)
        peak_idx = np.argmax(env)
        thresh = 0.1 * env[peak_idx]
        # search after peak
        post = env[peak_idx:]
        below = np.where(post < thresh)[0]
        if below.size > 0:
            dur = float(below[0] / sr)
        else:
            dur = float(len(env) / sr)
    except Exception:
        dur = np.nan

    # distance & distance-corrected amplitude (if event lat/lon provided)
    distance_km = np.nan
    amp_corr = np.nan
    if event_lat is not None and event_lon is not None:
        coords = station_coords(tr.stats.station)
        if coords is not None:
            sta_lat, sta_lon = coords
            # approximate great-circle distance (Haversine)
            R = 6371.0
            lat1 = math.radians(event_lat); lon1 = math.radians(event_lon)
            lat2 = math.radians(sta_lat); lon2 = math.radians(sta_lon)
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat/2.0)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2.0)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            distance_km = R * c
            # simple geometric spreading correction: amp * distance (or * distance**p)
            if not np.isnan(distance_km) and distance_km > 0:
                amp_corr = peak_abs * distance_km  # you can try distance**1.0 or distance**0.5

    features = {
        "station": tr.stats.station,
        "peak_abs": peak_abs,
        "rms": rms,
        "envelope_peak": envelope_peak,
        "dominant_freq": dominant_freq,
        "snr": snr,
        "duration": dur,
        "distance_km": distance_km,
        "amp_distance_corr": amp_corr,
        "sampling_rate": sr
    }
    return features

# ---------------------------
# Aggregate station-level features into event-level feature vector
# ---------------------------
def aggregate_features_for_event(st_trim, picks, origin_time, event_lat=None, event_lon=None):
    """
    st_trim: Stream trimmed to window containing event (as in your pipeline)
    picks: list of tuples (station, pick_time)
    origin_time: UTCDateTime
    """
    station_features = []
    for station, pick_time in picks:
        # find trace for station (prefer vertical? here we pick first matching trace)
        tr_candidates = [tr for tr in st_trim if tr.stats.station == station]
        if not tr_candidates:
            continue
        # pick best trace (first) - you can choose component Z if available by tr.stats.channel
        tr = tr_candidates[0]
        feat = extract_features_from_trace(tr, pick_time, origin_time, event_lat, event_lon)
        if feat:
            station_features.append(feat)

    if len(station_features) == 0:
        return None

    df = pd.DataFrame(station_features)

    # aggregate: median, mean, count, std
    agg = {}
    numeric_cols = ["peak_abs", "rms", "envelope_peak", "dominant_freq", "snr", "duration", "distance_km", "amp_distance_corr"]
    for col in numeric_cols:
        agg[f"{col}_median"] = df[col].median(skipna=True)
        agg[f"{col}_mean"] = df[col].mean(skipna=True)
        agg[f"{col}_std"] = df[col].std(skipna=True)
    agg["n_stations"] = len(df)
    return agg

# ---------------------------
# Build training matrix from labeled events
# ---------------------------
def build_training_dataframe(mseed_dir, catalog_events_df):
    """
    catalog_events_df: DataFrame with columns ['event_id', 'origin_time' (ISO), 'magnitude']
    Returns: X (DataFrame of features), y (Series of magnitudes)
    """
    X_rows = []
    y = []

    # load streams once (map station->list of traces)
    files = glob.glob(os.path.join(mseed_dir, "*.mseed"))
    # read all into one stream (be careful with memory for large datasets)
    streams = [read(f) for f in files]
    all_traces = sum(streams, start=Stream())
    all_traces = all_traces.copy()

    for _, row in tqdm(catalog_events_df.iterrows(), total=len(catalog_events_df), desc="features per training event"):
        origin_time = UTCDateTime(row["origin_time"])
        # trim a window around origin to look for picks (this simple approach uses STA/LTA as before to find station picks)
        window_start = origin_time - 10
        window_end = origin_time + 60
        st_trim = all_traces.copy().trim(starttime=window_start, endtime=window_end, pad=True, fill_value=0)

        # re-run STA/LTA quickly to find pick times per trace (or use available picks if you have them)
        picks = []
        for tr in st_trim:
            try:
                from obspy.signal.trigger import classic_sta_lta, trigger_onset
                cft = classic_sta_lta(tr.data, int(2*tr.stats.sampling_rate), int(10*tr.stats.sampling_rate))
                on_of = trigger_onset(cft, 2.5, 1.0)
                if len(on_of) > 0:
                    pick_time = tr.stats.starttime + on_of[0][0] / tr.stats.sampling_rate
                    # only accept picks close to origin_time (adjust tolerance)
                    if abs(pick_time - origin_time) < 60:
                        picks.append((tr.stats.station, pick_time))
            except Exception:
                continue

        if len(picks) < 1:
            # skip events without picks
            continue

        agg = aggregate_features_for_event(st_trim, picks, origin_time, event_lat=None, event_lon=None)
        if agg is None:
            continue

        X_rows.append(agg)
        y.append(row["magnitude"])

    X = pd.DataFrame(X_rows)
    y = pd.Series(y, name="magnitude")
    # optionally save features
    X.to_csv(MODEL_FEATURES_OUT, index=False)
    return X, y

# ---------------------------
# Train model
# ---------------------------
def train_model(X, y, model_out=MODEL_OUT):
    # Handle infinities and NaNs
    X = X.replace([np.inf, -np.inf], np.nan)  # Replace infinities with NaN
    X = X.fillna(-999)  # simple imputation; tune as needed
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # evaluate
    y_pred = model.predict(X_test)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"Model RMSE: {rmse:.3f}, R2: {r2:.3f}")

    # persist
    joblib.dump(model, model_out)
    print(f"Model saved to: {model_out}")
    return model

# ---------------------------
# Predict magnitudes for catalogue produced earlier
# ---------------------------
def predict_magnitudes_for_catalog(catalog, mseed_dir, model_path=MODEL_OUT):
    """
    catalog: obspy Catalog with Event.origins available
    Reads model from disk, extracts features for each event, predicts magnitude and writes to QuakeML
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Train first or supply model_path.")

    model = joblib.load(model_path)
    # read all traces
    files = glob.glob(os.path.join(mseed_dir, "*.mseed"))
    streams = [read(f) for f in files]
    all_traces = sum(streams, start=Stream())

    event_preds = []
    for ev in catalog:
        try:
            origin = ev.preferred_origin() or ev.origins[0]
            origin_time = origin.time
            event_lat = getattr(origin, "latitude", None)
            event_lon = getattr(origin, "longitude", None)
        except Exception:
            continue

        # window around event (same as in build)
        st_trim = all_traces.copy().trim(starttime=origin_time - 10, endtime=origin_time + 60, pad=True, fill_value=0)
        # find picks per station (quick STA/LTA)
        picks = []
        for tr in st_trim:
            try:
                from obspy.signal.trigger import classic_sta_lta, trigger_onset
                cft = classic_sta_lta(tr.data, int(2*tr.stats.sampling_rate), int(10*tr.stats.sampling_rate))
                on_of = trigger_onset(cft, 2.5, 1.0)
                if len(on_of) > 0:
                    pick_time = tr.stats.starttime + on_of[0][0] / tr.stats.sampling_rate
                    if pick_time >= origin_time - 10 and pick_time <= origin_time + 60:
                        picks.append((tr.stats.station, pick_time))
            except Exception:
                continue

        if len(picks) == 0:
            # skip
            continue

        agg = aggregate_features_for_event(st_trim, picks, origin_time, event_lat, event_lon)
        if agg is None:
            continue

        X_row = pd.DataFrame([agg]).fillna(-999)
        pred_mag = model.predict(X_row)[0]
        # store predicted magnitude into event magnitude element
        from obspy.core.event import Magnitude, MagnitudeType
        mag_obj = Magnitude(mag=float(pred_mag), magnitude_type="ML")  # or "Mw" depending on your label
        ev.magnitudes.append(mag_obj)
        event_preds.append((ev, pred_mag))

    return catalog, event_preds

# ---------------------------
# Example: training flow
# ---------------------------
if __name__ == "__main__":
    # If you have a training csv, build dataset and train
    if os.path.exists(TRAINING_CSV):
        train_df = pd.read_csv(TRAINING_CSV)
        X, y = build_training_dataframe(DATA_DIR, train_df)
        if len(X) > 0:
            model = train_model(X, y)
        else:
            print("No training features extracted. Check your training CSV and waveforms.")
    else:
        print("No TRAINING_CSV found - skipping training step. Provide one to train a model.")

    # Example usage after your catalog is built (you can import this script and call predict_magnitudes_for_catalog)
    # from obspy import read_events
    # catalog = read_events("ml_detected_events.xml")
    # catalog_with_mags, preds = predict_magnitudes_for_catalog(catalog, DATA_DIR)
    # catalog_with_mags.write("ml_detected_events_with_mags.xml", format="QUAKEML")
