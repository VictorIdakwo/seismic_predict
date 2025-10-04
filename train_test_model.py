import os
import glob
import math
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from obspy import read, Stream, UTCDateTime, read_events
from obspy.signal.filter import envelope
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# ---------------------------
# Config
# ---------------------------
DATA_DIR = r"C:\Users\victor.idakwo\Documents\ehealth Africa\ehealth Africa\eHA GitHub\seismic Analytics\Jan"
STATION_FILE = r"C:\Users\victor.idakwo\Documents\ehealth Africa\ehealth Africa\eHA GitHub\seismic Analytics\stations.csv"
TRAINING_CSV = r"C:\Users\victor.idakwo\Documents\ehealth Africa\ehealth Africa\eHA GitHub\seismic Analytics\training_events.csv"
DETECTED_EVENTS = r"C:\Users\victor.idakwo\Documents\ehealth Africa\ehealth Africa\eHA GitHub\seismic Analytics\ml_detected_events.xml"

MODEL_OUT = r"C:\Users\victor.idakwo\Documents\ehealth Africa\ehealth Africa\eHA GitHub\seismic Analytics\magnitude_model.joblib"
MODEL_FEATURES_OUT = r"C:\Users\victor.idakwo\Documents\ehealth Africa\ehealth Africa\eHA GitHub\seismic Analytics\magnitude_features.csv"
PREDICTIONS_OUT = r"C:\Users\victor.idakwo\Documents\ehealth Africa\ehealth Africa\eHA GitHub\seismic Analytics\predictions.csv"
CATALOG_WITH_MAGS = r"C:\Users\victor.idakwo\Documents\ehealth Africa\ehealth Africa\eHA GitHub\seismic Analytics\ml_detected_events_with_mags.xml"

# ---------------------------
# Station metadata
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
# Feature extraction functions
# ---------------------------
def compute_dominant_frequency(data, sampling_rate):
    """Estimate dominant frequency via FFT peak in power spectrum."""
    n = len(data)
    if n == 0:
        return np.nan
    data = data - np.mean(data)
    freqs = np.fft.rfftfreq(n, d=1.0/sampling_rate)
    ps = np.abs(np.fft.rfft(data))**2
    idx = np.argmax(ps)
    return float(freqs[idx]) if idx < len(freqs) else np.nan

def compute_snr(tr, pick_time, pre_sec=2.0, win_sec=2.0):
    """Estimate simple SNR as ratio of RMS(signal) / RMS(noise)."""
    start_noise = pick_time - pre_sec
    end_noise = pick_time - 0.5
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
        return np.inf
    return float(rms_sig / rms_noise)

def extract_features_from_trace(tr, pick_time, event_origin_time, event_lat=None, event_lon=None):
    """Given a Trace, compute features around pick_time."""
    sr = tr.stats.sampling_rate
    t0 = pick_time - 1.0
    t1 = pick_time + 5.0
    try:
        seg = tr.slice(starttime=t0, endtime=t1)
        data = seg.data.astype(float)
    except Exception:
        return None

    # Basic amplitude features
    peak_abs = float(np.max(np.abs(data))) if data.size > 0 else np.nan
    rms = float(np.sqrt(np.mean(data**2))) if data.size > 0 else np.nan
    envelope_peak = float(np.max(envelope(data))) if data.size > 0 else np.nan
    dominant_freq = compute_dominant_frequency(data, sr)

    # SNR
    snr = compute_snr(tr, pick_time)

    # Duration: time when envelope decays below 10% of peak
    try:
        env = envelope(data)
        peak_idx = np.argmax(env)
        thresh = 0.1 * env[peak_idx]
        post = env[peak_idx:]
        below = np.where(post < thresh)[0]
        if below.size > 0:
            dur = float(below[0] / sr)
        else:
            dur = float(len(env) / sr)
    except Exception:
        dur = np.nan

    # Distance & distance-corrected amplitude
    distance_km = np.nan
    amp_corr = np.nan
    if event_lat is not None and event_lon is not None:
        coords = station_coords(tr.stats.station)
        if coords is not None:
            sta_lat, sta_lon = coords
            # Haversine distance
            R = 6371.0
            lat1 = math.radians(event_lat); lon1 = math.radians(event_lon)
            lat2 = math.radians(sta_lat); lon2 = math.radians(sta_lon)
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat/2.0)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2.0)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            distance_km = R * c
            if not np.isnan(distance_km) and distance_km > 0:
                amp_corr = peak_abs * distance_km

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

def aggregate_features_for_event(st_trim, picks, origin_time, event_lat=None, event_lon=None):
    """Aggregate station-level features into event-level feature vector."""
    station_features = []
    for station, pick_time in picks:
        tr_candidates = [tr for tr in st_trim if tr.stats.station == station]
        if not tr_candidates:
            continue
        tr = tr_candidates[0]
        feat = extract_features_from_trace(tr, pick_time, origin_time, event_lat, event_lon)
        if feat:
            station_features.append(feat)

    if len(station_features) == 0:
        return None

    df = pd.DataFrame(station_features)

    # Aggregate: median, mean, std
    agg = {}
    numeric_cols = ["peak_abs", "rms", "envelope_peak", "dominant_freq", "snr", "duration", "distance_km", "amp_distance_corr"]
    for col in numeric_cols:
        agg[f"{col}_median"] = df[col].median(skipna=True)
        agg[f"{col}_mean"] = df[col].mean(skipna=True)
        agg[f"{col}_std"] = df[col].std(skipna=True)
    agg["n_stations"] = len(df)
    return agg

# ---------------------------
# Build training data
# ---------------------------
def build_training_dataframe(mseed_dir, catalog_events_df):
    """
    Build training features from labeled events.
    catalog_events_df: DataFrame with columns ['event_id', 'origin_time' (ISO), 'magnitude']
    Returns: X (DataFrame of features), y (Series of magnitudes)
    """
    X_rows = []
    y = []

    # Load all waveforms
    print("Loading waveform files...")
    files = glob.glob(os.path.join(mseed_dir, "*.mseed"))
    streams = [read(f) for f in tqdm(files, desc="Reading mseed files")]
    all_traces = sum(streams, start=Stream())
    all_traces = all_traces.copy()

    print(f"\nProcessing {len(catalog_events_df)} training events...")
    for _, row in tqdm(catalog_events_df.iterrows(), total=len(catalog_events_df), desc="Extracting features"):
        origin_time = UTCDateTime(row["origin_time"])
        event_lat = row.get("latitude", None)
        event_lon = row.get("longitude", None)
        
        # Trim window around origin
        window_start = origin_time - 10
        window_end = origin_time + 60
        st_trim = all_traces.copy().trim(starttime=window_start, endtime=window_end, pad=True, fill_value=0)

        # Find picks using STA/LTA
        picks = []
        for tr in st_trim:
            try:
                cft = classic_sta_lta(tr.data, int(2*tr.stats.sampling_rate), int(10*tr.stats.sampling_rate))
                on_of = trigger_onset(cft, 2.5, 1.0)
                if len(on_of) > 0:
                    pick_time = tr.stats.starttime + on_of[0][0] / tr.stats.sampling_rate
                    if abs(pick_time - origin_time) < 60:
                        picks.append((tr.stats.station, pick_time))
            except Exception:
                continue

        if len(picks) < 1:
            continue

        agg = aggregate_features_for_event(st_trim, picks, origin_time, event_lat, event_lon)
        if agg is None:
            continue

        X_rows.append(agg)
        y.append(row["magnitude"])

    X = pd.DataFrame(X_rows)
    y = pd.Series(y, name="magnitude")
    
    # Save features
    X.to_csv(MODEL_FEATURES_OUT, index=False)
    print(f"\nFeatures saved to: {MODEL_FEATURES_OUT}")
    
    return X, y

# ---------------------------
# Train model
# ---------------------------
def train_model(X, y, model_out=MODEL_OUT):
    """Train Random Forest model."""
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    X = X.fillna(-999)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train Random Forest
    model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    # Evaluate on training set
    y_train_pred = model.predict(X_train)
    train_rmse = math.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    # Evaluate on test set
    y_test_pred = model.predict(X_test)
    test_rmse = math.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("\n" + "="*60)
    print("MODEL PERFORMANCE")
    print("="*60)
    print(f"\nTraining Set:")
    print(f"  RMSE: {train_rmse:.3f}")
    print(f"  MAE:  {train_mae:.3f}")
    print(f"  R²:   {train_r2:.3f}")
    
    print(f"\nTest Set:")
    print(f"  RMSE: {test_rmse:.3f}")
    print(f"  MAE:  {test_mae:.3f}")
    print(f"  R²:   {test_r2:.3f}")

    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.6, edgecolors='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Magnitude', fontsize=12)
    plt.ylabel('Predicted Magnitude', fontsize=12)
    plt.title('Model Performance: Predicted vs Actual Magnitudes', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_file = os.path.join(os.path.dirname(model_out), "model_performance.png")
    plt.savefig(plot_file, dpi=300)
    print(f"\nPerformance plot saved to: {plot_file}")
    plt.close()

    # Save model
    joblib.dump(model, model_out)
    print(f"Model saved to: {model_out}")
    
    return model

# ---------------------------
# Predict magnitudes for catalog
# ---------------------------
def predict_magnitudes_for_catalog(catalog, mseed_dir, model_path=MODEL_OUT):
    """
    Predict magnitudes for ObsPy catalog.
    catalog: obspy Catalog with Event.origins available
    Returns: catalog with magnitudes, predictions DataFrame
    """
    print("\n" + "="*60)
    print("PREDICTING MAGNITUDES")
    print("="*60)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train first!")

    model = joblib.load(model_path)
    print(f"Model loaded from: {model_path}")

    # Load waveforms
    print("\nLoading waveform files...")
    files = glob.glob(os.path.join(mseed_dir, "*.mseed"))
    streams = [read(f) for f in tqdm(files, desc="Reading mseed files")]
    all_traces = sum(streams, start=Stream())

    event_preds = []
    print(f"\nProcessing {len(catalog)} events...")
    
    for idx, ev in enumerate(tqdm(catalog, desc="Predicting magnitudes")):
        try:
            origin = ev.preferred_origin() or ev.origins[0]
            origin_time = origin.time
            event_lat = getattr(origin, "latitude", None)
            event_lon = getattr(origin, "longitude", None)
        except Exception:
            continue

        # Trim window around event
        st_trim = all_traces.copy().trim(starttime=origin_time - 10, endtime=origin_time + 60, pad=True, fill_value=0)
        
        # Find picks
        picks = []
        for tr in st_trim:
            try:
                cft = classic_sta_lta(tr.data, int(2*tr.stats.sampling_rate), int(10*tr.stats.sampling_rate))
                on_of = trigger_onset(cft, 2.5, 1.0)
                if len(on_of) > 0:
                    pick_time = tr.stats.starttime + on_of[0][0] / tr.stats.sampling_rate
                    if pick_time >= origin_time - 10 and pick_time <= origin_time + 60:
                        picks.append((tr.stats.station, pick_time))
            except Exception:
                continue

        if len(picks) == 0:
            continue

        agg = aggregate_features_for_event(st_trim, picks, origin_time, event_lat, event_lon)
        if agg is None:
            continue

        X_row = pd.DataFrame([agg]).fillna(-999)
        pred_mag = model.predict(X_row)[0]
        
        # Store predicted magnitude
        from obspy.core.event import Magnitude
        mag_obj = Magnitude(mag=float(pred_mag), magnitude_type="ML")
        ev.magnitudes.append(mag_obj)
        
        event_preds.append({
            "event_index": idx,
            "origin_time": str(origin_time),
            "predicted_magnitude": float(pred_mag),
            "n_stations": agg.get("n_stations", 0),
            "latitude": event_lat,
            "longitude": event_lon
        })

    # Save predictions
    pred_df = pd.DataFrame(event_preds)
    pred_df.to_csv(PREDICTIONS_OUT, index=False)
    print(f"\nPredictions saved to: {PREDICTIONS_OUT}")
    print(f"Predicted magnitudes for {len(event_preds)} events")
    
    if len(event_preds) > 0:
        print(f"\nMagnitude Statistics:")
        print(f"  Mean: {pred_df['predicted_magnitude'].mean():.2f}")
        print(f"  Min:  {pred_df['predicted_magnitude'].min():.2f}")
        print(f"  Max:  {pred_df['predicted_magnitude'].max():.2f}")
        print(f"  Std:  {pred_df['predicted_magnitude'].std():.2f}")
    
    return catalog, event_preds

# ---------------------------
# Create sample training data
# ---------------------------
def create_sample_training_data():
    """Create a sample training CSV from detected events with synthetic magnitudes."""
    print("\n" + "="*60)
    print("CREATING SAMPLE TRAINING DATA")
    print("="*60)
    
    if os.path.exists(DETECTED_EVENTS):
        catalog = read_events(DETECTED_EVENTS)
        print(f"Loaded {len(catalog)} events from {DETECTED_EVENTS}")
        
        training_data = []
        for idx, ev in enumerate(catalog):
            try:
                origin = ev.preferred_origin() or ev.origins[0]
                # Generate synthetic magnitude (in practice, these should be real measurements)
                # Here we use a random magnitude for demonstration
                synthetic_mag = np.random.uniform(2.0, 5.0)
                
                training_data.append({
                    "event_id": f"event_{idx:04d}",
                    "origin_time": str(origin.time),
                    "magnitude": round(synthetic_mag, 2),
                    "latitude": getattr(origin, "latitude", None),
                    "longitude": getattr(origin, "longitude", None)
                })
            except Exception as e:
                print(f"Error processing event {idx}: {e}")
                continue
        
        df = pd.DataFrame(training_data)
        df.to_csv(TRAINING_CSV, index=False)
        print(f"\nSample training data saved to: {TRAINING_CSV}")
        print(f"Created {len(df)} training samples")
        print("\nWARNING: These are SYNTHETIC magnitudes for demonstration.")
        print("Replace with real magnitude measurements for actual training!")
        return df
    else:
        print(f"No detected events file found at: {DETECTED_EVENTS}")
        print("Please run event detection first.")
        return None

# ---------------------------
# Main execution
# ---------------------------
if __name__ == "__main__":
    import sys
    
    print("\n" + "="*60)
    print("SEISMIC MAGNITUDE ESTIMATION")
    print("="*60)
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        print("\nUsage:")
        print("  python train_test_model.py train     - Train a new model")
        print("  python train_test_model.py predict   - Predict magnitudes for catalog")
        print("  python train_test_model.py sample    - Create sample training data")
        print("  python train_test_model.py full      - Run full pipeline (train + predict)")
        sys.exit(0)
    
    if mode == "sample":
        # Create sample training data
        create_sample_training_data()
    
    elif mode == "train":
        # Train model
        if not os.path.exists(TRAINING_CSV):
            print(f"\nTraining CSV not found: {TRAINING_CSV}")
            print("Run 'python train_test_model.py sample' to create sample data,")
            print("or provide your own training CSV with columns: event_id, origin_time, magnitude")
            sys.exit(1)
        
        train_df = pd.read_csv(TRAINING_CSV)
        print(f"\nLoaded {len(train_df)} training events from {TRAINING_CSV}")
        
        X, y = build_training_dataframe(DATA_DIR, train_df)
        
        if len(X) > 0:
            model = train_model(X, y)
            print("\n" + "="*60)
            print("TRAINING COMPLETE")
            print("="*60)
        else:
            print("\nNo training features extracted. Check your training CSV and waveforms.")
    
    elif mode == "predict":
        # Predict magnitudes
        if not os.path.exists(MODEL_OUT):
            print(f"\nModel not found: {MODEL_OUT}")
            print("Train a model first using: python train_test_model.py train")
            sys.exit(1)
        
        if not os.path.exists(DETECTED_EVENTS):
            print(f"\nDetected events file not found: {DETECTED_EVENTS}")
            print("Run event detection first.")
            sys.exit(1)
        
        catalog = read_events(DETECTED_EVENTS)
        catalog_with_mags, preds = predict_magnitudes_for_catalog(catalog, DATA_DIR)
        
        # Save updated catalog
        catalog_with_mags.write(CATALOG_WITH_MAGS, format="QUAKEML")
        print(f"\nCatalog with magnitudes saved to: {CATALOG_WITH_MAGS}")
        
        print("\n" + "="*60)
        print("PREDICTION COMPLETE")
        print("="*60)
    
    elif mode == "full":
        # Full pipeline: train + predict
        print("\nRunning FULL pipeline (training + prediction)...\n")
        
        # Check/create training data
        if not os.path.exists(TRAINING_CSV):
            print("Training data not found. Creating sample data...")
            create_sample_training_data()
        
        # Train
        train_df = pd.read_csv(TRAINING_CSV)
        X, y = build_training_dataframe(DATA_DIR, train_df)
        if len(X) > 0:
            model = train_model(X, y)
        else:
            print("No training features extracted.")
            sys.exit(1)
        
        # Predict
        if os.path.exists(DETECTED_EVENTS):
            catalog = read_events(DETECTED_EVENTS)
            catalog_with_mags, preds = predict_magnitudes_for_catalog(catalog, DATA_DIR)
            catalog_with_mags.write(CATALOG_WITH_MAGS, format="QUAKEML")
            print(f"\nCatalog with magnitudes saved to: {CATALOG_WITH_MAGS}")
        
        print("\n" + "="*60)
        print("FULL PIPELINE COMPLETE")
        print("="*60)
    
    else:
        print(f"\nUnknown mode: {mode}")
        print("Use 'train', 'predict', 'sample', or 'full'")
