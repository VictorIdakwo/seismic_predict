import os
import glob
import math
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
from obspy import read, Stream, UTCDateTime
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# ---------------------------
# Configuration
# ---------------------------
class Config:
    # Paths
    DATA_DIR = r"C:\Users\victor.idakwo\Documents\ehealth Africa\ehealth Africa\eHA GitHub\seismic Analytics\Jan"
    STATION_FILE = r"C:\Users\victor.idakwo\Documents\ehealth Africa\ehealth Africa\eHA GitHub\seismic Analytics\stations.csv"
    TRAINING_CSV = r"C:\Users\victor.idakwo\Documents\ehealth Africa\ehealth Africa\eHA GitHub\seismic Analytics\training_events.csv"
    
    # Output paths
    MODEL_DIR = r"C:\Users\victor.idakwo\Documents\ehealth Africa\ehealth Africa\eHA GitHub\seismic Analytics\models"
    RESULTS_DIR = r"C:\Users\victor.idakwo\Documents\ehealth Africa\ehealth Africa\eHA GitHub\seismic Analytics\results"
    
    # Model files
    MODEL_PATH = os.path.join(MODEL_DIR, "magnitude_model.joblib")
    SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler.joblib")
    FEATURE_COLUMNS_PATH = os.path.join(MODEL_DIR, "feature_columns.joblib")
    
    # Feature and result files
    FEATURES_CSV = os.path.join(RESULTS_DIR, "training_features.csv")
    PREDICTIONS_CSV = os.path.join(RESULTS_DIR, "predictions.csv")
    METRICS_CSV = os.path.join(RESULTS_DIR, "model_metrics.csv")
    
    # Model parameters
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CV_FOLDS = 5
    
    # Feature extraction parameters
    WINDOW_BEFORE_ORIGIN = 10  # seconds
    WINDOW_AFTER_ORIGIN = 60   # seconds
    STA_WINDOW = 2             # seconds for STA/LTA
    LTA_WINDOW = 10            # seconds for STA/LTA
    TRIGGER_ON = 2.5
    TRIGGER_OFF = 1.0

# Create directories if they don't exist
os.makedirs(Config.MODEL_DIR, exist_ok=True)
os.makedirs(Config.RESULTS_DIR, exist_ok=True)

# ---------------------------
# Station Metadata Handler
# ---------------------------
class StationMetadata:
    def __init__(self, station_file):
        self.stations_df = pd.read_csv(station_file)
        self.stations_df.columns = (
            self.stations_df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace("/", "_")
        )
    
    def get_coordinates(self, station_code):
        """Get station coordinates (lat, lon)"""
        row = self.stations_df[self.stations_df["station"] == station_code]
        if row.empty:
            return None
        return float(row["latitude"].values[0]), float(row["longitude"].values[0])
    
    def get_all_stations(self):
        """Get list of all station codes"""
        return self.stations_df["station"].tolist()

# ---------------------------
# Feature Extraction
# ---------------------------
class FeatureExtractor:
    def __init__(self, station_metadata):
        self.station_metadata = station_metadata
    
    @staticmethod
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
    
    @staticmethod
    def compute_snr(tr, pick_time, pre_sec=2.0, win_sec=2.0):
        """Estimate SNR as ratio of RMS(signal) / RMS(noise)."""
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
    
    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate great-circle distance between two points on Earth (in km)."""
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2.0)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2.0)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c
    
    def extract_trace_features(self, tr, pick_time, origin_time, event_lat=None, event_lon=None):
        """Extract features from a single trace."""
        sr = tr.stats.sampling_rate
        t0 = pick_time - 1.0
        t1 = pick_time + 5.0
        
        try:
            seg = tr.slice(starttime=t0, endtime=t1)
            data = seg.data.astype(float)
        except Exception:
            return None
        
        if data.size == 0:
            return None
        
        # Amplitude features
        peak_abs = float(np.max(np.abs(data)))
        rms = float(np.sqrt(np.mean(data**2)))
        peak_to_peak = float(np.max(data) - np.min(data))
        
        # Envelope features
        from obspy.signal.filter import envelope
        env = envelope(data)
        envelope_peak = float(np.max(env))
        envelope_mean = float(np.mean(env))
        
        # Frequency features
        dominant_freq = self.compute_dominant_frequency(data, sr)
        
        # SNR
        snr = self.compute_snr(tr, pick_time)
        
        # Signal duration
        try:
            peak_idx = np.argmax(env)
            thresh = 0.1 * env[peak_idx]
            post = env[peak_idx:]
            below = np.where(post < thresh)[0]
            if below.size > 0:
                duration = float(below[0] / sr)
            else:
                duration = float(len(env) / sr)
        except Exception:
            duration = np.nan
        
        # Distance and distance-corrected amplitude
        distance_km = np.nan
        amp_distance_corr = np.nan
        if event_lat is not None and event_lon is not None:
            coords = self.station_metadata.get_coordinates(tr.stats.station)
            if coords is not None:
                sta_lat, sta_lon = coords
                distance_km = self.haversine_distance(event_lat, event_lon, sta_lat, sta_lon)
                if not np.isnan(distance_km) and distance_km > 0:
                    amp_distance_corr = peak_abs * distance_km
        
        features = {
            "station": tr.stats.station,
            "peak_abs": peak_abs,
            "rms": rms,
            "peak_to_peak": peak_to_peak,
            "envelope_peak": envelope_peak,
            "envelope_mean": envelope_mean,
            "dominant_freq": dominant_freq,
            "snr": snr,
            "duration": duration,
            "distance_km": distance_km,
            "amp_distance_corr": amp_distance_corr,
            "sampling_rate": sr
        }
        return features
    
    def aggregate_event_features(self, st_trim, picks, origin_time, event_lat=None, event_lon=None):
        """Aggregate station-level features into event-level feature vector."""
        station_features = []
        for station, pick_time in picks:
            tr_candidates = [tr for tr in st_trim if tr.stats.station == station]
            if not tr_candidates:
                continue
            
            tr = tr_candidates[0]
            feat = self.extract_trace_features(tr, pick_time, origin_time, event_lat, event_lon)
            if feat:
                station_features.append(feat)
        
        if len(station_features) == 0:
            return None
        
        df = pd.DataFrame(station_features)
        
        # Aggregate statistics
        agg = {}
        numeric_cols = ["peak_abs", "rms", "peak_to_peak", "envelope_peak", "envelope_mean",
                       "dominant_freq", "snr", "duration", "distance_km", "amp_distance_corr"]
        
        for col in numeric_cols:
            agg[f"{col}_median"] = df[col].median(skipna=True)
            agg[f"{col}_mean"] = df[col].mean(skipna=True)
            agg[f"{col}_std"] = df[col].std(skipna=True)
            agg[f"{col}_max"] = df[col].max(skipna=True)
            agg[f"{col}_min"] = df[col].min(skipna=True)
        
        agg["n_stations"] = len(df)
        return agg

# ---------------------------
# Data Loader
# ---------------------------
class DataLoader:
    def __init__(self, mseed_dir):
        self.mseed_dir = mseed_dir
        self.all_traces = None
    
    def load_traces(self):
        """Load all traces from mseed directory."""
        print("Loading waveform data...")
        files = glob.glob(os.path.join(self.mseed_dir, "*.mseed"))
        if not files:
            raise FileNotFoundError(f"No .mseed files found in {self.mseed_dir}")
        
        streams = []
        for f in tqdm(files, desc="Loading mseed files"):
            try:
                streams.append(read(f))
            except Exception as e:
                print(f"Error reading {f}: {e}")
        
        self.all_traces = sum(streams, start=Stream())
        print(f"Loaded {len(self.all_traces)} traces")
        return self.all_traces
    
    def find_picks(self, st_trim, origin_time):
        """Find picks in trimmed stream using STA/LTA."""
        picks = []
        for tr in st_trim:
            try:
                cft = classic_sta_lta(
                    tr.data, 
                    int(Config.STA_WINDOW * tr.stats.sampling_rate), 
                    int(Config.LTA_WINDOW * tr.stats.sampling_rate)
                )
                on_of = trigger_onset(cft, Config.TRIGGER_ON, Config.TRIGGER_OFF)
                if len(on_of) > 0:
                    pick_time = tr.stats.starttime + on_of[0][0] / tr.stats.sampling_rate
                    if abs(pick_time - origin_time) < Config.WINDOW_AFTER_ORIGIN:
                        picks.append((tr.stats.station, pick_time))
            except Exception:
                continue
        return picks

# ---------------------------
# Model Trainer
# ---------------------------
class ModelTrainer:
    def __init__(self, feature_extractor, data_loader):
        self.feature_extractor = feature_extractor
        self.data_loader = data_loader
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.metrics = {}
    
    def build_training_data(self, catalog_df):
        """Build training data from catalog."""
        print("Building training features...")
        X_rows = []
        y = []
        
        if self.data_loader.all_traces is None:
            self.data_loader.load_traces()
        
        for idx, row in tqdm(catalog_df.iterrows(), total=len(catalog_df), desc="Processing events"):
            origin_time = UTCDateTime(row["origin_time"])
            event_lat = row.get("latitude", None)
            event_lon = row.get("longitude", None)
            magnitude = row["magnitude"]
            
            # Trim stream around event
            window_start = origin_time - Config.WINDOW_BEFORE_ORIGIN
            window_end = origin_time + Config.WINDOW_AFTER_ORIGIN
            st_trim = self.data_loader.all_traces.copy().trim(
                starttime=window_start, 
                endtime=window_end, 
                pad=True, 
                fill_value=0
            )
            
            # Find picks
            picks = self.data_loader.find_picks(st_trim, origin_time)
            
            if len(picks) < 1:
                continue
            
            # Extract and aggregate features
            agg = self.feature_extractor.aggregate_event_features(
                st_trim, picks, origin_time, event_lat, event_lon
            )
            
            if agg is None:
                continue
            
            agg["event_id"] = row.get("event_id", idx)
            agg["origin_time"] = str(origin_time)
            X_rows.append(agg)
            y.append(magnitude)
        
        X = pd.DataFrame(X_rows)
        y = pd.Series(y, name="magnitude")
        
        # Save features
        X.to_csv(Config.FEATURES_CSV, index=False)
        print(f"Features saved to {Config.FEATURES_CSV}")
        
        return X, y
    
    def train(self, X, y, model_type="random_forest", tune_hyperparameters=False):
        """Train the model."""
        print(f"\nTraining {model_type} model...")
        
        # Separate metadata from features
        metadata_cols = ["event_id", "origin_time"]
        feature_cols = [col for col in X.columns if col not in metadata_cols]
        X_features = X[feature_cols].copy()
        
        # Handle missing values
        X_features = X_features.fillna(-999)
        
        # Store feature columns
        self.feature_columns = feature_cols
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Select and train model
        if model_type == "random_forest":
            if tune_hyperparameters:
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 15, 20, None],
                    'min_samples_split': [2, 5, 10]
                }
                base_model = RandomForestRegressor(random_state=Config.RANDOM_STATE, n_jobs=-1)
                grid_search = GridSearchCV(
                    base_model, param_grid, cv=Config.CV_FOLDS, 
                    scoring='neg_mean_squared_error', n_jobs=-1
                )
                grid_search.fit(X_train_scaled, y_train)
                self.model = grid_search.best_estimator_
                print(f"Best parameters: {grid_search.best_params_}")
            else:
                self.model = RandomForestRegressor(
                    n_estimators=200, max_depth=15, 
                    random_state=Config.RANDOM_STATE, n_jobs=-1
                )
                self.model.fit(X_train_scaled, y_train)
        
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                random_state=Config.RANDOM_STATE
            )
            self.model.fit(X_train_scaled, y_train)
        
        elif model_type == "ridge":
            self.model = Ridge(alpha=1.0, random_state=Config.RANDOM_STATE)
            self.model.fit(X_train_scaled, y_train)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Evaluate
        self._evaluate(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # Save model
        self._save_model()
        
        return self.model
    
    def _evaluate(self, X_train, X_test, y_train, y_test):
        """Evaluate model performance."""
        print("\nEvaluating model...")
        
        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Metrics
        train_rmse = math.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = math.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        self.metrics = {
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "train_mae": train_mae,
            "test_mae": test_mae,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "n_train": len(y_train),
            "n_test": len(y_test)
        }
        
        print(f"\nTraining Metrics:")
        print(f"  RMSE: {train_rmse:.3f}")
        print(f"  MAE:  {train_mae:.3f}")
        print(f"  R²:   {train_r2:.3f}")
        
        print(f"\nTest Metrics:")
        print(f"  RMSE: {test_rmse:.3f}")
        print(f"  MAE:  {test_mae:.3f}")
        print(f"  R²:   {test_r2:.3f}")
        
        # Save metrics
        metrics_df = pd.DataFrame([self.metrics])
        metrics_df.to_csv(Config.METRICS_CSV, index=False)
        print(f"\nMetrics saved to {Config.METRICS_CSV}")
        
        # Plot predictions vs actual
        self._plot_predictions(y_test, y_test_pred)
    
    def _plot_predictions(self, y_true, y_pred):
        """Plot predicted vs actual magnitudes."""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k')
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                 'r--', lw=2, label='Perfect Prediction')
        plt.xlabel('Actual Magnitude', fontsize=12)
        plt.ylabel('Predicted Magnitude', fontsize=12)
        plt.title('Predicted vs Actual Magnitudes', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(Config.RESULTS_DIR, "predictions_vs_actual.png")
        plt.savefig(plot_path, dpi=300)
        print(f"Prediction plot saved to {plot_path}")
        plt.close()
    
    def _save_model(self):
        """Save model, scaler, and feature columns."""
        joblib.dump(self.model, Config.MODEL_PATH)
        joblib.dump(self.scaler, Config.SCALER_PATH)
        joblib.dump(self.feature_columns, Config.FEATURE_COLUMNS_PATH)
        print(f"\nModel saved to {Config.MODEL_PATH}")
        print(f"Scaler saved to {Config.SCALER_PATH}")
        print(f"Feature columns saved to {Config.FEATURE_COLUMNS_PATH}")

# ---------------------------
# Predictor
# ---------------------------
class MagnitudePredictor:
    def __init__(self, feature_extractor, data_loader):
        self.feature_extractor = feature_extractor
        self.data_loader = data_loader
        self.model = None
        self.scaler = None
        self.feature_columns = None
    
    def load_model(self):
        """Load trained model."""
        if not os.path.exists(Config.MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {Config.MODEL_PATH}")
        
        self.model = joblib.load(Config.MODEL_PATH)
        self.scaler = joblib.load(Config.SCALER_PATH)
        self.feature_columns = joblib.load(Config.FEATURE_COLUMNS_PATH)
        print("Model loaded successfully")
    
    def predict_catalog(self, catalog):
        """Predict magnitudes for ObsPy catalog."""
        print("\nPredicting magnitudes for catalog...")
        
        if self.data_loader.all_traces is None:
            self.data_loader.load_traces()
        
        predictions = []
        
        for ev in tqdm(catalog, desc="Processing events"):
            try:
                origin = ev.preferred_origin() or ev.origins[0]
                origin_time = origin.time
                event_lat = getattr(origin, "latitude", None)
                event_lon = getattr(origin, "longitude", None)
            except Exception:
                continue
            
            # Trim stream
            st_trim = self.data_loader.all_traces.copy().trim(
                starttime=origin_time - Config.WINDOW_BEFORE_ORIGIN,
                endtime=origin_time + Config.WINDOW_AFTER_ORIGIN,
                pad=True, fill_value=0
            )
            
            # Find picks
            picks = self.data_loader.find_picks(st_trim, origin_time)
            
            if len(picks) == 0:
                continue
            
            # Extract features
            agg = self.feature_extractor.aggregate_event_features(
                st_trim, picks, origin_time, event_lat, event_lon
            )
            
            if agg is None:
                continue
            
            # Predict
            X_row = pd.DataFrame([agg])
            X_row = X_row[self.feature_columns].fillna(-999)
            X_scaled = self.scaler.transform(X_row)
            pred_mag = self.model.predict(X_scaled)[0]
            
            # Store magnitude in event
            from obspy.core.event import Magnitude
            mag_obj = Magnitude(mag=float(pred_mag), magnitude_type="ML")
            ev.magnitudes.append(mag_obj)
            
            predictions.append({
                "origin_time": str(origin_time),
                "predicted_magnitude": float(pred_mag),
                "n_stations": agg.get("n_stations", 0),
                "latitude": event_lat,
                "longitude": event_lon
            })
        
        # Save predictions
        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv(Config.PREDICTIONS_CSV, index=False)
        print(f"\nPredictions saved to {Config.PREDICTIONS_CSV}")
        
        return catalog, pred_df
    
    def predict_single_event(self, origin_time, event_lat=None, event_lon=None):
        """Predict magnitude for a single event."""
        if self.data_loader.all_traces is None:
            self.data_loader.load_traces()
        
        origin_time = UTCDateTime(origin_time)
        
        # Trim stream
        st_trim = self.data_loader.all_traces.copy().trim(
            starttime=origin_time - Config.WINDOW_BEFORE_ORIGIN,
            endtime=origin_time + Config.WINDOW_AFTER_ORIGIN,
            pad=True, fill_value=0
        )
        
        # Find picks
        picks = self.data_loader.find_picks(st_trim, origin_time)
        
        if len(picks) == 0:
            print("No picks found for event")
            return None
        
        # Extract features
        agg = self.feature_extractor.aggregate_event_features(
            st_trim, picks, origin_time, event_lat, event_lon
        )
        
        if agg is None:
            print("Failed to extract features")
            return None
        
        # Predict
        X_row = pd.DataFrame([agg])
        X_row = X_row[self.feature_columns].fillna(-999)
        X_scaled = self.scaler.transform(X_row)
        pred_mag = self.model.predict(X_scaled)[0]
        
        result = {
            "predicted_magnitude": float(pred_mag),
            "n_stations": agg.get("n_stations", 0),
            "features": agg
        }
        
        return result

# ---------------------------
# Main execution
# ---------------------------
def train_pipeline(training_csv=Config.TRAINING_CSV, model_type="random_forest", tune=False):
    """Complete training pipeline."""
    print("=" * 60)
    print("SEISMIC MAGNITUDE ESTIMATION - TRAINING PIPELINE")
    print("=" * 60)
    
    # Initialize components
    station_meta = StationMetadata(Config.STATION_FILE)
    feature_extractor = FeatureExtractor(station_meta)
    data_loader = DataLoader(Config.DATA_DIR)
    trainer = ModelTrainer(feature_extractor, data_loader)
    
    # Load training catalog
    print(f"\nLoading training catalog from {training_csv}")
    training_df = pd.read_csv(training_csv)
    print(f"Loaded {len(training_df)} training events")
    
    # Build features
    X, y = trainer.build_training_data(training_df)
    print(f"\nExtracted features for {len(X)} events")
    
    if len(X) == 0:
        print("No features extracted. Check data and configuration.")
        return None
    
    # Train model
    model = trainer.train(X, y, model_type=model_type, tune_hyperparameters=tune)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    return trainer

def predict_pipeline(catalog_path=None, catalog=None):
    """Complete prediction pipeline."""
    print("=" * 60)
    print("SEISMIC MAGNITUDE ESTIMATION - PREDICTION PIPELINE")
    print("=" * 60)
    
    # Initialize components
    station_meta = StationMetadata(Config.STATION_FILE)
    feature_extractor = FeatureExtractor(station_meta)
    data_loader = DataLoader(Config.DATA_DIR)
    predictor = MagnitudePredictor(feature_extractor, data_loader)
    
    # Load model
    predictor.load_model()
    
    # Load catalog
    if catalog is None:
        if catalog_path is None:
            raise ValueError("Must provide either catalog_path or catalog")
        from obspy import read_events
        print(f"\nLoading catalog from {catalog_path}")
        catalog = read_events(catalog_path)
    
    print(f"Processing {len(catalog)} events")
    
    # Predict
    catalog_with_mags, predictions = predictor.predict_catalog(catalog)
    
    print("\n" + "=" * 60)
    print("PREDICTION COMPLETE")
    print("=" * 60)
    
    return catalog_with_mags, predictions

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Training:   python model_training_prediction.py train [model_type] [tune]")
        print("  Prediction: python model_training_prediction.py predict [catalog_path]")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    
    if mode == "train":
        model_type = sys.argv[2] if len(sys.argv) > 2 else "random_forest"
        tune = sys.argv[3].lower() == "true" if len(sys.argv) > 3 else False
        train_pipeline(model_type=model_type, tune=tune)
    
    elif mode == "predict":
        catalog_path = sys.argv[2] if len(sys.argv) > 2 else "ml_detected_events.xml"
        catalog_with_mags, predictions = predict_pipeline(catalog_path=catalog_path)
        
        # Save updated catalog
        output_path = catalog_path.replace(".xml", "_with_magnitudes.xml")
        catalog_with_mags.write(output_path, format="QUAKEML")
        print(f"\nCatalog with magnitudes saved to {output_path}")
    
    else:
        print(f"Unknown mode: {mode}")
        print("Use 'train' or 'predict'")
