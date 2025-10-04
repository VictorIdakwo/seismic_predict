# 🌍 Seismic Data Analyzer & Magnitude Predictor

A comprehensive Streamlit web application for analyzing seismic data and predicting earthquake magnitudes using machine learning.

## 🚀 Features

### 📊 Data Analysis
- **Waveform Visualization**: Interactive plots of seismic waveforms
- **Spectrogram Analysis**: Time-frequency analysis of seismic signals
- **Frequency Spectrum**: Power spectrum analysis
- **Multi-trace Support**: Handle multiple seismic traces simultaneously

### 🔍 Event Detection
- **STA/LTA Algorithm**: Classic short-term/long-term average trigger detection
- **Adjustable Parameters**: Fine-tune sensitivity with configurable thresholds
- **Automatic Pick Detection**: Identify P-wave and S-wave arrivals
- **Event Statistics**: Duration, amplitude, and trigger information

### 🎯 Feature Extraction
- **Amplitude Metrics**: Peak amplitude, RMS, envelope peak
- **Frequency Analysis**: Dominant frequency detection
- **Signal Quality**: SNR (Signal-to-Noise Ratio) calculation
- **Duration Analysis**: Signal decay time estimation
- **Distance Correction**: Geometric spreading corrections

### 🔮 Magnitude Prediction
- **ML-Based Estimation**: Random Forest model for magnitude prediction
- **Multi-station Aggregation**: Combines data from multiple stations
- **Confidence Metrics**: Statistical measures of prediction quality
- **Classification**: Automatic earthquake severity classification

### 🗺️ Visualization
- **Interactive Maps**: Geographic display of seismic stations
- **Station Highlighting**: Show active recording stations
- **Plotly Charts**: Interactive, zoomable visualizations
- **Responsive Design**: Works on different screen sizes

## 📋 Requirements

```bash
streamlit>=1.28.0
plotly>=5.17.0
obspy>=1.4.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
joblib>=1.3.0
```

## 🛠️ Installation

1. **Install dependencies**:
```bash
pip install -r requirements_app.txt
```

Or install individually:
```bash
pip install streamlit plotly obspy numpy pandas matplotlib scikit-learn joblib
```

2. **Ensure you have the required files**:
   - `stations.csv` - Station metadata (latitude, longitude, elevation)
   - `magnitude_model.joblib` - Trained ML model (create using `ml_magnitude_estimator.py`)

## 🎮 Usage

### Launch the App

```bash
streamlit run seismic_app.py
```

The app will open in your default web browser at `http://localhost:8501`

### Using the Application

#### 1. **Upload Data**
   - Click "Browse files" in the sidebar
   - Select a `.mseed` file from your computer
   - The file will be automatically processed

#### 2. **View Waveforms**
   - Select a trace from the dropdown menu
   - Explore the interactive waveform plot
   - Zoom, pan, and inspect the data

#### 3. **Analyze Frequency Content**
   - Switch to the "Spectrogram" tab for time-frequency analysis
   - View the "Frequency Spectrum" for dominant frequencies
   - Check the "Station Map" to see recording locations

#### 4. **Configure Detection Parameters**
   - Adjust STA/LTA parameters in the sidebar:
     - **STA Length**: Short-term averaging window (1-5 seconds)
     - **LTA Length**: Long-term averaging window (5-20 seconds)
     - **Trigger On**: Threshold for event detection (1.5-5.0)
     - **Trigger Off**: Threshold for event end (0.5-2.0)

#### 5. **Optional: Add Event Location**
   - Check "Use event location" in the sidebar
   - Enter latitude and longitude
   - This enables distance-based corrections

#### 6. **View Results**
   - Detected events are shown in a table
   - Station-level features are extracted
   - Magnitude prediction is displayed with classification
   - View detailed features and statistics

## 📁 File Structure

```
seismic Analytics/
├── seismic_app.py              # Main Streamlit application
├── ml_magnitude_estimator.py   # ML model training script
├── train_test_model.py         # Enhanced training pipeline
├── magnitude_model.joblib      # Trained model (generated)
├── stations.csv                # Station metadata
├── training_events.csv         # Training data
├── requirements_app.txt        # Python dependencies
└── Jan/                        # Seismic data directory
    └── *.mseed                 # MiniSEED files
```

## 🎛️ Configuration

### STA/LTA Parameters

**STA (Short-Term Average) Length**:
- Typical range: 1-5 seconds
- Shorter values: More sensitive to transient signals
- Longer values: Better noise rejection

**LTA (Long-Term Average) Length**:
- Typical range: 5-20 seconds
- Should be 5-10× longer than STA
- Represents background noise level

**Trigger On Threshold**:
- Typical range: 2.0-5.0
- Higher values: Fewer false detections
- Lower values: More sensitive detection

**Trigger Off Threshold**:
- Typical range: 0.5-2.0
- Should be lower than Trigger On
- Determines when event ends

### Recommended Settings

**For Local Earthquakes** (< 100 km):
- STA: 2 seconds
- LTA: 10 seconds
- Trigger On: 2.5
- Trigger Off: 1.0

**For Regional Events** (100-1000 km):
- STA: 3 seconds
- LTA: 15 seconds
- Trigger On: 3.0
- Trigger Off: 1.5

**For Noisy Data**:
- STA: 2 seconds
- LTA: 15 seconds
- Trigger On: 4.0
- Trigger Off: 1.5

## 📊 Understanding the Results

### Magnitude Classification

- **Micro** (< 2.0): Not felt, detected only by instruments
- **Minor** (2.0-2.9): Felt slightly, no damage
- **Light** (3.0-3.9): Often felt, rarely causes damage
- **Moderate** (4.0-4.9): Can damage poorly constructed buildings
- **Strong** (≥ 5.0): Significant damage in populated areas

### Feature Interpretation

**Peak Amplitude**: Maximum absolute value in the waveform
**RMS**: Root mean square (energy measure)
**Envelope Peak**: Maximum of the signal envelope
**Dominant Frequency**: Primary frequency component (Hz)
**SNR**: Signal-to-noise ratio (higher is better)
**Duration**: Signal decay time (seconds)
**Distance**: Epicentral distance (if location provided)

## 🐛 Troubleshooting

### "Model not found" Error
- Ensure `magnitude_model.joblib` exists
- Run `python ml_magnitude_estimator.py` to train a model

### "No events detected" Warning
- Adjust STA/LTA parameters (try lower Trigger On threshold)
- Check if data contains actual seismic signals
- Verify sampling rate and data quality

### File Upload Issues
- Ensure file is in MiniSEED format (.mseed)
- Check file is not corrupted
- Try with a smaller file first

### Slow Performance
- Large files may take time to process
- Spectrogram generation is computationally intensive
- Consider preprocessing data to shorter time windows

## 🔧 Customization

### Modify Paths

Edit the constants at the top of `seismic_app.py`:

```python
STATION_FILE = r"path/to/your/stations.csv"
MODEL_PATH = r"path/to/your/magnitude_model.joblib"
```

### Add New Features

1. Create new feature extraction function
2. Update `extract_features_from_trace()`
3. Retrain the model with new features
4. Display new features in the app

### Change Model

Replace the Random Forest with another model:

```python
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(n_estimators=200, max_depth=5)
```

## 📚 References

- **ObsPy**: https://docs.obspy.org/
- **Streamlit**: https://docs.streamlit.io/
- **STA/LTA**: Allen, R.V. (1978), Automatic earthquake recognition and timing from single traces
- **Magnitude Estimation**: Richter, C.F. (1935), An instrumental earthquake magnitude scale

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional magnitude estimation algorithms
- More sophisticated event detection
- P-wave/S-wave separation
- Location estimation (not just magnitude)
- Real-time streaming support

## 📝 License

This project is for research and educational purposes.

## 👨‍💻 Author

eHealth Africa Seismology Team

## 📧 Support

For issues or questions, please create an issue in the repository.

---

**Happy Seismic Analysis! 🌍📊**
