# 🌍 Seismic Event Detection & Magnitude Prediction

A comprehensive machine learning-based system for seismic event detection and earthquake magnitude estimation using Random Forest regression.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![ObsPy](https://img.shields.io/badge/ObsPy-1.4+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 📋 Overview

This project provides tools for:
- **Seismic Event Detection**: Using STA/LTA (Short-Term Average/Long-Term Average) algorithms
- **Feature Extraction**: Comprehensive waveform analysis including amplitude, frequency, SNR, and duration
- **Magnitude Estimation**: ML-based prediction using Random Forest regression
- **Interactive Analysis**: Streamlit web application for real-time analysis

## 🚀 Features

### 1. Event Detection
- Classic STA/LTA trigger algorithm
- Configurable sensitivity parameters
- Multi-station processing
- Automatic pick detection

### 2. Feature Extraction
- **Amplitude Metrics**: Peak amplitude, RMS, envelope peak
- **Frequency Analysis**: Dominant frequency via FFT
- **Signal Quality**: SNR calculation
- **Duration Estimation**: Signal decay analysis
- **Distance Correction**: Geometric spreading corrections

### 3. Magnitude Prediction
- Random Forest regression model
- Multi-station feature aggregation
- Statistical confidence metrics
- Earthquake classification

### 4. Web Application
- Upload and analyze .mseed files
- Interactive waveform visualization
- Spectrogram and frequency analysis
- Real-time magnitude prediction
- Station location maps

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/VictorIdakwo/seismic_predict.git
cd seismic_predict
```

2. **Create virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements_app.txt
```

## 🎯 Usage

### 1. Train the Magnitude Estimation Model

```bash
python ml_magnitude_estimator.py
```

This will:
- Load training data from `training_events.csv`
- Extract features from seismic waveforms
- Train a Random Forest model
- Save the model to `magnitude_model.joblib`

### 2. Launch the Streamlit Web App

```bash
streamlit run seismic_app.py
```

The app will open at `http://localhost:8501`

### 3. Use the Training Pipeline (Advanced)

```bash
# Create sample training data
python train_test_model.py sample

# Train model
python train_test_model.py train

# Predict magnitudes for catalog
python train_test_model.py predict

# Run full pipeline
python train_test_model.py full
```

## 📁 Project Structure

```
seismic_predict/
│
├── seismic_app.py              # Streamlit web application
├── ml_magnitude_estimator.py   # ML model training script
├── train_test_model.py         # Enhanced training pipeline
│
├── stations.csv                # Station metadata (lat, lon, elevation)
├── training_events.csv         # Training data with known magnitudes
│
├── magnitude_model.joblib      # Trained ML model (generated)
├── magnitude_features.csv      # Extracted features (generated)
│
├── requirements_app.txt        # Python dependencies
├── README.md                   # This file
├── README_STREAMLIT_APP.md     # Detailed app documentation
└── .gitignore                  # Git ignore rules
```

## 🔧 Configuration

### Station Metadata (`stations.csv`)
```csv
s/n,station name,station,latitude,longitude,elevation
1,ILE-EFE,IFE,7.54667,4.54692,289
2,AWKA,AWK,6.24268,7.11155,50
...
```

### Training Data (`training_events.csv`)
```csv
event_id,origin_time,magnitude,latitude,longitude
event_0000,2024-01-03T09:59:52.992000Z,2.69,6.86685,7.41742
...
```

## 📊 Model Performance

The Random Forest model uses 25 aggregated features:
- **Amplitude features**: median, mean, std (peak_abs, rms, envelope_peak)
- **Frequency features**: median, mean, std (dominant_freq)
- **Quality features**: median, mean, std (snr)
- **Duration features**: median, mean, std (duration)
- **Distance features**: median, mean, std (distance_km, amp_distance_corr)
- **Station count**: n_stations

Current performance (5 training samples):
- RMSE: 0.642
- Model: Random Forest with 200 estimators

*Note: Model performance improves significantly with more training data (50+ events recommended)*

## 🌐 Web Application Features

### Upload & Analyze
1. Upload `.mseed` files via drag-and-drop
2. View metadata (stations, sampling rate, duration)
3. Select and visualize individual traces

### Visualizations
- **Waveform**: Interactive time-series plots
- **Spectrogram**: Time-frequency analysis
- **Frequency Spectrum**: Power spectral density
- **Station Map**: Geographic locations

### Event Detection
- Adjustable STA/LTA parameters
- Real-time detection results
- Event statistics table

### Magnitude Prediction
- Automatic feature extraction
- ML-based magnitude estimation
- Earthquake classification
- Confidence metrics

## 📚 Dependencies

```
streamlit>=1.28.0
plotly>=5.17.0
obspy>=1.4.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
joblib>=1.3.0
tqdm>=4.65.0
```

## 🎓 Methodology

### STA/LTA Event Detection
Short-Term Average (STA) to Long-Term Average (LTA) ratio:
```
STA/LTA = avg(|signal|²[t-STA:t]) / avg(|signal|²[t-LTA:t])
```

### Magnitude Estimation
Features aggregated across stations → Random Forest → Predicted Magnitude

### Earthquake Classification
- **Micro**: < 2.0
- **Minor**: 2.0-2.9
- **Light**: 3.0-3.9
- **Moderate**: 4.0-4.9
- **Strong**: ≥ 5.0

## 🔬 Research Applications

This system can be used for:
- Real-time earthquake monitoring
- Seismic hazard assessment
- Catalog generation
- Educational purposes
- Research on magnitude estimation algorithms

## 🛠️ Customization

### Modify Detection Parameters
Edit in `seismic_app.py`:
```python
sta_len = 2.0      # Short-term window (seconds)
lta_len = 10.0     # Long-term window (seconds)
trig_on = 2.5      # Trigger threshold
trig_off = 1.0     # De-trigger threshold
```

### Change ML Model
Replace Random Forest in `ml_magnitude_estimator.py`:
```python
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(n_estimators=200)
```

### Add New Features
1. Implement feature extraction function
2. Update `extract_features_from_trace()`
3. Add to aggregation in `aggregate_features()`
4. Retrain model

## 📖 Documentation

- **Main README**: This file
- **App Documentation**: [README_STREAMLIT_APP.md](README_STREAMLIT_APP.md)
- **ObsPy Docs**: https://docs.obspy.org/
- **Streamlit Docs**: https://docs.streamlit.io/

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Enhanced feature engineering
- Deep learning models
- Real-time streaming support
- P-wave/S-wave separation
- Location estimation
- Multiple magnitude scales (Mw, ML, Mb)

## 📝 License

This project is licensed under the MIT License.

## 👨‍💻 Author

**Victor Idakwo**
- GitHub: [@VictorIdakwo](https://github.com/VictorIdakwo)
- Organization: eHealth Africa

## 🙏 Acknowledgments

- ObsPy Development Team
- eHealth Africa Seismology Team
- Seismological research community

## 📧 Contact

For questions or collaboration:
- Open an issue on GitHub
- Email: [Your Email]

## 🔗 References

1. Allen, R.V. (1978). "Automatic earthquake recognition and timing from single traces"
2. Richter, C.F. (1935). "An instrumental earthquake magnitude scale"
3. Bormann, P. (2012). "New Manual of Seismological Observatory Practice"

---

**Built with ❤️ for seismology research and earthquake monitoring**

🌍 **Making seismic analysis accessible to everyone**
