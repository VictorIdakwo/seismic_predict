import streamlit as st
import os
import math
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from obspy import read, UTCDateTime, Stream
from obspy.signal.filter import bandpass, envelope
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from datetime import datetime
import tempfile
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Seismic Data Analyzer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Constants - Use relative paths for deployment
STATION_FILE = os.path.join(os.path.dirname(__file__), "stations.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "magnitude_model.joblib")

# Load station metadata
@st.cache_data
def load_stations():
    """Load and process station metadata."""
    if os.path.exists(STATION_FILE):
        df = pd.read_csv(STATION_FILE)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("/", "_")
        return df
    return None

# Load trained model
@st.cache_resource
def load_model():
    """Load the trained magnitude estimation model."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

def station_coords(station_code, stations_df):
    """Get coordinates for a station."""
    if stations_df is None:
        return None
    row = stations_df[stations_df["station"] == station_code]
    if row.empty:
        return None
    return float(row["latitude"].values[0]), float(row["longitude"].values[0])

# Feature extraction functions
def compute_dominant_frequency(data, sampling_rate):
    """Estimate dominant frequency via FFT peak."""
    n = len(data)
    if n == 0:
        return np.nan
    data = data - np.mean(data)
    freqs = np.fft.rfftfreq(n, d=1.0/sampling_rate)
    ps = np.abs(np.fft.rfft(data))**2
    idx = np.argmax(ps)
    return float(freqs[idx]) if idx < len(freqs) else np.nan

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
        return 1000.0
    snr = rms_sig / rms_noise
    return float(min(snr, 1000.0))

def extract_features_from_trace(tr, pick_time, stations_df, event_lat=None, event_lon=None):
    """Extract features from a trace around pick time."""
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
    envelope_peak = float(np.max(envelope(data)))
    dominant_freq = compute_dominant_frequency(data, sr)
    snr = compute_snr(tr, pick_time)
    
    # Duration estimation
    try:
        env = envelope(data)
        peak_idx = np.argmax(env)
        thresh = 0.1 * env[peak_idx]
        post = env[peak_idx:]
        below = np.where(post < thresh)[0]
        dur = float(below[0] / sr) if below.size > 0 else float(len(env) / sr)
    except Exception:
        dur = np.nan
    
    # Distance calculation
    distance_km = np.nan
    amp_corr = np.nan
    if event_lat is not None and event_lon is not None:
        coords = station_coords(tr.stats.station, stations_df)
        if coords is not None:
            sta_lat, sta_lon = coords
            # Haversine distance
            R = 6371.0
            lat1, lon1 = math.radians(event_lat), math.radians(event_lon)
            lat2, lon2 = math.radians(sta_lat), math.radians(sta_lon)
            dlat, dlon = lat2 - lat1, lon2 - lon1
            a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            distance_km = R * c
            if not np.isnan(distance_km) and distance_km > 0:
                amp_corr = peak_abs * distance_km
    
    return {
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

def aggregate_features(station_features):
    """Aggregate station-level features into event-level features."""
    if len(station_features) == 0:
        return None
    
    df = pd.DataFrame(station_features)
    agg = {}
    numeric_cols = ["peak_abs", "rms", "envelope_peak", "dominant_freq", 
                    "snr", "duration", "distance_km", "amp_distance_corr"]
    
    for col in numeric_cols:
        agg[f"{col}_median"] = df[col].median(skipna=True)
        agg[f"{col}_mean"] = df[col].mean(skipna=True)
        agg[f"{col}_std"] = df[col].std(skipna=True)
    agg["n_stations"] = len(df)
    return agg

def detect_events_stalta(st, sta_len=2, lta_len=10, trig_on=2.5, trig_off=1.0):
    """Detect seismic events using STA/LTA."""
    events = []
    
    for tr in st:
        sr = tr.stats.sampling_rate
        try:
            cft = classic_sta_lta(tr.data, int(sta_len * sr), int(lta_len * sr))
            triggers = trigger_onset(cft, trig_on, trig_off)
            
            for trigger in triggers:
                pick_time = tr.stats.starttime + trigger[0] / sr
                end_time = tr.stats.starttime + trigger[1] / sr
                events.append({
                    "station": tr.stats.station,
                    "pick_time": pick_time,
                    "end_time": end_time,
                    "duration": end_time - pick_time,
                    "cft_max": cft[trigger[0]:trigger[1]].max()
                })
        except Exception as e:
            continue
    
    return pd.DataFrame(events)

def plot_waveform(tr, title=None):
    """Plot waveform using plotly."""
    times = tr.times()
    data = tr.data
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times,
        y=data,
        mode='lines',
        name='Amplitude',
        line=dict(color='#1f77b4', width=1)
    ))
    
    fig.update_layout(
        title=title or f"Station: {tr.stats.station} | Channel: {tr.stats.channel}",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude (counts)",
        height=400,
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

def plot_spectrogram(tr):
    """Plot spectrogram of the trace."""
    fig, ax = plt.subplots(figsize=(12, 4))
    tr.spectrogram(wlen=2.0, per_lap=0.9, axes=ax, cmap='viridis')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Frequency (Hz)', fontsize=12)
    ax.set_title(f'Spectrogram - Station: {tr.stats.station}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_frequency_spectrum(tr):
    """Plot frequency spectrum."""
    sr = tr.stats.sampling_rate
    data = tr.data - np.mean(tr.data)
    freqs = np.fft.rfftfreq(len(data), d=1.0/sr)
    ps = np.abs(np.fft.rfft(data))**2
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freqs,
        y=ps,
        mode='lines',
        fill='tozeroy',
        name='Power Spectrum',
        line=dict(color='#ff7f0e', width=2)
    ))
    
    fig.update_layout(
        title=f"Frequency Spectrum - Station: {tr.stats.station}",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power",
        height=400,
        template='plotly_white',
        xaxis=dict(range=[0, min(50, freqs.max())])
    )
    
    return fig

def plot_station_map(stations_df, highlight_station=None):
    """Plot interactive station map."""
    fig = go.Figure()
    
    # All stations
    fig.add_trace(go.Scattergeo(
        lon=stations_df['longitude'],
        lat=stations_df['latitude'],
        text=stations_df['station'],
        mode='markers+text',
        marker=dict(size=10, color='blue', symbol='triangle-up'),
        textposition='top center',
        name='Stations'
    ))
    
    # Highlight specific station
    if highlight_station:
        station_row = stations_df[stations_df['station'] == highlight_station]
        if not station_row.empty:
            fig.add_trace(go.Scattergeo(
                lon=[station_row['longitude'].values[0]],
                lat=[station_row['latitude'].values[0]],
                text=[highlight_station],
                mode='markers',
                marker=dict(size=15, color='red', symbol='star'),
                name='Current Station'
            ))
    
    fig.update_geos(
        projection_type="natural earth",
        showland=True,
        landcolor="lightgray",
        showlakes=True,
        lakecolor="lightblue",
        showcountries=True,
        center=dict(lat=stations_df['latitude'].mean(), lon=stations_df['longitude'].mean()),
        projection_scale=4
    )
    
    fig.update_layout(
        title="Station Locations",
        height=500,
        showlegend=True
    )
    
    return fig

# Main App
def main():
    st.markdown('<div class="main-header">🌍 Seismic Data Analyzer & Magnitude Predictor</div>', 
                unsafe_allow_html=True)
    
    # Load resources
    stations_df = load_stations()
    model = load_model()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/earthquake.png", width=80)
        st.title("⚙️ Configuration")
        
        # File upload
        uploaded_file = st.file_uploader(
            "📁 Upload Seismic Data (.mseed)",
            type=['mseed'],
            help="Upload a MiniSEED file for analysis"
        )
        
        st.markdown("---")
        
        # Analysis parameters
        st.subheader("🔧 Detection Parameters")
        sta_len = st.slider("STA Length (s)", 1.0, 5.0, 2.0, 0.5)
        lta_len = st.slider("LTA Length (s)", 5, 20, 10, 1)
        trig_on = st.slider("Trigger On", 1.5, 5.0, 2.5, 0.1)
        trig_off = st.slider("Trigger Off", 0.5, 2.0, 1.0, 0.1)
        
        st.markdown("---")
        
        # Event location (optional)
        st.subheader("📍 Event Location (Optional)")
        use_location = st.checkbox("Use event location for distance calculations")
        event_lat = None
        event_lon = None
        if use_location:
            event_lat = st.number_input("Latitude", value=6.86685, format="%.5f")
            event_lon = st.number_input("Longitude", value=7.41742, format="%.5f")
        
        st.markdown("---")
        st.info("💡 **Tip:** Adjust STA/LTA parameters to fine-tune event detection sensitivity.")
    
    # Main content
    if uploaded_file is None:
        # Welcome screen
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 📊 Features
            - Waveform visualization
            - Spectrogram analysis
            - Frequency spectrum
            - Event detection (STA/LTA)
            """)
        
        with col2:
            st.markdown("""
            ### 🎯 Analysis
            - Automatic pick detection
            - Feature extraction
            - Multi-station processing
            - Statistical aggregation
            """)
        
        with col3:
            st.markdown("""
            ### 🔮 Prediction
            - ML-based magnitude estimation
            - Confidence metrics
            - Station information
            - Interactive maps
            """)
        
        st.info("👆 Upload a .mseed file from the sidebar to begin analysis!")
        
        # Show station map if available
        if stations_df is not None:
            st.subheader("📍 Available Seismic Stations")
            st.plotly_chart(plot_station_map(stations_df), use_container_width=True)
            
            st.subheader("📋 Station Details")
            st.dataframe(stations_df, use_container_width=True)
        
        return
    
    # Process uploaded file
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mseed') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        # Read seismic data
        with st.spinner("📖 Reading seismic data..."):
            st_data = read(tmp_path)
        
        st.success(f"✅ Successfully loaded {len(st_data)} trace(s)")
        
        # Display metadata
        st.subheader("📄 Data Information")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Number of Traces", len(st_data))
        with col2:
            st.metric("Station(s)", ", ".join(set([tr.stats.station for tr in st_data])))
        with col3:
            st.metric("Sampling Rate", f"{st_data[0].stats.sampling_rate} Hz")
        with col4:
            duration = (st_data[0].stats.endtime - st_data[0].stats.starttime)
            st.metric("Duration", f"{duration:.1f} s")
        
        # Trace selection
        st.subheader("📈 Waveform Visualization")
        trace_options = [f"Trace {i}: {tr.stats.station}.{tr.stats.channel}" 
                        for i, tr in enumerate(st_data)]
        selected_trace_idx = st.selectbox("Select Trace", range(len(st_data)), 
                                         format_func=lambda x: trace_options[x])
        
        tr = st_data[selected_trace_idx]
        
        # Waveform plot
        st.plotly_chart(plot_waveform(tr), use_container_width=True)
        
        # Additional visualizations
        viz_tabs = st.tabs(["🎵 Spectrogram", "📊 Frequency Spectrum", "🗺️ Station Map"])
        
        with viz_tabs[0]:
            with st.spinner("Generating spectrogram..."):
                fig_spec = plot_spectrogram(tr.copy())
                st.pyplot(fig_spec)
        
        with viz_tabs[1]:
            st.plotly_chart(plot_frequency_spectrum(tr), use_container_width=True)
        
        with viz_tabs[2]:
            if stations_df is not None:
                st.plotly_chart(plot_station_map(stations_df, tr.stats.station), 
                              use_container_width=True)
            else:
                st.warning("Station information not available")
        
        # Event Detection
        st.subheader("🔍 Event Detection (STA/LTA)")
        
        with st.spinner("Detecting events..."):
            events_df = detect_events_stalta(st_data, sta_len, lta_len, trig_on, trig_off)
        
        if len(events_df) > 0:
            st.success(f"✅ Detected {len(events_df)} event(s)")
            
            # Show events table
            events_display = events_df.copy()
            events_display['pick_time'] = events_display['pick_time'].apply(str)
            events_display['end_time'] = events_display['end_time'].apply(str)
            events_display['duration'] = events_display['duration'].apply(lambda x: f"{x:.2f} s")
            events_display['cft_max'] = events_display['cft_max'].apply(lambda x: f"{x:.2f}")
            
            st.dataframe(events_display, use_container_width=True)
            
            # Magnitude Prediction
            st.subheader("🔮 Magnitude Prediction")
            
            if model is None:
                st.warning("⚠️ Trained model not found. Please train a model first.")
            else:
                with st.spinner("Extracting features and predicting magnitude..."):
                    # Extract features for all detected picks
                    station_features = []
                    for _, event in events_df.iterrows():
                        tr_candidates = [t for t in st_data if t.stats.station == event['station']]
                        if tr_candidates:
                            feat = extract_features_from_trace(
                                tr_candidates[0], 
                                event['pick_time'],
                                stations_df,
                                event_lat,
                                event_lon
                            )
                            if feat:
                                station_features.append(feat)
                    
                    if len(station_features) > 0:
                        # Aggregate features
                        agg_features = aggregate_features(station_features)
                        
                        if agg_features:
                            # Prepare for prediction
                            X = pd.DataFrame([agg_features])
                            X = X.replace([np.inf, -np.inf], np.nan)
                            X = X.fillna(-999)
                            
                            # Predict
                            predicted_magnitude = model.predict(X)[0]
                            
                            # Display results
                            st.markdown("---")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                st.metric("🎯 Predicted Magnitude", f"{predicted_magnitude:.2f}")
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                st.metric("📡 Stations Used", int(agg_features['n_stations']))
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            with col3:
                                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                magnitude_class = (
                                    "Minor" if predicted_magnitude < 3.0 else
                                    "Light" if predicted_magnitude < 4.0 else
                                    "Moderate" if predicted_magnitude < 5.0 else
                                    "Strong"
                                )
                                st.metric("📊 Classification", magnitude_class)
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Feature details
                            with st.expander("📊 View Extracted Features"):
                                feature_df = pd.DataFrame([agg_features]).T
                                feature_df.columns = ['Value']
                                feature_df.index.name = 'Feature'
                                st.dataframe(feature_df, use_container_width=True)
                            
                            # Station-level features
                            with st.expander("📡 View Station-Level Features"):
                                st.dataframe(pd.DataFrame(station_features), use_container_width=True)
                            
                            # Interpretation
                            st.markdown("---")
                            st.subheader("💡 Interpretation")
                            
                            if predicted_magnitude < 2.0:
                                st.info("🟢 **Micro Earthquake**: Usually not felt, detected only by seismographs.")
                            elif predicted_magnitude < 3.0:
                                st.info("🟢 **Minor**: Felt slightly by some people. No damage to buildings.")
                            elif predicted_magnitude < 4.0:
                                st.success("🟡 **Light**: Often felt but rarely causes damage.")
                            elif predicted_magnitude < 5.0:
                                st.warning("🟠 **Moderate**: Some damage to poorly constructed buildings.")
                            else:
                                st.error("🔴 **Strong**: Can cause significant damage in populated areas.")
                        else:
                            st.error("Failed to aggregate features")
                    else:
                        st.warning("No valid features could be extracted from the detected events")
        else:
            st.warning("⚠️ No events detected. Try adjusting the STA/LTA parameters in the sidebar.")
        
        # Cleanup
        os.unlink(tmp_path)
        
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()
