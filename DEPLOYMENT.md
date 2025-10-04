# 🚀 Deployment Guide for Streamlit Cloud

## Prerequisites
- GitHub account with your repository: `https://github.com/VictorIdakwo/seismic_predict`
- Streamlit Cloud account (free at https://streamlit.io/cloud)

## Step-by-Step Deployment

### 1. Sign Up/Login to Streamlit Cloud
1. Go to https://share.streamlit.io/
2. Click "Sign in with GitHub"
3. Authorize Streamlit to access your GitHub account

### 2. Create New App

1. Click **"New app"** button
2. Fill in the deployment form:

   **Repository:**
   ```
   VictorIdakwo/seismic_predict
   ```

   **Branch:**
   ```
   main
   ```

   **Main file path:**
   ```
   seismic_app.py
   ```

   **App URL (optional):**
   ```
   seismic-predict
   ```
   (This will create: https://seismic-predict.streamlit.app)

3. Click **"Deploy!"**

### 3. Wait for Deployment

The deployment process takes 2-5 minutes. Streamlit Cloud will:
- Clone your repository
- Install dependencies from `requirements.txt`
- Install system packages from `packages.txt`
- Start the Streamlit server

### 4. Access Your App

Once deployed, your app will be available at:
```
https://[your-app-name].streamlit.app
```

Example:
```
https://seismic-predict.streamlit.app
```

## 📝 Required Files (Already Created)

✅ **seismic_app.py** - Main application file
✅ **requirements.txt** - Python dependencies
✅ **packages.txt** - System-level dependencies
✅ **stations.csv** - Station metadata
✅ **magnitude_model.joblib** - Trained ML model
✅ **.streamlit/config.toml** - Streamlit configuration

## 🔧 Configuration Files

### requirements.txt
```
streamlit==1.40.0
plotly==5.24.1
obspy==1.4.1
numpy==1.26.4
pandas==2.2.3
matplotlib==3.9.2
scikit-learn==1.5.2
joblib==1.4.2
scipy==1.14.1
tqdm==4.67.1
```

### packages.txt (System dependencies)
```
libgomp1
```

### .streamlit/config.toml (App settings)
```toml
[server]
headless = true
port = 8501
maxUploadSize = 200
```

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution:** Ensure all dependencies are in `requirements.txt` with correct versions

### Issue: "File not found" errors
**Solution:** All file paths use relative paths (already fixed in the code)

### Issue: "Memory limit exceeded"
**Solution:** 
- Free tier has 1GB RAM limit
- Reduce model size or simplify features
- Process data in chunks

### Issue: Slow loading
**Solution:**
- Use `@st.cache_data` for data loading (already implemented)
- Use `@st.cache_resource` for model loading (already implemented)

### Issue: ObsPy installation fails
**Solution:** The `packages.txt` file should handle this

## 🔄 Updating Your Deployed App

When you push changes to GitHub, the app auto-updates:

```bash
# Make changes to your code
git add .
git commit -m "Update app features"
git push origin main
```

The app will automatically redeploy within 1-2 minutes.

## 🎯 Manual Reboot

If your app has issues:
1. Go to https://share.streamlit.io/
2. Find your app
3. Click the menu (⋮)
4. Select "Reboot app"

## 📊 App Management

### View Logs
1. Go to your app's dashboard
2. Click "Manage app"
3. View logs for debugging

### App Settings
- **App URL**: Custom subdomain
- **Python version**: 3.9-3.11
- **Secrets**: Store API keys securely

## 🔒 Adding Secrets (Optional)

If you need to store sensitive data:

1. Go to app settings
2. Click "Secrets"
3. Add in TOML format:
```toml
[passwords]
admin = "your_password_here"

[api_keys]
service = "your_api_key_here"
```

Access in code:
```python
import streamlit as st
password = st.secrets["passwords"]["admin"]
```

## 📈 Usage Limits (Free Tier)

- **Resources**: 1 GB RAM, 1 CPU
- **App limit**: 1 private app, unlimited public apps
- **Usage**: Community Cloud is free for public apps

## 🎉 Success Checklist

✅ Repository is public on GitHub
✅ All required files are committed
✅ File paths are relative (not absolute)
✅ requirements.txt has correct dependencies
✅ Model file (magnitude_model.joblib) is included
✅ Station data (stations.csv) is included
✅ App deployed successfully
✅ Can upload .mseed files
✅ Predictions work correctly

## 🌐 Share Your App

Once deployed, share your app URL:
```
https://seismic-predict.streamlit.app
```

Add it to your GitHub README:
```markdown
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://seismic-predict.streamlit.app)
```

## 📞 Support

- **Streamlit Docs**: https://docs.streamlit.io/streamlit-community-cloud
- **Community Forum**: https://discuss.streamlit.io/
- **GitHub Issues**: https://github.com/streamlit/streamlit/issues

---

**Your app is now live! 🎊**

Users can upload seismic data and get instant magnitude predictions! 🌍
