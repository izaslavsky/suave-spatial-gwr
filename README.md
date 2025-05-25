# 📊 SuAVE Spatial Statistics

This Streamlit app performs spatial analysis on georeferenced SuAVE surveys, including Geographically-Weighted Regression (GWR), residual analysis, and spatial autocorrelation.

---

## 🔧 Features

- Geographically-Weighted Regression with variable bandwidth selection
- Residual and fitted value computation
- Global and Local Moran’s I statistics
- Interactive mapping with `folium`
- Optional upload of derived variables to SuAVE
- Annotated coefficients with `#number` tags for compatibility

---

## 🚀 Getting Started

### Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run spatial_statistics.py
```

---

## 🔗 URL Parameters

Pass the following via URL to load the correct dataset:

- `user`: SuAVE username
- `csv`: CSV filename hosted on SuAVE
- `surveyurl`: Full SuAVE survey URL
- `dzc`: *(Optional)* Deep Zoom config file

**Example:**

```
https://your-app-url/spatial_statistics?user=suavedemos&csv=suavedemos_map.csv&surveyurl=https://suave-net.sdsc.edu/main/file=suavedemos_map.csv
```

---

## 🧾 Output and Upload

After GWR runs, you can:

- Download coefficients and residuals
- Select which derived variables to publish:
  - `residual#number`
  - `local_I#number`
  - `coef_<variable>#number`

The app uploads these variables along with your original dataset to SuAVE.
