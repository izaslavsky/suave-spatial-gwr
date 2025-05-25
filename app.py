import streamlit as st
st.set_page_config(page_title="Spatial Statistics", layout="wide")

# ---- Import dependencies ----
import os
import sys
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import requests
import io
from urllib.parse import urlencode, urlparse
from datetime import datetime
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from shapely.geometry import Point
from libpysal.weights import Queen, KNN
from esda.moran import Moran, Moran_Local
import folium
import streamlit_folium
import json
import branca.colormap as cm

# ---- Ensure uploader import path ----
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from suave_uploader import upload_to_suave

# ---- Read query params from URL ----
query_params = st.query_params
user = query_params.get("user", None)
csv_filename = query_params.get("csv", None)
survey_url = query_params.get("surveyurl", None)
dzc_file = query_params.get("dzc", None)


# ---- Page Title and Description ----
st.title("📊 Spatial Statistics")
st.markdown("**Geographically-Weighted Regression with Residuals and Autocorrelation Measures.**")
st.markdown("For polygon data, GWR uses Queen Weights; for point data, it uses 5-nearest-neighbors weights.")

# ---- Collapsible Diagnostics ----
with st.expander("⚙️ Diagnostics and Input Info", expanded=False):
    st.markdown(f"🧪 <span style='font-size: 0.85em;'>**Streamlit version:** {st.__version__}</span>", unsafe_allow_html=True)
    st.markdown(f"👤 <span style='font-size: 0.85em;'>**User:** {user}</span>", unsafe_allow_html=True)
    st.markdown(f"📂 <span style='font-size: 0.85em;'>**CSV File:** {csv_filename}</span>", unsafe_allow_html=True)

    if not csv_filename or not survey_url:
        st.error("❌ Missing CSV filename or survey URL. Use ?csv=...&surveyurl=... in the URL.")
        st.stop()

    parsed = urlparse(survey_url)
    base_url = f"{parsed.scheme}://{parsed.netloc}/surveys/"
    csv_url = base_url + csv_filename
    st.markdown(f"🔗 <span style='font-size: 0.85em;'>Trying URL: {csv_url}</span>", unsafe_allow_html=True)

    try:
        response = requests.get(csv_url)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
    except Exception as e:
        st.error(f"❌ Could not fetch CSV: {e}")
        st.stop()

    st.markdown("<span style='font-size: 0.9em;'>📋 Column Check</span>", unsafe_allow_html=True)
    df.columns = df.columns.str.strip()
    st.write(df.columns.tolist())
    st.write(df.dtypes.head())
    st.write(df.head(2))


# ---- Geometry Detection ----
gdf = None
geometry_col = next((col for col in df.columns if "geometry" in col.lower()), None)
lat_col = next((col for col in df.columns if "latitude" in col.lower()), None)
lon_col = next((col for col in df.columns if "longitude" in col.lower()), None)

if geometry_col:
    try:
        gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df[geometry_col]), crs="EPSG:4326")
    except Exception as e:
        st.error(f"❌ Failed to convert WKT to geometry: {e}")
        st.stop()
elif lat_col and lon_col:
    try:
        df = df.dropna(subset=[lat_col, lon_col])
        df['geometry'] = df.apply(lambda row: Point(row[lon_col], row[lat_col]), axis=1)
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    except Exception as e:
        st.error(f"❌ Failed to construct Point geometry: {e}")
        st.stop()
else:
    st.error("❌ Could not detect valid geometry. Ensure a WKT column or Latitude/Longitude columns are present.")
    st.stop()

# ---- Scroll Hint ----
st.markdown("<div style='font-size:0.9em; margin-bottom: -1em;'>⬇️ <i>Scroll down for analysis and model output...</i></div>", unsafe_allow_html=True)

# ---- Initialize Session State ----
if "gwr_results" not in st.session_state:
    st.session_state.gwr_results = None
    st.session_state.gwr_df = None
    st.session_state.coords = None
    st.session_state.bw = None
    st.session_state.dependent_var = None
    st.session_state.independent_vars = None
    st.session_state.residuals_dropped = 0

# ---- Map Display ----
st.subheader("🌍 Map of Features")
try:
    centroid = gdf.geometry.centroid.unary_union.centroid
    center = [centroid.y, centroid.x]
except:
    center = [0, 0]

m = folium.Map(location=center, zoom_start=4, tiles="CartoDB positron")

if gdf.geometry.geom_type.iloc[0] == 'Point':
    for _, row in gdf.iterrows():
        if not row.geometry.is_empty:
            tooltip_parts = [f"Lat: {row.geometry.y:.2f}, Lon: {row.geometry.x:.2f}"]
            for key in df.columns:
                if "#name" in key.lower():
                    tooltip_parts.append(f"Name: {row[key]}")
                elif "#img" in key.lower():
                    tooltip_parts.append(f"<img src='{row[key]}' width='100'>")
                elif "#href" in key.lower():
                    tooltip_parts.append(f"<a href='{row[key]}' target='_blank'>Link</a>")
            tooltip_html = "<br>".join(tooltip_parts)
            folium.Marker(
                location=[row.geometry.y, row.geometry.x],
                icon=folium.Icon(color='blue', icon='info-sign'),
                tooltip=folium.Tooltip(tooltip_html, sticky=True)
            ).add_to(m)
else:
    folium.GeoJson(
        json.loads(gdf.to_crs("EPSG:4326").to_json()),
        tooltip=folium.GeoJsonTooltip(fields=gdf.select_dtypes(include='object').columns[:4].tolist())
    ).add_to(m)

streamlit_folium.st_folium(m, width=800, height=500)


# ---- GWR variable selection ----
st.markdown("---")
numeric_cols = gdf.select_dtypes(include='number').columns.tolist()
dependent_var = st.selectbox("📌 Select Dependent Variable", numeric_cols)
independent_vars = st.multiselect("📈 Select Independent Variables", numeric_cols, default=numeric_cols[:2])

# ---- Run GWR and save session state ----
if st.button("▶️ Run GWR") and dependent_var and independent_vars:
    gwr_df = gdf[[dependent_var] + independent_vars + ["geometry"]].copy()
    n_before = len(gwr_df)
    gwr_df = gwr_df.dropna()
    n_after = len(gwr_df)
    st.session_state.residuals_dropped = n_before - n_after
    st.markdown(f"✅ Prepared {n_after} rows for GWR. Dropped {n_before - n_after} rows with missing values.")

    coords = list(zip(gwr_df.geometry.centroid.x, gwr_df.geometry.centroid.y))
    y = gwr_df[[dependent_var]].values
    X = gwr_df[independent_vars].values

    bw = Sel_BW(coords, y, X).search()
    gwr_model = GWR(coords, y, X, bw=bw)
    gwr_results = gwr_model.fit()

    # Pre-compute residuals and local Moran's I before saving
    gwr_df = gwr_df.copy()
    gwr_df["residual"] = gwr_results.resid_response.flatten()

    try:
        if gwr_df.geometry.geom_type.iloc[0] == 'Point':
            w = KNN.from_dataframe(gwr_df, k=5)
        else:
            w = Queen.from_dataframe(gwr_df)
        w.transform = 'r'
        moran_loc = Moran_Local(gwr_df["residual"], w)
        gwr_df["local_I"] = moran_loc.Is
    except Exception as e:
        st.warning(f"⚠️ Could not compute Local Moran's I: {e}")

    # Store results
    st.session_state.gwr_results = gwr_results
    st.session_state.gwr_df = gwr_df
    st.session_state.coords = coords
    st.session_state.bw = bw
    st.session_state.dependent_var = dependent_var
    st.session_state.independent_vars = independent_vars


# ---- Display GWR results ----
if st.session_state.gwr_results is not None:
    gwr_results = st.session_state.gwr_results
    gwr_df = st.session_state.gwr_df
    coords = st.session_state.coords
    bw = st.session_state.bw
    dependent_var = st.session_state.dependent_var
    independent_vars = st.session_state.independent_vars

    st.success(f"✅ Bandwidth selected: {bw}")
    st.write("R² Score:", gwr_results.R2)

    # Add coefficients to DataFrame
    coeff_cols = ['Intercept'] + independent_vars
    coeff_df = pd.DataFrame(gwr_results.params, columns=coeff_cols)
    coeff_df.index = gwr_df.index
    for col in coeff_df.columns:
        coeff_col_name = f"{col}#number"
        gwr_df[coeff_col_name] = coeff_df[col]

    # Add residuals
    gwr_df["residual#number"] = gwr_results.resid_response.flatten()
    gwr_df["fitted#number"] = gwr_results.predy.flatten()

    # Store for upload
    st.session_state.modified_df = pd.merge(
        df.copy(),
        gwr_df[[f"{col}#number" for col in coeff_df.columns] + ["residual#number", "fitted#number"]],
        left_index=True,
        right_index=True,
        how="left"
    )
    st.session_state.last_new_var = "GWR output"

    # Show table
    st.subheader("📋 GWR Coefficient Summary")
    st.dataframe(coeff_df.head())

    # Download buttons
    coeff_csv = io.StringIO()
    coeff_df.to_csv(coeff_csv, index=False)
    st.download_button("⬇️ Download Coefficients", data=coeff_csv.getvalue(), file_name="gwr_coefficients.csv", mime="text/csv")

    residuals_csv = io.StringIO()
    gwr_df[["residual#number", "fitted#number"]].to_csv(residuals_csv, index=False)
    st.download_button("⬇️ Download Residuals & Fitted Values", data=residuals_csv.getvalue(), file_name="gwr_residuals.csv", mime="text/csv")


    # ---- Residual Choropleth Map ----
    st.subheader("🗺️ Residuals Choropleth Map")
    res_map = folium.Map(location=center, zoom_start=4, tiles="CartoDB positron")

    if gwr_df.geometry.geom_type.iloc[0] == 'Point':
        colormap = cm.linear.RdYlBu_11.scale(gwr_df["residual#number"].min(), gwr_df["residual#number"].max())
        for _, row in gwr_df.iterrows():
            color = colormap(row["residual#number"])
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=5,
                fill=True,
                fill_opacity=0.8,
                color=color,
                fill_color=color,
                tooltip=f"Residual: {row['residual#number']:.3f}"
            ).add_to(res_map)
        colormap.caption = "Residuals"
        colormap.add_to(res_map)
    else:
        folium.Choropleth(
            geo_data=json.loads(gwr_df.to_crs("EPSG:4326").to_json()),
            data=gwr_df.reset_index(),
            columns=["index", "residual#number"],
            key_on="feature.id",
            fill_color="RdYlBu",
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name="Residuals"
        ).add_to(res_map)

    streamlit_folium.st_folium(res_map, width=800, height=500)

    # ---- Global and Local Moran's I ----
    st.subheader("🧪 Global Moran's I")
    try:
        if isinstance(gwr_df, gpd.GeoDataFrame) and gwr_df.geometry.geom_type.iloc[0] == 'Point':
            w = KNN.from_dataframe(gwr_df, k=5)
        else:
            w = Queen.from_dataframe(gwr_df)
        w.transform = 'r'
        moran = Moran(gwr_df["residual#number"], w)
        st.write(f"Moran's I: {moran.I:.4f}, p-value: {moran.p_sim:.4f}")

        # Local Moran
        st.subheader("🧭 Local Moran's I Map")
        moran_loc = Moran_Local(gwr_df["residual#number"], w)
        gwr_df["local_I#number"] = moran_loc.Is

        local_map = folium.Map(location=center, zoom_start=4, tiles="CartoDB positron")
        if gwr_df.geometry.geom_type.iloc[0] == 'Point':
            colormap = cm.linear.PuOr_11.scale(gwr_df["local_I#number"].min(), gwr_df["local_I#number"].max())
            for _, row in gwr_df.iterrows():
                color = colormap(row["local_I#number"])
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=5,
                    fill=True,
                    fill_opacity=0.8,
                    color=color,
                    fill_color=color,
                    tooltip=f"Local I: {row['local_I#number']:.3f}"
                ).add_to(local_map)
            colormap.caption = "Local Moran's I"
            colormap.add_to(local_map)
        else:
            folium.Choropleth(
                geo_data=json.loads(gwr_df.to_crs("EPSG:4326").to_json()),
                data=gwr_df.reset_index(),
                columns=["index", "local_I#number"],
                key_on="feature.id",
                fill_color="PuOr",
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name="Local Moran's I"
            ).add_to(local_map)

        streamlit_folium.st_folium(local_map, width=800, height=500)
    except Exception as e:
        st.error(f"❌ Moran's I computation failed: {e}")

# ---- SuAVE Upload (Optional) ----
st.markdown("---")
st.subheader("📤 Publish GWR Results to SuAVE")

if st.session_state.gwr_results is not None:
    from suave_uploader import upload_to_suave

    st.markdown(
        "The following variables are derived from GWR and can be added to a new survey:\n"
        "- **residual#number**: Difference between observed and predicted values\n"
        "- **local_I#number**: Local Moran's I statistic (spatial autocorrelation)\n"
        "- **coef_<var>#number**: Local regression coefficient for `<var>` estimated by GWR"
    )

    possible_vars = []
    gwr_df = st.session_state.gwr_df

    # ---- Add coefficients
    coeff_cols = ['Intercept'] + st.session_state.independent_vars
    coeff_df = pd.DataFrame(st.session_state.gwr_results.params, columns=coeff_cols)
    coeff_df.index = gwr_df.index

    for col in coeff_df.columns:
        if col == "Intercept":
            name = "Intercept#number"
        else:
            name = f"coef_{col}#number"
        if name not in gwr_df.columns:
            gwr_df[name] = coeff_df[col]
        possible_vars.append(name)

    # ---- Add residuals and local Moran’s I
    if "residual" in gwr_df.columns:
        res_name = "residual#number"
        if res_name not in gwr_df.columns:
            gwr_df[res_name] = gwr_df["residual"]
        possible_vars.append(res_name)

    if "local_I" in gwr_df.columns:
        local_name = "local_I#number"
        if local_name not in gwr_df.columns:
            gwr_df[local_name] = gwr_df["local_I"]
        possible_vars.append(local_name)

    # ---- User Selection
    selected_vars = st.multiselect(
        "🧠 Select GWR-derived variables to include",
        sorted(set(possible_vars)),
        default=sorted(set(possible_vars))
    )

    auth_user = st.text_input("🔐 SuAVE Login:")
    auth_pass = st.text_input("🔑 SuAVE Password:", type="password")
    base_name = csv_filename.replace(".csv", "").split("_", 1)[-1]
    suggested_name = f"{base_name}_GWR_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    survey_name = st.text_input("📛 New Survey Name", value=suggested_name)

    if st.button("📦 Upload to SuAVE"):
        if not auth_user or not auth_pass or not survey_name:
            st.warning("⚠️ Please fill in all required fields.")
        else:
            # Append selected GWR variables to original df
            df_with_gwr = df.copy()
            for var in selected_vars:
                if var in gwr_df.columns:
                    df_with_gwr[var] = gwr_df[var]

            st.session_state.modified_df = df_with_gwr

            parsed = urlparse(survey_url)
            referer = survey_url.split("/main")[0] + "/"

            success, message, new_url = upload_to_suave(
                df_with_gwr,
                survey_name,
                auth_user,
                auth_pass,
                referer,
                dzc_file=query_params.get("dzc", None)
            )

            if success:
                st.success(message)
                st.markdown(f"🔗 [Open New Survey in SuAVE]({new_url})")
            else:
                st.error(f"❌ {message}")



# ---- Return to Home button ----
param_str = urlencode({k: v[0] if isinstance(v, list) else v for k, v in query_params.items()})
button_css = """
<style>
.back-button {
    display: inline-block;
    padding: 0.6em 1.2em;
    margin-top: 2em;
    font-size: 1.1em;
    font-weight: bold;
    color: white !important;
    background-color: #1f77b4;
    border: none;
    border-radius: 8px;
    text-decoration: none;
}
.back-button:hover {
    background-color: #16699b;
    color: white !important;
}
</style>
"""
st.markdown(button_css, unsafe_allow_html=True)
st.markdown(f'<a href="/?{param_str}" class="back-button">⬅️ Return to Home</a>', unsafe_allow_html=True)
