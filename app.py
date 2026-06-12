import streamlit as st
st.set_page_config(page_title="Spatial Statistics", layout="wide")

import os
import sys
import io
import json
from datetime import datetime
from urllib.parse import urlencode, urlparse

import pandas as pd
import geopandas as gpd
import requests
import matplotlib.pyplot as plt
import folium
import streamlit_folium
import branca.colormap as cm
from shapely.geometry import Point
from libpysal.weights import Queen, KNN
from esda.moran import Moran, Moran_Local
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from suave_uploader import upload_to_suave

# ── Query params ─────────────────────────────────────────────────────────────
query_params = st.query_params
user         = query_params.get("user", None)
csv_filename = query_params.get("csv", None)
survey_url   = query_params.get("surveyurl", None)
dzc_file     = query_params.get("dzc", None)

st.title("📊 Spatial Statistics")
st.markdown("**Geographically-Weighted Regression with Residuals and Autocorrelation Measures.**")
st.markdown("For polygon data GWR uses Queen weights; for point data it uses 5-nearest-neighbor weights.")

# ── Diagnostics ───────────────────────────────────────────────────────────────
with st.expander("⚙️ Diagnostics and Input Info", expanded=False):
    st.markdown(f"🧪 **Streamlit version:** {st.__version__}")
    st.markdown(f"👤 **User:** {user}  📂 **CSV:** {csv_filename}")
    if not csv_filename or not survey_url:
        st.error("❌ Missing CSV filename or survey URL.")
        st.stop()

# ── Cached data loading and geometry construction ─────────────────────────────
@st.cache_data(show_spinner="Loading survey data…")
def load_geodataframe(csv_url: str):
    resp = requests.get(csv_url, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = df.columns.str.strip()

    geometry_col = next((c for c in df.columns if "geometry" in c.lower()), None)
    lat_col      = next((c for c in df.columns if "latitude"  in c.lower()), None)
    lon_col      = next((c for c in df.columns if "longitude" in c.lower()), None)

    if geometry_col:
        gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df[geometry_col]),
                               crs="EPSG:4326")
    elif lat_col and lon_col:
        df = df.dropna(subset=[lat_col, lon_col]).copy()
        df["geometry"] = df.apply(lambda r: Point(r[lon_col], r[lat_col]), axis=1)
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    else:
        return None, df, "No geometry detected. Survey needs a WKT geometry column or Latitude/Longitude columns."

    return gdf, df, None

parsed   = urlparse(survey_url)
csv_url  = f"{parsed.scheme}://{parsed.netloc}/surveys/{csv_filename}"
gdf, df, geo_err = load_geodataframe(csv_url)

if geo_err:
    st.error(f"❌ {geo_err}")
    st.stop()

_geom_type = gdf.geometry.geom_type.iloc[0]

# ── Cached feature map ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Building map…")
def make_feature_map(geojson_str: str, geom_type: str, name_cols: list):
    """Build the initial features map once; cache keyed on the raw GeoJSON string."""
    data = json.loads(geojson_str)
    try:
        import shapely.geometry as sg
        pts = [sg.shape(f["geometry"]).centroid for f in data["features"] if f["geometry"]]
        cx = sum(p.x for p in pts) / len(pts)
        cy = sum(p.y for p in pts) / len(pts)
        center = [cy, cx]
    except Exception:
        center = [0, 0]

    m = folium.Map(location=center, zoom_start=4, tiles="CartoDB positron")

    if geom_type == "Point":
        # GeoJson is far cheaper than one Marker object per row
        folium.GeoJson(
            data,
            name="Features",
            marker=folium.CircleMarker(radius=4, fill=True, fill_opacity=0.7,
                                       color="#1f77b4", fill_color="#1f77b4"),
            tooltip=folium.GeoJsonTooltip(fields=name_cols[:3]) if name_cols else None,
        ).add_to(m)
    else:
        obj_cols = [f["properties"] for f in data["features"][:1]]
        tip_fields = list(obj_cols[0].keys())[:4] if obj_cols else []
        folium.GeoJson(
            data,
            tooltip=folium.GeoJsonTooltip(fields=tip_fields) if tip_fields else None,
        ).add_to(m)

    return m, center

_geojson_str = gdf.to_crs("EPSG:4326").to_json()
_name_cols   = [c for c in df.columns if "#name" in c.lower() or "#href" in c.lower()]
feat_map, _center = make_feature_map(_geojson_str, _geom_type, _name_cols)

st.subheader("🌍 Map of Features")
st.markdown("⬇️ *Scroll down for variable selection and model output.*")
streamlit_folium.st_folium(feat_map, width=800, height=500)

# ── Variable selection ────────────────────────────────────────────────────────
st.markdown("---")
numeric_cols = gdf.select_dtypes(include="number").columns.tolist()
dependent_var   = st.selectbox("📌 Dependent Variable", numeric_cols)
independent_vars = st.multiselect("📈 Independent Variables", numeric_cols,
                                   default=numeric_cols[:min(2, len(numeric_cols))])

# ── Session state initialisation ──────────────────────────────────────────────
for key in ("gwr_results", "gwr_df", "bw", "dependent_var", "independent_vars",
            "moran_I", "moran_p", "local_I_arr", "center"):
    if key not in st.session_state:
        st.session_state[key] = None

# ── Run GWR ───────────────────────────────────────────────────────────────────
if st.button("▶️ Run GWR") and dependent_var and independent_vars:
    with st.spinner("Fitting GWR model…"):
        gwr_df = gdf[[dependent_var] + independent_vars + ["geometry"]].dropna().copy()
        st.markdown(f"✅ {len(gwr_df)} rows after dropping missing values.")

        coords = list(zip(gwr_df.geometry.centroid.x, gwr_df.geometry.centroid.y))
        y = gwr_df[[dependent_var]].values
        X = gwr_df[independent_vars].values

        bw          = Sel_BW(coords, y, X).search()
        gwr_results = GWR(coords, y, X, bw=bw).fit()

        gwr_df["residual#number"] = gwr_results.resid_response.flatten()
        gwr_df["fitted#number"]   = gwr_results.predy.flatten()

        coeff_cols = ["Intercept"] + independent_vars
        coeff_df   = pd.DataFrame(gwr_results.params, columns=coeff_cols, index=gwr_df.index)
        for col in coeff_cols:
            gwr_df[f"{col}#number"] = coeff_df[col]

        # Compute Moran's I once here; reuse in display — no recomputation on reruns
        try:
            w = KNN.from_dataframe(gwr_df, k=5) if _geom_type == "Point" \
                else Queen.from_dataframe(gwr_df)
            w.transform = "r"
            moran     = Moran(gwr_df["residual#number"], w)
            moran_loc = Moran_Local(gwr_df["residual#number"], w)
            st.session_state.moran_I       = moran.I
            st.session_state.moran_p       = moran.p_sim
            st.session_state.local_I_arr   = moran_loc.Is
            gwr_df["local_I#number"]       = moran_loc.Is
        except Exception as e:
            st.warning(f"⚠️ Could not compute Moran's I: {e}")

        st.session_state.gwr_results     = gwr_results
        st.session_state.gwr_df          = gwr_df
        st.session_state.bw              = bw
        st.session_state.dependent_var   = dependent_var
        st.session_state.independent_vars = independent_vars
        st.session_state.center          = _center

# ── Display results ───────────────────────────────────────────────────────────
if st.session_state.gwr_results is not None:
    gwr_results   = st.session_state.gwr_results
    gwr_df        = st.session_state.gwr_df
    bw            = st.session_state.bw
    dependent_var = st.session_state.dependent_var
    ind_vars      = st.session_state.independent_vars
    center        = st.session_state.center or _center
    coeff_cols    = ["Intercept"] + ind_vars

    st.success(f"✅ Bandwidth: {bw}   R²: {gwr_results.R2:.4f}")

    st.subheader("📋 GWR Coefficient Summary")
    coeff_df = pd.DataFrame(gwr_results.params, columns=coeff_cols, index=gwr_df.index)
    st.dataframe(coeff_df.head())

    col_a, col_b = st.columns(2)
    with col_a:
        coeff_csv = coeff_df.to_csv(index=False)
        st.download_button("⬇️ Coefficients CSV", coeff_csv,
                           "gwr_coefficients.csv", "text/csv")
    with col_b:
        resid_csv = gwr_df[["residual#number", "fitted#number"]].to_csv(index=False)
        st.download_button("⬇️ Residuals CSV", resid_csv,
                           "gwr_residuals.csv", "text/csv")

    # ── Residuals map ─────────────────────────────────────────────────────────
    @st.cache_data(show_spinner=False)
    def make_residual_map(geojson_str: str, geom_type: str, center: list):
        data = json.loads(geojson_str)
        res_vals = [f["properties"].get("residual#number") for f in data["features"]]
        res_vals = [v for v in res_vals if v is not None]
        rmin, rmax = min(res_vals), max(res_vals)

        m = folium.Map(location=center, zoom_start=4, tiles="CartoDB positron")
        cmap = cm.linear.RdYlBu_11.scale(rmin, rmax)
        if geom_type == "Point":
            for f in data["features"]:
                val = f["properties"].get("residual#number")
                if val is None:
                    continue
                coord = f["geometry"]["coordinates"]
                folium.CircleMarker(
                    location=[coord[1], coord[0]], radius=5,
                    fill=True, fill_opacity=0.8,
                    color=cmap(val), fill_color=cmap(val),
                    tooltip=f"Residual: {val:.3f}"
                ).add_to(m)
            cmap.caption = "Residuals"
            cmap.add_to(m)
        else:
            folium.Choropleth(
                geo_data=data, data=pd.DataFrame(
                    [(i, f["properties"]["residual#number"])
                     for i, f in enumerate(data["features"])
                     if "residual#number" in f["properties"]],
                    columns=["idx", "residual#number"]
                ),
                columns=["idx", "residual#number"], key_on="feature.id",
                fill_color="RdYlBu", fill_opacity=0.7, line_opacity=0.2,
                legend_name="Residuals"
            ).add_to(m)
        return m

    st.subheader("🗺️ Residuals Map")
    _res_geojson = gwr_df.to_crs("EPSG:4326").to_json()
    streamlit_folium.st_folium(
        make_residual_map(_res_geojson, _geom_type, center),
        width=800, height=500
    )

    # ── Global Moran's I (reuse stored result — no recomputation) ─────────────
    st.subheader("🧪 Global Moran's I on Residuals")
    if st.session_state.moran_I is not None:
        st.write(f"Moran's I: **{st.session_state.moran_I:.4f}**  "
                 f"p-value: **{st.session_state.moran_p:.4f}**")
    else:
        st.info("Moran's I could not be computed for this dataset.")

    # ── Local Moran's I map ───────────────────────────────────────────────────
    if st.session_state.local_I_arr is not None and "local_I#number" in gwr_df.columns:
        @st.cache_data(show_spinner=False)
        def make_lisa_map(geojson_str: str, geom_type: str, center: list):
            data   = json.loads(geojson_str)
            li_vals = [f["properties"].get("local_I#number") for f in data["features"]]
            li_vals = [v for v in li_vals if v is not None]
            lmin, lmax = min(li_vals), max(li_vals)

            m    = folium.Map(location=center, zoom_start=4, tiles="CartoDB positron")
            cmap = cm.linear.PuOr_11.scale(lmin, lmax)
            if geom_type == "Point":
                for f in data["features"]:
                    val   = f["properties"].get("local_I#number")
                    if val is None:
                        continue
                    coord = f["geometry"]["coordinates"]
                    folium.CircleMarker(
                        location=[coord[1], coord[0]], radius=5,
                        fill=True, fill_opacity=0.8,
                        color=cmap(val), fill_color=cmap(val),
                        tooltip=f"Local I: {val:.3f}"
                    ).add_to(m)
                cmap.caption = "Local Moran's I"
                cmap.add_to(m)
            else:
                folium.Choropleth(
                    geo_data=data, data=pd.DataFrame(
                        [(i, f["properties"]["local_I#number"])
                         for i, f in enumerate(data["features"])
                         if "local_I#number" in f["properties"]],
                        columns=["idx", "local_I#number"]
                    ),
                    columns=["idx", "local_I#number"], key_on="feature.id",
                    fill_color="PuOr", fill_opacity=0.7, line_opacity=0.2,
                    legend_name="Local Moran's I"
                ).add_to(m)
            return m

        st.subheader("🧭 Local Moran's I Map")
        streamlit_folium.st_folium(
            make_lisa_map(_res_geojson, _geom_type, center),
            width=800, height=500
        )

# ── Upload to SuAVE ───────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📤 Publish GWR Results to SuAVE")

if st.session_state.gwr_results is not None:
    gwr_df   = st.session_state.gwr_df
    ind_vars = st.session_state.independent_vars

    new_cols = ([f"Intercept#number"] +
                [f"{v}#number" for v in ind_vars] +
                ["residual#number", "fitted#number"] +
                (["local_I#number"] if "local_I#number" in gwr_df.columns else []))
    available = [c for c in new_cols if c in gwr_df.columns]

    st.markdown("GWR-derived columns available to publish:")
    for c in available:
        st.markdown(f"- `{c}`")

    selected_vars = st.multiselect("🧠 Select columns to include",
                                   available, default=available)
    auth_user   = st.text_input("🔐 SuAVE Login:")
    auth_pass   = st.text_input("🔑 SuAVE Password:", type="password")
    base_name   = csv_filename.replace(".csv", "").split("_", 1)[-1]
    survey_name = st.text_input("📛 New Survey Name",
                                value=f"{base_name}_GWR_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    if st.button("📦 Upload to SuAVE"):
        if not auth_user or not auth_pass or not survey_name:
            st.warning("⚠️ Fill in all fields.")
        else:
            df_out = df.copy()
            for var in selected_vars:
                if var in gwr_df.columns:
                    df_out[var] = gwr_df[var]
            referer = survey_url.split("/main")[0] + "/"
            success, message, new_url = upload_to_suave(
                df_out, survey_name, auth_user, auth_pass, referer,
                dzc_file=query_params.get("dzc", None)
            )
            if success:
                st.success(message)
                st.markdown(f"🔗 [Open New Survey in SuAVE]({new_url})")
            else:
                st.error(f"❌ {message}")

# ── Back button ───────────────────────────────────────────────────────────────
param_str    = urlencode({k: v[0] if isinstance(v, list) else v
                          for k, v in query_params.items()})
launcher_url = "https://suave-launcher.streamlit.app"
st.markdown(f"""
<style>
.back-button{{display:inline-block;padding:.6em 1.2em;margin-top:2em;font-size:1.1em;
font-weight:bold;color:white!important;background-color:#1f77b4;border:none;
border-radius:8px;text-decoration:none}}
.back-button:hover{{background-color:#16699b}}
</style>
<a href="{launcher_url}/?{param_str}" class="back-button">⬅️ Return to Home</a>
""", unsafe_allow_html=True)
