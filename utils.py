
# utils.py
import io, json, sqlite3, tempfile, os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ---------- Rwanda theme ----------
RW_COLORS = {
    "blue":   "#00A1DE",
    "yellow": "#FAD201",
    "green":  "#20603D",
    "sun":    "#E5BE01",
    "ink":    "#0B1B2B",
}
COLORWAY = [RW_COLORS["blue"], RW_COLORS["yellow"], RW_COLORS["green"], RW_COLORS["sun"], "#005EB8"]
RWA_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        colorway=COLORWAY,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, Arial, Helvetica, sans-serif", color=RW_COLORS["ink"]),
        title=dict(x=0.02, xanchor="left"),
        xaxis=dict(gridcolor="rgba(0,0,0,0.05)", zerolinecolor="rgba(0,0,0,0.1)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0.05)", zerolinecolor="rgba(0,0,0,0.1)"),
        legend=dict(orientation="h", y=1.02, x=0.02),
    )
)

CSS = f"""
<style>
  .main-header {{
    font-family: 'Arial Black', sans-serif; color: white; text-align: center;
    padding: 18px 20px;
    background: linear-gradient(90deg, {RW_COLORS['blue']} 0%, {RW_COLORS['yellow']} 50%, {RW_COLORS['green']} 100%);
    border-radius: 14px; margin-bottom: 20px; box-shadow: 0 6px 14px rgba(0,0,0,0.12);
  }}
  .stButton>button {{
    border-radius: 12px; padding: 0.6rem 1rem; background: {RW_COLORS['blue']};
    color: white; font-weight: 700; border: none; box-shadow: 0 1px 0 rgba(0,0,0,0.05);
  }}
  .stButton>button:hover {{ background: {RW_COLORS['green']}; }}
  .stCard {{
    border-radius: 16px; border: 2px solid rgba(0,0,0,0.06);
    box-shadow: 0 6px 12px rgba(0,0,0,0.08);
    padding: 16px; margin-bottom: 20px; background-color: #ffffff;
  }}
  .metric-value {{ font-size: 2.2rem; font-weight: 800; color:{RW_COLORS['ink']}; }}
  .metric-label {{ font-size: 0.9rem; color:#334155; letter-spacing: .3px; }}
</style>
"""

# ---------- Geofence: keep all points in Rwanda ----------
RW_BBOX = dict(lat_min=-2.84, lat_max=-1.04, lon_min=28.85, lon_max=30.90)

def enforce_rwanda_bounds(df: pd.DataFrame, lat_col="latitude", lon_col="longitude", strategy="clip", jitter=True, jitter_deg=0.0008):
    """Clamp ('clip') or drop coordinates to Rwanda bbox; optional tiny jitter prevents stacking on edges."""
    if lat_col not in df.columns or lon_col not in df.columns:
        return df
    df = df.copy()
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    mask = df[lat_col].between(RW_BBOX["lat_min"], RW_BBOX["lat_max"]) & df[lon_col].between(RW_BBOX["lon_min"], RW_BBOX["lon_max"])
    if strategy == "drop":
        df = df[mask].copy()
    else:  # clip
        df[lat_col] = df[lat_col].clip(RW_BBOX["lat_min"], RW_BBOX["lat_max"])
        df[lon_col] = df[lon_col].clip(RW_BBOX["lon_min"], RW_BBOX["lon_max"])
    if jitter and len(df) > 0:
        rng = np.random.default_rng(2025)
        jlat = (rng.random(len(df)) - 0.5) * 2 * jitter_deg
        jlon = (rng.random(len(df)) - 0.5) * 2 * jitter_deg
        df[lat_col] = np.clip(df[lat_col] + jlat, RW_BBOX["lat_min"], RW_BBOX["lat_max"])
        df[lon_col] = np.clip(df[lon_col] + jlon, RW_BBOX["lon_min"], RW_BBOX["lon_max"])
    return df

# ---------- Data IO ----------
def read_input_streamlit(st, source: str) -> pd.DataFrame | None:
    if source == "CSV":
        up = st.file_uploader("📤 Upload accident CSV file", type=["csv"])
        if not up: return None
        return pd.read_csv(up)
    if source == "JSON":
        up = st.file_uploader("📤 Upload accident JSON (array/object) or JSONL", type=["json","jsonl","ndjson"])
        if not up: return None
        txt = up.read().decode("utf-8", errors="ignore")
        try:
            return pd.read_json(io.StringIO(txt), lines=True)
        except ValueError:
            obj = json.loads(txt)
            if isinstance(obj, list):  return pd.json_normalize(obj)
            if isinstance(obj, dict):  return pd.json_normalize([obj])
            st.error("Unsupported JSON structure."); return None
    # SQLite
    up = st.file_uploader("📤 Upload SQLite database (.db / .sqlite)", type=["db","sqlite","sqlite3"])
    if not up: return None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.write(up.getbuffer()); tmp.flush()
    with sqlite3.connect(tmp.name) as con:
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';", con)["name"].tolist()
        if not tables: st.error("No user tables found."); return None
        tbl = st.selectbox("Select table", tables)
        return pd.read_sql(f"SELECT * FROM '{tbl}'", con)

# ---------- Preprocess ----------
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["created"] = pd.to_datetime(df.get("created"), errors="coerce")
    df = df.dropna(subset=["created"]).reset_index(drop=True)
    df["hour"] = df["created"].dt.hour
    df["dayofweek"] = df["created"].dt.dayofweek
    df["description"] = df.get("description", "").fillna("").astype(str)

    # severity → serious(0/1)
    if "severity" in df.columns:
        sev = df["severity"].astype(str).str.lower()
        df["serious"] = sev.str.contains("high|serious|severe").astype(int)
    else:
        df["severity"] = "Unknown"
        df["serious"] = 0

    # weather encoding (simple)
    if "weather" in df.columns:
        from sklearn.preprocessing import LabelEncoder
        df["weather_encoded"] = LabelEncoder().fit_transform(df["weather"].astype(str))
    else:
        df["weather"] = "Unknown"
        df["weather_encoded"] = 0

    # vehicle count
    plates = [c for c in ["vehicle_plate_1","vehicle_plate_2","vehicle_plate_3"] if c in df.columns]
    df["vehicle_count"] = df[plates].notnull().sum(axis=1) if plates else 0

    # location fallbacks
    if "location_province" not in df: df["location_province"] = "Unknown"
    if "location_name" not in df:     df["location_name"] = "Unknown"

    # ensure numeric fields exist
    for c in ["latitude","longitude","injured_victims","crash_victims","hour","dayofweek","vehicle_count","weather_encoded"]:
        if c not in df.columns: df[c] = 0
    df[["latitude","longitude","injured_victims","crash_victims","hour","dayofweek","vehicle_count","weather_encoded"]] =             df[["latitude","longitude","injured_victims","crash_victims","hour","dayofweek","vehicle_count","weather_encoded"]].apply(pd.to_numeric, errors="coerce").fillna(0)

    # keep all points inside Rwanda
    df = enforce_rwanda_bounds(df, strategy="clip", jitter=True)
    return df

# ---------- CI helpers ----------
try:
    from scipy import stats as _scistats
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def poisson_ci_95(k):
    k = np.asarray(k, dtype=float)
    if _HAS_SCIPY:
        lo, hi = _scistats.poisson.interval(0.95, k)
        return lo, hi
    # normal approx
    z = 1.96
    se = np.sqrt(np.maximum(k, 1e-9))
    lo = np.clip(k - z*se, 0, None)
    hi = k + z*se
    return lo, hi

def bar_with_ci(x, y, title, xlab, ylab, template=RWA_TEMPLATE):
    lo, hi = poisson_ci_95(y)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=y, error_y=dict(type='data', array=(hi - y), arrayminus=(y - lo), thickness=1.2)))
    fig.update_layout(title=title, xaxis_title=xlab, yaxis_title=ylab, template=template)
    return fig
