# descriptive.py
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pandas.api.types import is_datetime64_any_dtype
from utils import RWA_TEMPLATE, bar_with_ci

try:
    from scipy import stats
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ----------------------------- helpers ----------------------------------------
def _pearson_r_p(x, y):
    if not _HAS_SCIPY:
        return None, None
    try:
        r, p = stats.pearsonr(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        return float(r), float(p)
    except Exception:
        return None, None


def _apply_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    q = df[
        df["location_province"].isin(f["location_province"]) &
        df["location_name"].isin(f["location_name"])
    ]
    # Optional filters if present in f
    for key in ["vehicle_type", "road_type", "road_condition"]:
        if key in f and f.get(key) is not None:
            q = q[q[key].isin(f[key])]
    return q


def _detect_column(df: pd.DataFrame, keywords, alt_candidates=None):
    """Find the first column whose name contains ANY of the keywords (case-insensitive)."""
    key_l = [k.lower() for k in keywords]
    for c in df.columns:
        lc = str(c).lower()
        if any(k in lc for k in key_l):
            return c
    if alt_candidates:
        for c in alt_candidates:
            if c in df.columns:
                return c
    return None


def _collect_plate_series(df: pd.DataFrame) -> pd.Series | None:
    """
    Collects all plate-like columns into a single Series.
    Looks for columns containing any of these tokens in their name.
    """
    tokens = ["plate", "number_plate", "registration", "reg", "license", "licence"]
    plate_cols = [c for c in df.columns if any(tok in str(c).lower() for tok in tokens)]
    if not plate_cols:
        return None
    s = (
        df[plate_cols]
        .astype(str)
        .replace({"nan": None, "NaT": None, "": None, "NONE": None, "NAN": None}, regex=False)
        .stack(dropna=True)
    )
    s.name = "Plate"
    s.index = s.index.droplevel(-1)  # align to df row index for drilldown
    return s


def _mask_middle(s: str, keep=2, dot="•"):
    if not isinstance(s, str):
        s = str(s)
    if len(s) <= 2 * keep:
        return s
    return s[:keep] + (dot * (len(s) - 2 * keep)) + s[-keep:]


def _normalize_series(s: pd.Series, case=True, strip=True, collapse_ws=True):
    s = s.astype(str)
    if strip:
        s = s.str.strip()
    if case:
        s = s.str.upper()
    if collapse_ws:
        s = s.str.replace(r"\s+", " ", regex=True)
    s = s.replace({"": None, "NAN": None, "NONE": None, "NULL": None})
    return s.dropna()


def _freq_block(
    df: pd.DataFrame,
    value_series: pd.Series,
    title_icon: str,
    title_text: str,
    label: str,
    template=RWA_TEMPLATE,
    default_min_occ=3,
    default_top_k=20,
    anonymize=True,
    mask_keep=2,
    normalize=True,
    y_axis_title="Frequency",
    sortable_cols=("created", "location_province", "location_name", "severity", "vehicle_type"),
    show_text_pairs=False,
):
    """
    Generic frequency analysis block with controls, chart, table, CSV download and drilldown.
    `value_series` should be a Series aligned to df.
    """
    st.markdown("---")
    st.subheader(f"{title_icon} {title_text}")

    col_controls = st.columns([1, 1, 1, 1])
    with col_controls[0]:
        min_occ = st.number_input("Minimum occurrences", min_value=2, max_value=100, value=default_min_occ, step=1)
    with col_controls[1]:
        top_k = st.slider("Show top N", min_value=5, max_value=200, value=default_top_k, step=5)
    with col_controls[2]:
        do_mask = st.checkbox("Anonymize", value=anonymize, help=f"Mask middle characters of {label.lower()} for privacy.")
    with col_controls[3]:
        case_norm = st.checkbox("Normalize case/whitespace", value=normalize)

    s = value_series
    if case_norm:
        s = _normalize_series(s)
    else:
        s = s.astype(str).replace({"": None}).dropna()

    counts = s.value_counts()
    frequent = counts[counts >= min_occ].head(top_k).reset_index()
    frequent.columns = [label, "Crash Count"]

    if frequent.empty:
        st.warning(f"No {label.lower()} found with ≥ {min_occ} occurrences under current filters.")
        return

    show_df = frequent.copy()
    if do_mask:
        show_df[label] = show_df[label].map(lambda x: _mask_middle(str(x), keep=mask_keep))

    # Plot
    fig_top = px.bar(
        show_df,
        x=label, y="Crash Count",
        title=f"{title_text} (≥ {min_occ})",
    )
    fig_top.update_layout(template=template, xaxis_title=label, yaxis_title=y_axis_title)
    st.plotly_chart(fig_top, use_container_width=True)

    # Optional text pairs "ITEM: count"
    if show_text_pairs:
        as_pairs = "\n".join([f"{row[label]}: {int(row['Crash Count'])}" for _, row in show_df.iterrows()])
        st.text(as_pairs)

    # Table + Download
    st.dataframe(show_df, use_container_width=True)
    csv_bytes = frequent.to_csv(index=False).encode("utf-8")
    st.download_button(
        f"⬇️ Download frequent {label.lower()} (CSV)",
        data=csv_bytes,
        file_name=f"frequent_{label.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )

    # Drilldown
    with st.expander("🔎 Drilldown"):
        opts = frequent[label].tolist()
        pick = st.selectbox(f"Select a {label.lower()} to view its accidents", opts)
        # Use un-normalized matching if user disabled normalization; else match normalized
        if case_norm:
            base_series = _normalize_series(value_series).reindex(df.index)
            dd = df[base_series == pick]
        else:
            dd = df[value_series.astype(str) == str(pick)]

        st.write(f"Accidents for **{_mask_middle(pick, keep=mask_keep) if do_mask else pick}**: {len(dd)}")

        cols = [c for c in sortable_cols if c in dd.columns]
        if cols:
            sort_col = cols[0]
            show_cols = cols + [c for c in [value_series.name] if c not in cols]
            try:
                st.dataframe(dd[show_cols].sort_values(by=sort_col, ascending=False), use_container_width=True)
            except Exception:
                st.dataframe(dd[show_cols], use_container_width=True)
        else:
            st.dataframe(dd, use_container_width=True)


# ----------------------------- main UI ----------------------------------------
def render(df: pd.DataFrame, f: dict):
    st.title("📊 Descriptive Analytics")

    # --- SAFE datetime handling (timezone aware included) ---
    if "created" in df.columns and not is_datetime64_any_dtype(df["created"]):
        try:
            df["created"] = pd.to_datetime(df["created"], errors="coerce", utc=True)
        except Exception:
            pass

    df_descriptive = _apply_filters(df, f)
    if df_descriptive.empty:
        st.warning("No data available for the selected filters. Adjust the filters.")
        st.stop()

    # KPIs
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Accidents", len(df_descriptive))
    with c2:
        st.metric("Total Victims", int(pd.to_numeric(df_descriptive.get("crash_victims", 0), errors="coerce").fillna(0).sum()))
    with c3:
        st.metric("High Severity", int(pd.to_numeric(df_descriptive.get("serious", 0), errors="coerce").fillna(0).eq(1).sum()))
    # --- Extra KPIs: plates & victims ----------------------------------
    plate1_count = int(df_descriptive['vehicle_plate_1'].notna().sum()) if 'vehicle_plate_1' in df_descriptive.columns else 0
    plate2_count = int(df_descriptive['vehicle_plate_2'].notna().sum()) if 'vehicle_plate_2' in df_descriptive.columns else 0
    plate3_count = int(df_descriptive['vehicle_plate_3'].notna().sum()) if 'vehicle_plate_3' in df_descriptive.columns else 0
    injured_sum  = int(pd.to_numeric(df_descriptive.get('injured_victims', 0), errors='coerce').fillna(0).sum())
    crash_sum    = int(pd.to_numeric(df_descriptive.get('crash_victims', 0), errors='coerce').fillna(0).sum())

    colp1, colp2, colp3, colp4, colp5 = st.columns(5)
    with colp1:
        st.metric("Plates (vehicle_plate_1)", plate1_count)
    with colp2:
        st.metric("Plates (vehicle_plate_2)", plate2_count)
    with colp3:
        st.metric("Plates (vehicle_plate_3)", plate3_count)
    with colp4:
        st.metric("Injured Victims (sum)", injured_sum)
    with colp5:
        st.metric("Crash Victims (sum)", crash_sum)


    # Time series
    if "created" in df_descriptive.columns:
        by_day = df_descriptive.groupby(df_descriptive["created"].dt.tz_convert(None).dt.date).size().reset_index(name="Accidents")
        by_day["Date"] = pd.to_datetime(by_day["created"])
        by_day = by_day.sort_values("Date")
        by_day["rolling_7d"] = by_day["Accidents"].rolling(7, min_periods=1).mean()
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(x=by_day["Date"], y=by_day["Accidents"], mode="lines+markers", name="Daily", opacity=0.45))
        fig_time.add_trace(go.Scatter(x=by_day["Date"], y=by_day["rolling_7d"], mode="lines", name="7-day avg"))
        fig_time.update_layout(title="Accidents over Time", xaxis_title="Date", yaxis_title="Accidents", template=RWA_TEMPLATE)
        st.plotly_chart(fig_time, use_container_width=True)

        with st.expander("📐 Statistical panel — trend"):
            ords = by_day["Date"].map(pd.Timestamp.toordinal)
            r, p = _pearson_r_p(ords, by_day["Accidents"])
            rng = np.random.default_rng(42)
            boots = [by_day["Accidents"].iloc[rng.integers(0, len(by_day), len(by_day))].mean() for _ in range(600)]
            ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
            st.write({"pearson_r": r, "p_value": p, "mean": float(by_day["Accidents"].mean()),
                      "mean_ci_95": (float(ci_lo), float(ci_hi))})

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📅 By Day of Week (95% CI)")
        days = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
        if "dayofweek" in df_descriptive.columns:
            day_counts = df_descriptive["dayofweek"].map(days).value_counts().reindex(days.values(), fill_value=0)
            fig_day = bar_with_ci(list(day_counts.index), day_counts.values.astype(float), "Accidents by Day of Week",
                                  "Day", "Accidents", template=RWA_TEMPLATE)
            st.plotly_chart(fig_day, use_container_width=True)
            with st.expander("Stat test — chi-square vs uniform"):
                if _HAS_SCIPY and len(day_counts) > 1:
                    expected = np.repeat(day_counts.sum() / len(day_counts), len(day_counts))
                    chi2, p = stats.chisquare(day_counts.values, expected)
                    st.write({'chi2_uniform': float(chi2), 'p_value': float(p)})
                else:
                    st.info("Install SciPy to compute chi-square.")
        else:
            st.info("`dayofweek` column not found.")

    with c2:
        st.subheader("🕒 By Hour (95% CI)")
        if "hour" in df_descriptive.columns:
            hr_counts = df_descriptive["hour"].value_counts().sort_index()
            fig_hr = bar_with_ci(hr_counts.index.tolist(), hr_counts.values.astype(float), "Accidents by Hour",
                                 "Hour", "Accidents", template=RWA_TEMPLATE)
            st.plotly_chart(fig_hr, use_container_width=True)
        else:
            st.info("`hour` column not found.")

    st.markdown("---")
    if "weather" in df_descriptive.columns and "severity" in df_descriptive.columns:
        st.subheader("🌦️ Weather × Severity")
        cross = df_descriptive.groupby(["weather", "severity"]).size().reset_index(name="count")
        fig_w = px.bar(cross, x="weather", y="count", color="severity", barmode="group",
                       title="Weather vs Severity", template=RWA_TEMPLATE)
        st.plotly_chart(fig_w, use_container_width=True)
        with st.expander("Stat panel — association (chi-square)"):
            try:
                if "serious" in df_descriptive.columns:
                    ct = pd.crosstab(df_descriptive["weather"], df_descriptive["serious"])
                    if _HAS_SCIPY and ct.shape[0] > 1 and ct.shape[1] > 1:
                        chi2, p, dof, _ = stats.chi2_contingency(ct)
                        st.write({'chi2': float(chi2), 'p_value': float(p), 'dof': int(dof)})
            except Exception:
                pass

    if "location_province" in df_descriptive.columns:
        st.markdown("---")
        st.subheader("🗺️ Accidents by Province (95% CI)")
        pc = df_descriptive["location_province"].value_counts()
        fig_p = bar_with_ci(pc.index.tolist(), pc.values.astype(float), "Accidents by Province", "Province",
                            "Accidents", template=RWA_TEMPLATE)
        st.plotly_chart(fig_p, use_container_width=True)

    # ----------------- Repeated Plate Numbers -----------------------------
    # plate_series_all = _collect_plate_series(df_descriptive)
    # if plate_series_all is not None and not plate_series_all.empty:
    #     _freq_block(
    #         df=df_descriptive,
    #         value_series=plate_series_all,
    #         title_icon="🚗",
    #         title_text="Repeated Plate Numbers",
    #         label="Plate",
    #         default_min_occ=3,
    #         default_top_k=50,
    #         anonymize=False,          # SHOW RAW PLATES by default
    #         mask_keep=2,
    #         normalize=True,
    #         y_axis_title="Crash Count",
    #         show_text_pairs=True,     # Adds "PLATE: count" text list
    #     )
    # else:
    #     # fallback to single-column detection
    #     plate_col = _detect_column(
    #         df_descriptive,
    #         keywords=["plate", "number_plate", "license", "licence", "reg_no", "registration"],
    #         alt_candidates=["plate_number", "number_plate", "license_plate", "licence_plate", "reg_no", "vehicle_plate"]
    #     )
    #     if plate_col is not None:
    #         _freq_block(
    #             df=df_descriptive,
    #             value_series=df_descriptive[plate_col],
    #             title_icon="🚗",
    #             title_text="Repeated Plate Numbers",
    #             label="Plate",
    #             default_min_occ=3,
    #             default_top_k=50,
    #             anonymize=False,
    #             mask_keep=2,
    #             normalize=True,
    #             y_axis_title="Crash Count",
    #             show_text_pairs=True,
    #         )
    #     else:
    #         st.info("No plate-number-like column found (looked for: plate, number_plate, license/licence, reg/registration).")

    # # ----------------- Repeated People Names ------------------------------
    # name_col = _detect_column(
    #     df_descriptive,
    #     keywords=["driver", "owner", "person", "full_name", "name"],
    #     alt_candidates=["driver_name", "owner_name", "person_name", "reported_by", "victim_name"]
    # )
    # if name_col is not None:
    #     _freq_block(
    #         df=df_descriptive,
    #         value_series=df_descriptive[name_col],
    #         title_icon="🧑",
    #         title_text="Repeated People Names",
    #         label="Name",
    #         default_min_occ=2,
    #         default_top_k=30,
    #         anonymize=True,       # Keep names masked by default
    #         mask_keep=1,
    #         normalize=True,
    #         y_axis_title="Count",
    #         show_text_pairs=False,
    #     )
    # else:
    #     st.info("No people-name-like column found (looked for: driver/owner/person/full_name/name).")
