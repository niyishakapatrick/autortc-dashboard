
# predictive.py
import os, json, joblib, numpy as np, pandas as pd, streamlit as st, plotly.express as px, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_fscore_support, mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit.components.v1 import html

MODELS_DIR = "trained_models"
METRICS_DIR = "model_metrics"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# ---- Word2Vec
@st.cache_resource
def load_w2v():
    import gensim.downloader as api
    path = os.path.join(MODELS_DIR, "word2vec_model.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    model = api.load("glove-wiki-gigaword-50")
    joblib.dump(model, path)
    return model

def vec_text(series, model) -> np.ndarray:
    """Accepts str | list[str] | np.ndarray[str] | pd.Series; returns (n, dim)."""
    if isinstance(series, str):
        iterable = [series]
    elif isinstance(series, (list, tuple, np.ndarray)):
        iterable = [str(x) for x in series]
    elif isinstance(series, pd.Series):
        iterable = series.astype(str).tolist()
    else:
        iterable = [str(series)]
    out = []
    for doc in iterable:
        words = doc.lower().split()
        wv = [model[w] for w in words if w in model]
        out.append(np.mean(wv, axis=0) if wv else np.zeros(model.vector_size))
    return np.vstack(out)

def _apply_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    q = df[
        df["location_province"].isin(f["location_province"]) &
        df["location_name"].isin(f["location_name"])
    ]
    if all(v is not None for v in [f.get("vehicle_type"), f.get("road_type"), f.get("road_condition")]):
        q = q[
            q["vehicle_type"].isin(f["vehicle_type"]) &
            q["road_type"].isin(f["road_type"]) &
            q["road_condition"].isin(f["road_condition"])
        ]
    return q

def _save_cm_png(cm, name):
    fig, ax = plt.subplots(figsize=(4.5,4))
    ConfusionMatrixDisplay(cm, display_labels=["Minor","Serious"]).plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
    ax.set_title("Confusion Matrix")
    path = os.path.join(METRICS_DIR, f"{name}_confusion.png")
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return path

# ---------- Regression feature builders & compatibility helpers ----------
CLASS_FEATURES = ['latitude','longitude','weather_encoded','hour','dayofweek','vehicle_count']
REG_FEATURES   = ['latitude','longitude','weather_encoded','crash_victims','hour','dayofweek','vehicle_count']

def build_reg_struct(df_like: pd.DataFrame) -> np.ndarray:
    return df_like[REG_FEATURES].astype(float).values

def build_reg_struct_plus_text(df_like: pd.DataFrame, w2v) -> np.ndarray:
    Xs = build_reg_struct(df_like)
    Xt = vec_text(df_like['description'], w2v)
    return np.hstack([Xs, Xt])

def adapt_features_for_model(model, X_struct: np.ndarray, X_text: np.ndarray | None, w2v_dim: int) -> np.ndarray:
    """Return feature matrix matching model.n_features_in_. If model expects struct-only (7), drop text.
    If expects struct+text (7+dim), concat. If mismatch, pad or slice safely."""
    exp = getattr(model, 'n_features_in_', None)
    if exp is None:
        return X_struct
    if exp == X_struct.shape[1]:
        return X_struct
    if X_text is not None and exp == X_struct.shape[1] + w2v_dim:
        return np.hstack([X_struct, X_text])
    X = X_struct if X_text is None else np.hstack([X_struct, X_text])
    if X.shape[1] > exp:
        return X[:, :exp]
    if X.shape[1] < exp:
        pad = np.zeros((X.shape[0], exp - X.shape[1]))
        return np.hstack([X, pad])
    return X

def render(df: pd.DataFrame, f: dict):
    st.title("🤖 Predictive Analytics")

    dfp = _apply_filters(df, f)
    if dfp.empty:
        st.warning("No data for the selected filters.")
        st.stop()

    if "weather_encoded" not in dfp.columns:
        if "weather" in dfp.columns:
            dfp["weather_encoded"] = LabelEncoder().fit_transform(dfp["weather"].astype(str))
        else:
            dfp["weather_encoded"] = 0

    # ---------------- Classification ----------------
    with st.expander("🔬 Severity Prediction (Classification)", expanded=False):
        w2v = load_w2v()
        poly = PolynomialFeatures(degree=2, include_bias=False)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Start Training Classifiers"):
                Xp = poly.fit_transform(dfp[CLASS_FEATURES])
                Xt = vec_text(dfp['description'], w2v)
                X = np.hstack([Xp, Xt])
                y = dfp['serious'].astype(int)
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

                models = {
                    "Logistic Regression": LogisticRegression(max_iter=1200, solver="liblinear"),
                    "Random Forest": RandomForestClassifier(n_estimators=400, random_state=42),
                    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
                    "LightGBM": LGBMClassifier(random_state=42),
                    "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
                    "LDA": LinearDiscriminantAnalysis(),
                    "KNN": KNeighborsClassifier(n_neighbors=7),
                    "Gaussian NB": GaussianNB(),
                }
                rows = []
                for name, mdl in models.items():
                    mdl.fit(X_tr, y_tr)
                    y_pred = mdl.predict(X_te)
                    acc = (y_pred == y_te).mean()
                    prec, rec, f1, _ = precision_recall_fscore_support(y_te, y_pred, average="binary", zero_division=0)
                    rows.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1})
                    joblib.dump(mdl, os.path.join(MODELS_DIR, f"{name.replace(' ','_').lower()}.pkl"))
                    _save_cm_png(confusion_matrix(y_te, y_pred), name.replace(' ','_').lower())

                # Save artifacts
                joblib.dump(poly, os.path.join(MODELS_DIR, "poly_features.pkl"))
                joblib.dump(w2v, os.path.join(MODELS_DIR, "word2vec_model.pkl"))

                res_df = pd.DataFrame(rows).sort_values(["F1","Accuracy"], ascending=[False, False]).reset_index(drop=True)
                res_df.to_csv(os.path.join(METRICS_DIR, "classification_leaderboard.csv"), index=False)
                st.success(f"Training complete! Top model: {res_df.iloc[0]['Model']}")
                st.dataframe(res_df, use_container_width=True)

        with c2:
            st.markdown("**📦 Evaluate / Inference from saved classifier**")
            model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl') and not any(k in f for k in ['regressor','poly','word2vec'])]
            clf_name = st.selectbox("Pick classifier:", model_files)
            if st.button("Show test Accuracy + Confusion Matrix") and clf_name:
                mdl = joblib.load(os.path.join(MODELS_DIR, clf_name))
                poly2 = joblib.load(os.path.join(MODELS_DIR, "poly_features.pkl"))
                w2v2  = joblib.load(os.path.join(MODELS_DIR, "word2vec_model.pkl"))
                Xp = poly2.transform(dfp[CLASS_FEATURES])
                Xt = vec_text(dfp['description'], w2v2)
                X = np.hstack([Xp, Xt]); y = dfp['serious'].astype(int)
                _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                y_pred = mdl.predict(X_test)
                acc = (y_pred==y_test).mean()
                st.write(f"**Test Accuracy:** {acc:.4f}")
                st.text(classification_report(y_test, y_pred, target_names=["Minor","Serious"]))
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(5,4))
                ConfusionMatrixDisplay(cm, display_labels=["Minor","Serious"]).plot(ax=ax, cmap="Blues", values_format="d")
                st.pyplot(fig); plt.close(fig)

            st.markdown("**🔮 Single-instance prediction**")
            with st.form("clf_infer"):
                c11,c12,c13 = st.columns(3)
                with c11:
                    lat = st.number_input("Latitude", value=float(dfp['latitude'].mean()))
                    hour = st.slider("Hour", 0, 23, 12)
                with c12:
                    lon = st.number_input("Longitude", value=float(dfp['longitude'].mean()))
                    dow = st.slider("Day of week", 0, 6, 2)
                with c13:
                    weather = st.selectbox("Weather (encoded)", sorted(dfp['weather_encoded'].unique().tolist()))
                    vehs = st.number_input("Vehicle count", min_value=0, value=int(dfp['vehicle_count'].median()))
                desc = st.text_input("Short description (optional)", "")
                submit = st.form_submit_button("Predict severity")
                if submit and clf_name:
                    mdl = joblib.load(os.path.join(MODELS_DIR, clf_name))
                    poly2 = joblib.load(os.path.join(MODELS_DIR, "poly_features.pkl"))
                    w2v2  = joblib.load(os.path.join(MODELS_DIR, "word2vec_model.pkl"))
                    Xnew = pd.DataFrame([[lat, lon, weather, hour, dow, vehs]], columns=CLASS_FEATURES)
                    Xp = poly2.transform(Xnew); Xt = vec_text(desc, w2v2); X = np.hstack([Xp, Xt])
                    pred = mdl.predict(X)[0]
                    proba = getattr(mdl, "predict_proba", None)
                    if proba:
                        p = mdl.predict_proba(X)[0][1]
                        st.success(f"Predicted: **{'Serious' if pred==1 else 'Minor'}** (prob Serious ≈ {p:.2%})")
                    else:
                        st.success(f"Predicted: **{'Serious' if pred==1 else 'Minor'}**")

        # Feature importance (RF)
        st.markdown("---")
        st.subheader("💡 Feature Importance (Classification)")
        try:
            rf_path = None
            for f_ in os.listdir(MODELS_DIR):
                if f_.startswith("random_forest") and f_.endswith(".pkl") and "regressor" not in f_:
                    rf_path = os.path.join(MODELS_DIR, f_); break
            if rf_path:
                rf = joblib.load(rf_path)
                poly2 = joblib.load(os.path.join(MODELS_DIR, "poly_features.pkl"))
                feat_names = list(poly2.get_feature_names_out(CLASS_FEATURES)) + [f"text_{i}" for i in range(50)]
                imp = rf.feature_importances_
                tab = (pd.DataFrame({"Feature": feat_names, "Importance": imp})
                       .replace({f"text_{i}":"Text Description (Word2Vec)" for i in range(50)})
                       .groupby("Feature", as_index=False)["Importance"].sum()
                       .sort_values("Importance", ascending=False).head(20))
                st.plotly_chart(px.bar(tab, x="Importance", y="Feature", orientation="h", title="Top Feature Importances"), use_container_width=True)
            else:
                st.info("Train Random Forest to view importances.")
        except Exception as e:
            st.info(f"Feature importance unavailable: {e}")

    # ---------------- Regression ----------------
    with st.expander("📈 Injured Victims Prediction (Regression)", expanded=False):
        w2v = load_w2v()
        # Training on struct + text (new pipeline)
        X_struct = dfp[REG_FEATURES].astype(float)
        X_text   = vec_text(dfp['description'], w2v)
        X_full   = np.hstack([X_struct.values, X_text])
        y_reg    = dfp['injured_victims'].astype(float)

        c1,c2 = st.columns(2)
        with c1:
            if st.button("Start Training Regressors"):
                models = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest Regressor": RandomForestRegressor(n_estimators=500, random_state=42)
                }
                rows = []
                Xtr, Xte, ytr, yte = train_test_split(X_full, y_reg, test_size=0.2, random_state=42)
                for name, reg in models.items():
                    reg.fit(Xtr, ytr); yp = reg.predict(Xte)
                    rows.append({"Model": name, "R2": r2_score(yte, yp), "MAE": mean_absolute_error(yte, yp), "MSE": mean_squared_error(yte, yp)})
                    fname = f"{name.replace(' ','_').lower()}_regressor.pkl"
                    joblib.dump(reg, os.path.join(MODELS_DIR, fname))
                    # save metadata about feature shapes
                    meta = {"uses_text": True, "embedding_dim": int(w2v.vector_size), "reg_features": REG_FEATURES}
                    with open(os.path.join(MODELS_DIR, fname.replace(".pkl", ".meta.json")), "w") as fmeta:
                        json.dump(meta, fmeta)
                res = pd.DataFrame(rows).sort_values(["R2","MAE"], ascending=[False, True]).reset_index(drop=True)
                res.to_csv(os.path.join(METRICS_DIR, "regression_leaderboard.csv"), index=False)
                st.dataframe(res, use_container_width=True)
                # save mean text embedding for future forecasts
                mean_text = X_text.mean(axis=0)
                np.save(os.path.join(MODELS_DIR, "reg_text_mean.npy"), mean_text)

        with c2:
            reg_files = [f for f in os.listdir(MODELS_DIR) if f.endswith("_regressor.pkl")]
            sel = st.selectbox("Pick a saved regressor", reg_files)
            if st.button("Show R² / MAE / MSE") and sel:
                reg = joblib.load(os.path.join(MODELS_DIR, sel))
                # Decide feature shape based on model expectation / metadata
                meta_path = os.path.join(MODELS_DIR, sel.replace(".pkl", ".meta.json"))
                w2v_dim = load_w2v().vector_size
                if os.path.exists(meta_path):
                    with open(meta_path) as fmeta: meta = json.load(fmeta)
                    use_text = bool(meta.get("uses_text", False))
                else:
                    use_text = (getattr(reg, "n_features_in_", len(REG_FEATURES)) != len(REG_FEATURES))
                Xs = X_struct.values
                Xt = X_text if use_text else None
                Xfor = adapt_features_for_model(reg, Xs, Xt, w2v_dim)
                Xtr, Xte, ytr, yte = train_test_split(Xfor, y_reg, test_size=0.2, random_state=42)
                yp = reg.predict(Xte)
                st.write(f"**R²:** {r2_score(yte, yp):.4f} | **MAE:** {mean_absolute_error(yte, yp):.4f} | **MSE:** {mean_squared_error(yte, yp):.4f}")

            st.markdown("**🔮 Single-instance regression**")
            with st.form("reg_infer"):
                c11,c12,c13 = st.columns(3)
                with c11:
                    lat = st.number_input("Latitude", value=float(dfp['latitude'].mean()))
                    weather = st.selectbox("Weather (encoded)", sorted(dfp['weather_encoded'].unique().tolist()))
                    hour = st.slider("Hour", 0, 23, 12)
                with c12:
                    lon = st.number_input("Longitude", value=float(dfp['longitude'].mean()))
                    crash = st.number_input("Crash victims", min_value=0, value=int(dfp['crash_victims'].median()))
                    day = st.slider("Day of week", 0, 6, 2)
                with c13:
                    veh = st.number_input("Vehicles", min_value=0, value=int(dfp['vehicle_count'].median()))
                    desc = st.text_input("Accident description (optional)", "")
                sub = st.form_submit_button("Predict injured victims")
                if sub and sel:
                    reg = joblib.load(os.path.join(MODELS_DIR, sel))
                    w2v2 = load_w2v()
                    Xs = pd.DataFrame([[lat, lon, weather, crash, hour, day, veh]], columns=REG_FEATURES).astype(float).values
                    Xt = vec_text(desc, w2v2)
                    Xfull = adapt_features_for_model(reg, Xs, Xt, w2v2.vector_size)
                    pred = reg.predict(Xfull)[0]
                    st.success(f"Predicted Injured Victims: **{pred:.2f}**")

        # Feature importance (RF Regressor) — aggregate text to one bar
        st.markdown("---")
        st.subheader("💡 Feature Importance (Regression)")
        try:
            reg_files = [f for f in os.listdir(MODELS_DIR) if f.endswith("_regressor.pkl")]
            rf_path = None
            for f_ in reg_files:
                if f_.startswith("random_forest_regressor"):
                    rf_path = os.path.join(MODELS_DIR, f_); break
            if rf_path:
                rf = joblib.load(rf_path)
                imp = rf.feature_importances_
                exp = getattr(rf, 'n_features_in_', len(REG_FEATURES))
                w2v_dim = load_w2v().vector_size
                if exp == len(REG_FEATURES):
                    feat_names = list(REG_FEATURES)
                elif exp == len(REG_FEATURES) + w2v_dim:
                    feat_names = list(REG_FEATURES) + [f"text_{i}" for i in range(w2v_dim)]
                else:
                    extra = max(0, exp - len(REG_FEATURES))
                    feat_names = list(REG_FEATURES) + [f"text_{i}" for i in range(extra)]
                tab = (
                    pd.DataFrame({"Feature": feat_names, "Importance": imp})
                    .replace({fn: "Text Description (Word2Vec)" for fn in feat_names if fn.startswith("text_")})
                    .groupby("Feature", as_index=False)["Importance"].sum()
                    .sort_values("Importance", ascending=False)
                )
                st.plotly_chart(px.bar(tab.head(25), x="Importance", y="Feature", orientation="h",
                                       title="Top Feature Importances (Regression)"),
                                use_container_width=True)
            else:
                st.info("Train the Random Forest Regressor to view importances.")
        except Exception as e:
            st.info(f"Feature importance unavailable: {e}")

    # ---------------- Future Outlook ----------------
    with st.expander("📅 Future Outlook", expanded=True):
        st.info("Forecast injured victims for the next N days using the saved Random Forest Regressor. Dashboard summary only (no maps or bar charts).")

        horizon = st.slider("Forecast horizon (days)", 7, 60, 30)
        reg_files = [f for f in os.listdir(MODELS_DIR) if f.endswith("_regressor.pkl")]
        rf_path = next((os.path.join(MODELS_DIR, f) for f in reg_files if f.startswith("random_forest_regressor")), None)

        if rf_path:
            reg = joblib.load(rf_path)
            w2v = load_w2v()
            mean_text_path = os.path.join(MODELS_DIR, "reg_text_mean.npy")
            mean_text = np.load(mean_text_path) if os.path.exists(mean_text_path) else np.zeros(w2v.vector_size)

            last_date = dfp["created"].max() if "created" in dfp else pd.Timestamp.today()
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon)

            # Province centroids
            prov_centroids = dfp.groupby("location_province")[["latitude","longitude"]].mean().dropna()
            rows = []
            exp = getattr(reg, 'n_features_in_', len(REG_FEATURES))
            for prov, row in prov_centroids.iterrows():
                lat, lon = float(row["latitude"]), float(row["longitude"])
                for d in future_dates:
                    base = pd.DataFrame([[lat, lon,
                                           int(dfp["weather_encoded"].mode()[0]) if "weather_encoded" in dfp else 0,
                                           float(dfp["crash_victims"].mean()) if "crash_victims" in dfp else 0.0,
                                           int(dfp["hour"].mode()[0]) if "hour" in dfp else 12,
                                           d.dayofweek,
                                           int(dfp["vehicle_count"].mean()) if "vehicle_count" in dfp else 1]],
                                        columns=REG_FEATURES).astype(float).values
                    if exp == len(REG_FEATURES):
                        Xf = base
                    elif exp == len(REG_FEATURES) + w2v.vector_size:
                        Xf = np.hstack([base, mean_text.reshape(1, -1)])
                    else:
                        pad = np.zeros((1, max(0, exp - base.shape[1])))
                        Xf = np.hstack([base, pad])[:, :exp]
                    pred = reg.predict(Xf)[0]
                    rows.append({"Date": d, "Province": prov, "Predicted Victims": float(pred)})
            fdf = pd.DataFrame(rows)

            # Dashboard KPIs (no tables/plots)
            total = float(fdf["Predicted Victims"].sum())
            per_day = fdf.groupby("Date")["Predicted Victims"].sum()
            daily_avg = float(per_day.mean())
            peak_day = per_day.idxmax()
            peak_val = int(per_day.max())
            prov_totals = fdf.groupby("Province")["Predicted Victims"].sum().sort_values(ascending=False)
            top_prov = prov_totals.index[0] if len(prov_totals)>0 else "N/A"
            num_prov = int(prov_totals.shape[0])

            c1,c2,c3,c4,c5,c6 = st.columns(6)
            with c1: st.metric("Total victims (horizon)", f"{int(total)}")
            with c2: st.metric("Daily average", f"{daily_avg:.1f}")
            with c3: st.metric("Peak day", peak_day.strftime("%Y-%m-%d"))
            with c4: st.metric("Victims on peak day", f"{peak_val}")
            with c5: st.metric("Top province", top_prov)
            with c6: st.metric("Provinces covered", f"{num_prov}")
        else:
            st.warning("Please train and save a Random Forest Regressor to enable future forecasts.")

    # ---------------- Clustering / Hotspots ----------------
    with st.expander("📍 Accident Hotspot Analysis (Folium)", expanded=True):
        st.info("DBSCAN detects dense hotspots. Adjust the radius (km) & min points. Map auto-fits to your selection.")
        coords = dfp[['latitude','longitude']].dropna()
        if coords.empty:
            st.warning("No coordinates available after filters.")
        else:
            eps_km = st.slider("Neighborhood radius (km)", 1, 30, 5)
            eps_deg = eps_km / 111.0  # ~111 km per degree
            min_samples = st.slider("Min accidents per cluster", 3, 30, 6)

            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=eps_deg, min_samples=min_samples).fit(coords)
            dfc = dfp.copy()
            dfc['cluster'] = -1
            dfc.loc[coords.index, 'cluster'] = clustering.labels_

            hotspots = dfc[dfc['cluster'] != -1].copy()
            if hotspots.empty:
                st.info("No hotspots under these parameters.")
            else:
                cluster_ids = sorted(hotspots['cluster'].unique().tolist())
                choice = st.selectbox("Hotspot cluster", [-1]+cluster_ids, format_func=lambda x: "All hotspots" if x==-1 else f"Cluster {x}")
                show = hotspots if choice==-1 else hotspots[hotspots['cluster']==choice]

                # Folium map
                # Folium map
                center = [show['latitude'].mean(), show['longitude'].mean()]
                m = folium.Map(location=center, zoom_start=9, tiles='cartodbpositron', control_scale=True)
                mc = MarkerCluster().add_to(m)
                for _, r in show.iterrows():
                    color = "red" if r.get('serious',0)==1 else "blue"
                    folium.CircleMarker(
                        location=[r['latitude'], r['longitude']],
                        radius=5, color=color, fill=True, fill_color=color,
                        popup=f"{r.get('location_name','N/A')} | severity: {r.get('severity','?')}",
                    ).add_to(mc)

                # Save map once as HTML to avoid regeneration on refresh
                map_path = os.path.join("maps", "hotspot_map.html")
                os.makedirs("maps", exist_ok=True)
                m.save(map_path)
                with open(map_path, "r") as f:
                    map_html = f.read()
                html(map_html, height=540, width=1000)

                if 'location_name' in show.columns:
                    st.markdown("**Top 5 locations in selection**")
                    top5 = (show.groupby(['location_province','location_name'])
                                 .size().reset_index(name='accidents')
                                 .sort_values('accidents', ascending=False).head(5))
                    st.dataframe(top5, use_container_width=True)
