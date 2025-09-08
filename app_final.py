## streamlit run app_final.py
## app_final

import io
import re
import pickle
import unicodedata

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.base import clone

# Type checkers para dtypes datetime/timedelta
from pandas.api.types import is_datetime64_any_dtype, is_timedelta64_dtype

# =========================
# Configuración
# =========================
st.set_page_config(page_title="Entrenamiento €/m²", layout="wide")
st.title("🧠 Entrenar y predecir €/m² (OMP)")
st.caption("Sube un Excel o añade filas manualmente; entrena un pipeline (prep → OMP) y rellena 'PRECIO M2' solo donde esté vacío (NaN).")

# =========================
# Constantes y utilidades
# =========================
DEFAULT_TARGET = "PRECIO M2"
DEFAULT_YEAR   = "AÑO"
DEFAULT_MONTH  = "MES_NUM"
DEFAULT_DIST   = "DISTRITO"

ALLOWED_DISTRICTS = [
    "Carabanchel","Centro","Tetuan","Ciudad Lineal","Fuencarral-El Pardo","Moncloa-Aravaca",
    "Puente De Vallecas","San Blas-Canillejas","Chamartin","Hortaleza","Usera","Arganzuela",
    "Latina","Salamanca","Chamberi","Villaverde","Retiro","Barajas","Villa De Vallecas",
    "Moratalaz","Vicalvaro"
]

REQUIRED_FEATURES_EXCL_TARGET = [
    DEFAULT_YEAR, DEFAULT_MONTH,
    "EURIBOR_lag12","VARIACIÓN ANUAL_lag12","PRECIO M2_lag12","TRANSACCIONES_CM_lag12",
    "INDICE_PRECIO_lag12","TOTAL_HIPOTECAS_lag12","POBLACION_ACTIVA_lag12","POBLACION_lag12",
    "ESPERANZA_VIDA_lag12","VIVIENDAS_COMPRAVENTA_lag12","TRANSACCIONES_SUELO_lag12",
    "PRECIO_MEDIO_M2_CCMM_lag12",
    DEFAULT_DIST
]
TEMPLATE_COLUMNS = [DEFAULT_TARGET] + REQUIRED_FEATURES_EXCL_TARGET

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def normalize_key(s: str) -> str:
    s = _strip_accents(str(s)).lower().strip()
    s = re.sub(r"[\s\-_]+", "", s)
    return s

DIST_KEYS = {normalize_key(n): n for n in ALLOWED_DISTRICTS}

def clean_distrito_column(df: pd.DataFrame, col: str = DEFAULT_DIST):
    """Normaliza DISTRITO a la lista fija; devuelve (df_limpio, valores_invalidos_originales)."""
    if col not in df.columns:
        return df, []
    df2 = df.copy()
    def map_one(x):
        if pd.isna(x): return np.nan
        return DIST_KEYS.get(normalize_key(str(x)), np.nan)
    original_vals = df2[col].astype(str)
    df2[col] = df2[col].apply(map_one)
    invalid_mask = df2[col].isna()
    invalid_vals = sorted(pd.unique(original_vals[invalid_mask]))
    return df2, invalid_vals

def detect_binary_int_cols(df: pd.DataFrame) -> list:
    int_cols = df.select_dtypes(include=["int8","int16","int32","int64","uint8","uint16","uint32","uint64"]).columns
    return [c for c in int_cols if df[c].dropna().isin([0,1]).all()]

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def adjusted_r2(r2, n, p):
    if n - p - 1 <= 0:
        return np.nan
    return float(1.0 - (1.0 - r2) * (n - 1) / (n - p - 1))

def make_preprocessor(X: pd.DataFrame, scale_dummies: bool):
    """StandardScaler en floats y (opcional) en dummies 0/1; resto passthrough."""
    float_cols = list(X.select_dtypes(include=["float32","float64"]).columns)
    bin_cols   = detect_binary_int_cols(X) if scale_dummies else []
    scale_cols = sorted(set(float_cols + bin_cols))
    return ColumnTransformer(
        transformers=[("num", StandardScaler(with_mean=True, with_std=True), scale_cols)],
        remainder="passthrough",
        verbose_feature_names_out=False
    ), scale_cols

def to_excel_download(df: pd.DataFrame, filename="dataset.xlsx") -> tuple[bytes, str]:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="data")
    bio.seek(0)
    return bio.getvalue(), filename

def build_pipeline(prep: ColumnTransformer, *, fit_intercept: bool, tol, precompute) -> Pipeline:
    return Pipeline(steps=[
        ("prep", prep),
        ("omp", OrthogonalMatchingPursuit(
            fit_intercept=fit_intercept,
            tol=tol,                 # None o float
            precompute=precompute    # 'auto', True o False
        ))
    ])

def get_dummies_if_checked(df_in: pd.DataFrame, make_dummies: bool) -> pd.DataFrame:
    """Fuerza DISTRITO a Categorical con categorías fijas; si make_dummies=True, aplica OHE a todas las categóricas."""
    df = df_in.copy()
    if DEFAULT_DIST in df.columns:
        df[DEFAULT_DIST] = pd.Categorical(df[DEFAULT_DIST], categories=ALLOWED_DISTRICTS)
    if not make_dummies:
        return df
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    if not cat_cols:
        return df
    return pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype=np.int32)

def time_index_from_year_month(df, year_col, month_col):
    return pd.to_datetime(dict(
        year=df[year_col].astype(int),
        month=df[month_col].astype(int),
        day=1
    ))

def gridsearch_k(pipeline, X_tr, y_tr, k_min, k_max, cv, scoring_key: str):
    ks = list(range(k_min, k_max+1))
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid={"omp__n_nonzero_coefs": ks},
        scoring=scoring_key,
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True
    )
    grid.fit(X_tr, y_tr)
    cv_results = pd.DataFrame(grid.cv_results_)
    cv_results["k"] = cv_results["param_omp__n_nonzero_coefs"].astype(int)
    if scoring_key == "neg_mean_absolute_error":
        cv_results["score"] = -cv_results["mean_test_score"]; score_name = "MAE_CV"
    elif scoring_key == "neg_mean_squared_error":
        cv_results["score"] = -cv_results["mean_test_score"]; score_name = "MSE_CV"
    elif scoring_key == "r2":
        cv_results["score"] = cv_results["mean_test_score"];  score_name = "R2_CV"
    else:
        cv_results["score"] = cv_results["mean_test_score"];  score_name = "score"
    return grid, cv_results[["k","score"]].rename(columns={"score": score_name}).sort_values("k").reset_index(drop=True)

def metrics_table(y_tr, pred_tr, y_te, pred_te, p_nonzero):
    r2_tr = r2_score(y_tr, pred_tr); r2_te = r2_score(y_te, pred_te)
    out = pd.DataFrame({
        "métrica": ["MAE test", "RMSE test", "MAPE test (%)", "R² test", "R² ajustado test", "R² train"],
        "valor": [
            mean_absolute_error(y_te, pred_te),
            rmse(y_te, pred_te),
            float(np.nanmean(np.abs((y_te - pred_te) / np.where(y_te==0, np.nan, np.abs(y_te))) * 100)),
            r2_te,
            adjusted_r2(r2_te, n=len(y_te), p=p_nonzero),
            r2_tr
        ]
    })
    return out

def check_required_columns(df: pd.DataFrame, required_cols: list[str]) -> list[str]:
    return [c for c in required_cols if c not in df.columns]

def strip_datetime_cols(df: pd.DataFrame, extra_drop=None) -> pd.DataFrame:
    """Elimina columnas datetime/timedelta (con o sin tz) y nombres típicos de fecha."""
    extra_drop = extra_drop or []
    df2 = df.copy()
    dt_like_cols = [c for c in df2.columns if is_datetime64_any_dtype(df2[c]) or is_timedelta64_dtype(df2[c])]
    name_based = [c for c in df2.columns if c.lower() in {"fecha", "_fecha_"}]
    drop_cols = list(dict.fromkeys(extra_drop + dt_like_cols + name_based))
    return df2.drop(columns=drop_cols, errors="ignore")

# =========================
# Estado (filas manuales)
# =========================
if "manual_rows" not in st.session_state:
    st.session_state.manual_rows = pd.DataFrame()

# =========================
# Sidebar (parámetros)
# =========================
st.sidebar.header("⚙️ Parámetros")

with st.sidebar.expander("Preprocesado", expanded=True):
    auto_dummies   = st.checkbox("Crear dummies automáticamente (object/category)", value=True)
    scale_dummies  = st.checkbox("Escalar también dummies 0/1", value=True)

with st.sidebar.expander("Validación", expanded=True):
    use_time_cv    = st.checkbox("Usar validación temporal (TimeSeriesSplit)", value=True)
    n_splits       = st.number_input("n_splits (CV)", min_value=3, max_value=10, value=5, step=1)
    ventana        = st.number_input("max_train_size (meses, opcional)", min_value=0, value=72, step=12)
    gap            = st.number_input("gap (meses)", min_value=0, value=12, step=1)
    holdout_meses  = st.number_input("Meses para test (holdout al final)", min_value=1, value=12, step=1)

with st.sidebar.expander("Modelo OMP", expanded=True):
    k_min = st.number_input("k mínimo (n_nonzero_coefs)", min_value=1, value=3, step=1)
    k_max = st.number_input("k máximo (n_nonzero_coefs)", min_value=1, value=6, step=1)
    if k_max < k_min:
        st.warning("k máx < k mín → ajustando automáticamente."); k_max = k_min
    fit_intercept = st.checkbox("fit_intercept", value=True)
    tol_str = st.text_input("tol (opcional; vacío = None)", value="")
    precompute_opt = st.selectbox("precompute", options=["auto","True","False"], index=0)
    metric_choice  = st.selectbox("Métrica de CV (GridSearch)", options=["MAE","MSE","R²"], index=0)
    scoring_map    = {"MAE":"neg_mean_absolute_error", "MSE":"neg_mean_squared_error", "R²":"r2"}
    scoring_key    = scoring_map[metric_choice]
    tol_val        = None if tol_str.strip()=="" else float(tol_str)
    precompute_val = {"auto":"auto","True":True,"False":False}[precompute_opt]

# =========================
# Aviso de lags (muy importante)
# =========================
st.info("**IMPORTANTE**: salvo **AÑO**, **MES_NUM** y **PRECIO M2**, **todas las demás variables son del AÑO ANTERIOR (lag 12)**. Es decir, pon como **AÑO** el que quieres predecir y usa los **datos disponibles del presente** (que corresponden al año previo).")

# =========================
# Tabs
# =========================
tab_excel, tab_manual = st.tabs(["📥 Entrenar / Predecir con Excel", "📝 Alta manual + predecir"])

# -----------------------------------------------------------------------------
# TAB 1: EXCEL
# -----------------------------------------------------------------------------
with tab_excel:
    st.subheader("1) Cargar Excel")
    st.caption("El Excel debe contener **al menos** estas columnas: " + ", ".join(TEMPLATE_COLUMNS))
    if st.button("📄 Descargar PLANTILLA Excel"):
        template = pd.DataFrame(columns=TEMPLATE_COLUMNS)
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as w:
            template.to_excel(w, index=False, sheet_name="plantilla")
        bio.seek(0)
        st.download_button("Descargar plantilla.xlsx", data=bio.getvalue(),
                           file_name="plantilla_prediccion_omp.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    file = st.file_uploader("Sube un Excel", type=["xlsx","xls"])

    if file is not None:
        df_raw = pd.read_excel(file)
        st.write("Vista previa:"); st.dataframe(df_raw.head(10), width="stretch")

        # Verifica columnas requeridas
        missing_feats = check_required_columns(df_raw, REQUIRED_FEATURES_EXCL_TARGET)
        if missing_feats:
            st.error(f"Faltan columnas obligatorias: {missing_feats}")
            st.stop()

        # Normaliza DISTRITO
        df, invalid_vals = clean_distrito_column(df_raw, DEFAULT_DIST)
        if invalid_vals:
            st.warning(f"Se descartan filas con DISTRITO no válido: {invalid_vals}")
            df = df[~df[DEFAULT_DIST].isna()].copy()

        # ===== Entrenamiento =====
        st.subheader("2) Entrenamiento (si el Excel trae 'PRECIO M2')")
        if DEFAULT_TARGET in df.columns and df[DEFAULT_TARGET].notna().any():
            df_train = df[df[DEFAULT_TARGET].notna()].copy()

            df_train["_FECHA_"] = time_index_from_year_month(df_train, DEFAULT_YEAR, DEFAULT_MONTH)
            df_train.sort_values("_FECHA_", inplace=True)

            y = df_train[DEFAULT_TARGET].astype(float)

            # Quitar fechas antes de X
            X = df_train.drop(columns=[DEFAULT_TARGET], errors="ignore")
            X = strip_datetime_cols(X, extra_drop=["_FECHA_", "FECHA"])

            # OHE estable
            X = get_dummies_if_checked(X, auto_dummies)

            # Holdout temporal
            max_date = df_train["_FECHA_"].max()
            cutoff = max_date - pd.offsets.MonthBegin(holdout_meses)
            train_mask = df_train["_FECHA_"] <= cutoff
            test_mask  = ~train_mask
            X_train, X_test = X.loc[train_mask], X.loc[test_mask]
            y_train, y_test = y.loc[train_mask], y.loc[test_mask]

            st.write(f"**Tamaño train/test:** {X_train.shape} / {X_test.shape}")

            # Preprocesado y pipeline
            preproc, _ = make_preprocessor(X_train, scale_dummies)
            pipeline = build_pipeline(preproc, fit_intercept=fit_intercept, tol=tol_val, precompute=precompute_val)

            # CV temporal
            try:
                tscv = TimeSeriesSplit(n_splits=int(n_splits),
                                       max_train_size=(None if ventana==0 else int(ventana)),
                                       gap=int(gap))
            except TypeError:
                tscv = TimeSeriesSplit(n_splits=int(n_splits),
                                       max_train_size=(None if ventana==0 else int(ventana)))
            cv_obj = tscv

            st.subheader("3) Búsqueda de k (GridSearchCV)")
            grid, cv_table = gridsearch_k(pipeline, X_train, y_train, int(k_min), int(k_max), cv_obj, scoring_key)
            st.dataframe(cv_table, width="stretch")
            best_k = int(grid.best_params_["omp__n_nonzero_coefs"])
            st.success(f"Mejor k por {('MAE' if scoring_key=='neg_mean_absolute_error' else 'MSE' if scoring_key=='neg_mean_squared_error' else 'R²')}: **{best_k}**")

            # Entrenar y evaluar
            model_k = clone(pipeline).set_params(omp__n_nonzero_coefs=best_k).fit(X_train, y_train)
            y_pred_tr = model_k.predict(X_train); y_pred_te = model_k.predict(X_test)
            p_nonzero = int(np.sum(np.abs(model_k.named_steps["omp"].coef_) > 0))

            st.subheader("4) Métricas")
            st.dataframe(metrics_table(y_train.values, y_pred_tr, y_test.values, y_pred_te, p_nonzero), width="stretch")

            # Coefs
            st.subheader("5) Coeficientes (estandarizados)")
            feat_names = list(model_k.named_steps["prep"].get_feature_names_out())
            coefs = model_k.named_steps["omp"].coef_
            coefs_df = (pd.DataFrame({"feature": feat_names, "coef_estandarizado": coefs})
                        .assign(abs_coef=lambda d: d["coef_estandarizado"].abs())
                        .sort_values("abs_coef", ascending=False))
            st.dataframe(coefs_df.head(30), width="stretch")

            # Modelo final
            st.subheader("6) Modelo final (train+test)")
            X_all = pd.concat([X_train, X_test], axis=0); y_all = pd.concat([y_train, y_test], axis=0)
            final_model_k = clone(pipeline).set_params(omp__n_nonzero_coefs=best_k).fit(X_all, y_all)
            st.success("Modelo final entrenado con todo el histórico.")

            # Guardar en sesión para predicción
            st.session_state.final_model_k   = final_model_k
            st.session_state.feature_cols    = X_train.columns.tolist()
            st.session_state.feature_dtypes  = {c: X_train[c].dtype for c in X_train.columns}
            st.session_state.auto_dummies    = auto_dummies

            # Descargar modelo
            pkl_bytes = pickle.dumps(final_model_k)
            st.download_button("💾 Descargar modelo (.pkl)", data=pkl_bytes,
                               file_name="final_model_k.pkl", mime="application/octet-stream")
        else:
            st.info("No hay 'PRECIO M2' informado en el Excel: se omite el entrenamiento.")

        # ===== Predicción en el MISMO Excel (solo NaN) =====
        st.subheader("7) Predecir (rellenar solo 'PRECIO M2' vacío en este mismo Excel)")
        model = st.session_state.get("final_model_k", None)
        if model is None:
            st.info("Entrena el modelo arriba o carga un .pkl en otra pestaña si lo prefieres.")
        else:
            df_pred = df.copy()
            if DEFAULT_TARGET not in df_pred.columns:
                df_pred[DEFAULT_TARGET] = np.nan

            X_pred_raw = df_pred.drop(columns=[DEFAULT_TARGET], errors="ignore")
            X_pred_raw = strip_datetime_cols(X_pred_raw, extra_drop=["_FECHA_", "FECHA"])
            X_pred_ohe = get_dummies_if_checked(X_pred_raw, st.session_state.get("auto_dummies", True))
            X_pred = X_pred_ohe.reindex(columns=st.session_state.feature_cols, fill_value=0)

            for c, dt in st.session_state.feature_dtypes.items():
                try: X_pred[c] = X_pred[c].astype(dt, copy=False)
                except Exception: pass

            mask_nan = df_pred[DEFAULT_TARGET].isna()
            if mask_nan.any():
                preds_all = model.predict(X_pred)
                df_pred.loc[mask_nan, DEFAULT_TARGET] = preds_all[mask_nan.values]
                st.success(f"Rellenadas {int(mask_nan.sum())} filas con predicción en '{DEFAULT_TARGET}'.")
            else:
                st.info("No hay valores vacíos en 'PRECIO M2' para rellenar.")

            st.dataframe(df_pred.head(10), width="stretch")
            bio = io.BytesIO()
            with pd.ExcelWriter(bio, engine="openpyxl") as w:
                df_pred.to_excel(w, index=False, sheet_name="predicciones")
            bio.seek(0)
            st.download_button("💾 Descargar Excel con 'PRECIO M2' rellenado", data=bio.getvalue(),
                               file_name="predicciones_con_PRECIO_M2.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# -----------------------------------------------------------------------------
# TAB 2: MANUAL
# -----------------------------------------------------------------------------
with tab_manual:
    st.subheader("Añadir una fila manual y predecir si 'PRECIO M2' está vacío")

    # 🔔 AVISO visible pero NO bloquea el formulario
    if st.session_state.get("final_model_k", None) is None:
        st.warning("Para poder **predecir** en esta pestaña es necesario **entrenar previamente con un Excel** (pestaña anterior). "
                   "Aun así, puedes introducir filas manualmente: quedarán guardadas en el buffer.")

    with st.form("form_manual"):
        c1, c2, c3 = st.columns(3)
        ano = c1.number_input(DEFAULT_YEAR, 1950, 2100, 2025, step=1)
        mes = c2.number_input(DEFAULT_MONTH, 1, 12, 9, step=1)
        dist = c3.selectbox(DEFAULT_DIST, options=ALLOWED_DISTRICTS, index=ALLOWED_DISTRICTS.index("Centro"))

        st.markdown("**Variables (lag-12, del año anterior):**")
        lag_vars = [c for c in REQUIRED_FEATURES_EXCL_TARGET if c not in {DEFAULT_YEAR, DEFAULT_MONTH, DEFAULT_DIST}]
        values = {}
        for i in range(0, len(lag_vars), 3):
            cols = st.columns(3)
            for j in range(3):
                k = i + j
                if k < len(lag_vars):
                    values[lag_vars[k]] = cols[j].number_input(lag_vars[k], value=0.0, step=0.1, format="%.6f")

        precio_txt = st.text_input(f"{DEFAULT_TARGET} (opcional; déjalo vacío para predecir)", value="")
        submitted = st.form_submit_button("➕ Añadir al buffer")

    if submitted:
        row = {DEFAULT_YEAR: int(ano), DEFAULT_MONTH: int(mes), DEFAULT_DIST: dist}
        for k, v in values.items():
            row[k] = v
        row[DEFAULT_TARGET] = float(precio_txt.replace(",", ".")) if precio_txt.strip() != "" else np.nan
        for col in TEMPLATE_COLUMNS:
            row.setdefault(col, np.nan)
        st.session_state.manual_rows = pd.concat([st.session_state.manual_rows, pd.DataFrame([row])], ignore_index=True)
        st.success("Fila añadida.")

    st.markdown("**Buffer manual**")
    st.dataframe(st.session_state.manual_rows, width="stretch")

    st.subheader("Predecir buffer (solo rellena 'PRECIO M2' vacío)")
    model  = st.session_state.get("final_model_k", None)
    fcols  = st.session_state.get("feature_cols", None)
    fdtype = st.session_state.get("feature_dtypes", None)
    auto_d = st.session_state.get("auto_dummies", True)

    if model is None:
        st.info("⚠️ Aún no hay modelo entrenado: cuando lo entrenes en la pestaña Excel podrás predecir aquí.")
    elif st.session_state.manual_rows.empty:
        st.info("No hay filas manuales.")
    else:
        dfm = st.session_state.manual_rows.copy()
        dfm, _ = clean_distrito_column(dfm, DEFAULT_DIST)
        dfm = dfm[~dfm[DEFAULT_DIST].isna()]

        X_raw = dfm.drop(columns=[DEFAULT_TARGET], errors="ignore")
        X_raw = strip_datetime_cols(X_raw, extra_drop=["_FECHA_", "FECHA"])
        X_ohe = get_dummies_if_checked(X_raw, auto_d)
        X_pred = X_ohe.reindex(columns=fcols, fill_value=0)
        for c, dt in fdtype.items():
            try:
                X_pred[c] = X_pred[c].astype(dt, copy=False)
            except Exception:
                pass

        if DEFAULT_TARGET not in dfm.columns:
            dfm[DEFAULT_TARGET] = np.nan

        mask_nan = dfm[DEFAULT_TARGET].isna()
        if mask_nan.any():
            preds = model.predict(X_pred)
            dfm.loc[mask_nan, DEFAULT_TARGET] = preds[mask_nan.values]
            st.success(f"Rellenadas {int(mask_nan.sum())} filas con predicción en '{DEFAULT_TARGET}'.")
        else:
            st.info("No hay valores vacíos en 'PRECIO M2' en el buffer manual.")

        st.dataframe(dfm, width="stretch")
        st.session_state.manual_rows = dfm

        bio2 = io.BytesIO()
        with pd.ExcelWriter(bio2, engine="openpyxl") as w:
            dfm.to_excel(w, index=False, sheet_name="manual_pred")
        bio2.seek(0)
        st.download_button("💾 Descargar filas manuales con 'PRECIO M2'", data=bio2.getvalue(),
                           file_name="filas_manuales_con_PRECIO_M2.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
