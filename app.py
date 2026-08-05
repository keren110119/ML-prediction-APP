"""
Ming Wang Sales Forecasting App
================================
Upload your sales/product/inventory files (and optionally product photos),
process them into a modeling table, train a LightGBM forecasting model,
and look up predictions by style.

RUN LOCALLY:
    pip install streamlit lightgbm pandas openpyxl scikit-learn joblib
    streamlit run app.py

OPTIONAL (for image-based features - needs internet access to download models):
    pip install rembg onnxruntime transformers torch
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import zipfile
import joblib
from datetime import datetime

st.set_page_config(page_title="Ming Wang Sales Forecast", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Inter:wght@400;500;600&family=Jost:wght@400;500&display=swap');

:root {
    --mw-ivory: #FAF6F0;
    --mw-charcoal: #2B2420;
    --mw-gold: #A6803C;
    --mw-sapphire: #2A3F6B;
    --mw-line: #E4DCCC;
}

.stApp { background-color: var(--mw-ivory); }

h1, h2, h3 {
    font-family: 'Cormorant Garamond', serif !important;
    color: var(--mw-charcoal) !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em;
}

h1 { font-size: 2.6rem !important; margin-bottom: 0 !important; }

p, div, span, label, .stMarkdown { font-family: 'Inter', sans-serif; color: var(--mw-charcoal); }

.stCaption, [data-testid="stCaptionContainer"] {
    font-family: 'Jost', sans-serif !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.72rem !important;
    color: var(--mw-gold) !important;
}

/* Tabs styled like garment-label tabs */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--mw-line); }
.stTabs [data-baseweb="tab"] {
    font-family: 'Jost', sans-serif;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.82rem;
    color: var(--mw-charcoal);
    background-color: transparent;
    border: none;
    padding: 12px 4px;
}
.stTabs [aria-selected="true"] {
    color: var(--mw-gold) !important;
    border-bottom: 2px solid var(--mw-gold) !important;
}

/* Buttons */
.stButton > button {
    font-family: 'Jost', sans-serif;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.8rem;
    background-color: var(--mw-charcoal);
    color: var(--mw-ivory);
    border: none;
    border-radius: 2px;
    padding: 0.6rem 1.4rem;
}
.stButton > button:hover { background-color: var(--mw-gold); color: white; }

/* Metric cards -> spec-tag style */
[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid var(--mw-line);
    border-radius: 2px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"] {
    font-family: 'Jost', sans-serif !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: 0.7rem !important;
    color: var(--mw-gold) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Cormorant Garamond', serif !important;
    color: var(--mw-charcoal) !important;
    font-size: 2rem !important;
}

/* Divider under title, like a care-label rule */
.mw-rule { border-top: 1px solid var(--mw-line); margin: 0.6rem 0 1.6rem 0; }

/* File uploader + selects */
[data-testid="stFileUploader"], .stSelectbox, .stTextInput {
    font-family: 'Inter', sans-serif;
}
section[data-testid="stFileUploaderDropzone"] {
    background-color: white;
    border: 1px dashed var(--mw-line);
}
</style>
""", unsafe_allow_html=True)

# ---- Simple shared-password gate (Community Cloud apps are public by default) ----
APP_PASSWORD = "mingwang2026"  # <-- change this to your own shared password before deploying

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("""
    <style>
    .mw-login-wrap {
        display: flex; justify-content: center; margin-top: 8vh;
    }
    .mw-login-card {
        background: white;
        border: 1px solid var(--mw-line);
        border-radius: 2px;
        padding: 2.6rem 3rem;
        width: 380px;
        text-align: center;
    }
    .mw-login-card h1 { font-size: 2.2rem !important; margin-bottom: 0.1rem !important; }
    .mw-login-tag {
        font-family: 'Jost', sans-serif;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 0.7rem;
        color: var(--mw-gold);
        margin-bottom: 1.6rem;
    }
    </style>
    <div class="mw-login-wrap"><div class="mw-login-card">
        <h1>Ming Wang</h1>
        <div class="mw-login-tag">Sales Forecasting Studio</div>
    </div></div>
    """, unsafe_allow_html=True)

    _, center_col, _ = st.columns([1, 1.15, 1])
    with center_col:
        pw = st.text_input("Team password", type="password", label_visibility="collapsed",
                            placeholder="Enter team password")
        if st.button("Enter", use_container_width=True):
            if pw == APP_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password.")
    st.stop()

# Try to import optional image-embedding dependencies
IMAGE_DEPS_AVAILABLE = True
try:
    from rembg import remove
    from transformers import CLIPModel, CLIPProcessor
    import torch
    from sklearn.decomposition import PCA
    from PIL import Image
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()  # lets PIL.Image.open read .heic/.heif files
    except ImportError:
        pass
except ImportError:
    IMAGE_DEPS_AVAILABLE = False


# =============================================================================
# SESSION STATE
# =============================================================================
if "modeling_table" not in st.session_state:
    st.session_state.modeling_table = None
if "model" not in st.session_state:
    st.session_state.model = None
if "feature_importance" not in st.session_state:
    st.session_state.feature_importance = None
if "metrics" not in st.session_state:
    st.session_state.metrics = None
if "image_features" not in st.session_state:
    st.session_state.image_features = None


# =============================================================================
# STEP 1: DATA PROCESSING FUNCTIONS
# =============================================================================

def parse_sku(sku):
    """STYLE-COLORCODE-SIZE -> (style, color_code)"""
    if pd.isna(sku):
        return None, None
    parts = str(sku).split("-")
    style = parts[0]
    color_code = parts[1] if len(parts) > 1 else None
    return style, color_code


def read_any_spreadsheet(file_obj, **kwargs):
    """Reads CSV, TSV, XLS, or XLSX regardless of which one it actually is,
    detecting the format from the filename extension."""
    name = file_obj.name.lower()
    if name.endswith((".xlsx", ".xls", ".xlsm")):
        return pd.read_excel(file_obj, **{k: v for k, v in kwargs.items() if k != "low_memory"})
    elif name.endswith(".tsv"):
        return pd.read_csv(file_obj, sep="\t", **kwargs)
    else:  # .csv and anything else - default to comma-separated
        return pd.read_csv(file_obj, **kwargs)


def clean_order_files(order_files, exclude_emails=None):
    """Load and combine Shopify order exports (any spreadsheet format) into clean line-item data."""
    exclude_emails = exclude_emails or []
    cols_needed = ["Name", "Email", "Financial Status", "Created at", "Lineitem quantity",
                   "Lineitem price", "Lineitem compare at price", "Lineitem sku", "Cancelled at"]

    dfs = []
    for f in order_files:
        d = read_any_spreadsheet(f, low_memory=False)
        available = [c for c in cols_needed if c in d.columns]
        dfs.append(d[available])

    li = pd.concat(dfs, ignore_index=True)
    li = li.dropna(subset=["Lineitem sku"])
    if exclude_emails:
        li = li[~li["Email"].isin(exclude_emails)]
    if "Cancelled at" in li.columns:
        li = li[li["Cancelled at"].isna()]
    li["Created at"] = pd.to_datetime(li["Created at"], errors="coerce", utc=True).dt.tz_localize(None)
    li = li.dropna(subset=["Created at"])

    parsed = li["Lineitem sku"].apply(parse_sku)
    li["style"] = parsed.apply(lambda x: x[0])
    li["color_code"] = parsed.apply(lambda x: x[1])
    li["item_key"] = li["style"] + "_" + li["color_code"].fillna("NA")
    return li


def build_weekly_panel(li):
    """Aggregate line items to a complete weekly panel per item (fills zero-sales weeks)."""
    li["week"] = li["Created at"].dt.to_period("W").dt.start_time

    weekly = li.groupby(["item_key", "style", "color_code", "week"]).agg(
        qty_sold=("Lineitem quantity", "sum"),
        avg_price=("Lineitem price", "mean"),
        avg_compare_price=("Lineitem compare at price", "mean"),
    ).reset_index()

    panels = []
    for item, grp in weekly.groupby("item_key"):
        full_weeks = pd.date_range(grp["week"].min(), grp["week"].max(), freq="W-MON")
        panels.append(pd.DataFrame({"week": full_weeks, "item_key": item}))
    full_panel = pd.concat(panels, ignore_index=True)

    merged = full_panel.merge(weekly, on=["item_key", "week"], how="left")
    merged["qty_sold"] = merged["qty_sold"].fillna(0)
    merged = merged.sort_values(["item_key", "week"])
    for c in ["style", "color_code", "avg_price", "avg_compare_price"]:
        merged[c] = merged.groupby("item_key")[c].ffill().bfill()
    return merged


def add_lag_features(panel):
    g = panel.groupby("item_key")["qty_sold"]
    panel["lag_1w"] = g.shift(1)
    panel["lag_2w"] = g.shift(2)
    panel["lag_4w"] = g.shift(4)
    panel["roll_mean_4w"] = g.shift(1).rolling(4).mean().reset_index(drop=True)
    panel["roll_mean_8w"] = g.shift(1).rolling(8).mean().reset_index(drop=True)
    panel["discount_pct"] = np.where(
        panel["avg_compare_price"] > 0,
        (panel["avg_compare_price"] - panel["avg_price"]) / panel["avg_compare_price"] * 100, 0
    )
    panel["week_of_year"] = panel["week"].dt.isocalendar().week.astype(int)
    panel["month"] = panel["week"].dt.month
    return panel


def merge_netsuite_attrs(panel, netsuite_file):
    ns = read_any_spreadsheet(netsuite_file)
    attr_cols = [c for c in ["Color Family", "Fit", "Garment Length", "Sleeve Length", "Neckline",
                              "Closure Type", "Sleeve Type", "Silhouette (Dress/Skirt)",
                              "Material Type", "Category"] if c in ns.columns]
    ns_attrs = ns.groupby("Style #")[attr_cols].first().reset_index().rename(columns={"Style #": "style"})
    return panel.merge(ns_attrs, on="style", how="left")


def merge_inventory(panel, inventory_files):
    snapshots = []
    for f in inventory_files:
        name = f.name.lower()
        if name.endswith((".xlsx", ".xls")):
            raw = pd.read_excel(f, header=None)
            date_line = str(raw.iloc[3, 0]).strip().strip('"')
            snap_date = pd.to_datetime(date_line.replace("As of ", ""))
            d = pd.read_excel(f, skiprows=6)
        else:
            content = f.read().decode("utf-8", errors="ignore")
            lines = content.split("\n")
            date_line = lines[3].strip().strip('"')
            snap_date = pd.to_datetime(date_line.replace("As of ", ""))
            d = pd.read_csv(io.StringIO(content), skiprows=6)

        d.columns = [c.strip() for c in d.columns]
        if "Item" not in d.columns or "On Hand" not in d.columns:
            continue
        d = d[d["Item"].astype(str).str.match(r"^[MLP]\d{4,6}[A-Z]{0,3}-")].copy()
        d["On Hand"] = d["On Hand"].astype(str).str.replace(",", "").str.replace("(", "-").str.replace(")", "")
        d["On Hand"] = pd.to_numeric(d["On Hand"], errors="coerce").fillna(0)
        d["snapshot_date"] = snap_date
        snapshots.append(d[["Item", "On Hand", "snapshot_date"]])

    if not snapshots:
        panel["on_hand_qty"] = -1
        panel["likely_in_stock"] = False
        return panel

    inv = pd.concat(snapshots, ignore_index=True)
    parts = inv["Item"].astype(str).str.split("-")
    inv["style"] = parts.str[0]
    inv["color_code"] = parts.str[1]
    inv["item_key"] = inv["style"] + "_" + inv["color_code"]
    inv_agg = inv.groupby(["item_key", "snapshot_date"])["On Hand"].sum().reset_index()
    inv_agg["join_month"] = inv_agg["snapshot_date"].dt.to_period("M").astype(str)

    panel["join_month"] = panel["week"].dt.to_period("M").astype(str)
    panel = panel.merge(inv_agg[["item_key", "join_month", "On Hand"]], on=["item_key", "join_month"], how="left")
    panel["likely_in_stock"] = panel["On Hand"].notna()
    panel["on_hand_qty"] = panel["On Hand"].fillna(-1)
    panel = panel.drop(columns=["On Hand", "join_month"])
    return panel


# =============================================================================
# IMAGE FEATURE FUNCTIONS (only run if optional deps are installed)
# =============================================================================

VIEW_CODES = {"F", "B", "S", "D", "FL", "ML"}

def parse_image_filename(filename):
    base = filename.rsplit(".", 1)[0]
    parts = base.split("_")
    if len(parts) == 1:
        return parts[0], None, None
    elif len(parts) == 2:
        return (parts[0], None, parts[1]) if parts[1].upper() in VIEW_CODES else (parts[0], None, parts[1])
    else:
        return parts[0], "_".join(parts[1:-1]), parts[-1]


@st.cache_resource
def load_fashion_clip():
    try:
        model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
        processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
    except Exception:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor


def get_embedding(img, model, processor):
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    if hasattr(features, "squeeze"):
        return features.squeeze().numpy()
    elif hasattr(features, "image_embeds"):
        return features.image_embeds.squeeze().numpy()
    return features.pooler_output.squeeze().numpy()


def process_images(image_files, n_pca=24):
    model, processor = load_fashion_clip()
    records, embeddings = [], []
    progress = st.progress(0.0, text="Processing images...")

    for i, f in enumerate(image_files):
        style, color_code, view = parse_image_filename(f.name)
        try:
            img_bytes = remove(f.read())
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            emb = get_embedding(img, model, processor)
            records.append({"filename": f.name, "style": style, "color_code": color_code, "view": view})
            embeddings.append(emb)
        except Exception as e:
            st.warning(f"Skipped {f.name}: {e}")
        progress.progress((i + 1) / len(image_files), text=f"Processing images... ({i+1}/{len(image_files)})")

    progress.empty()
    if not embeddings:
        return None

    df_meta = pd.DataFrame(records)
    df_meta["item_key"] = df_meta["style"] + "_" + df_meta["color_code"].fillna("SINGLE")
    emb_df = pd.DataFrame(np.vstack(embeddings))
    emb_df["item_key"] = df_meta["item_key"].values
    item_embeddings = emb_df.groupby("item_key").mean()

    n_components = min(n_pca, item_embeddings.shape[0], item_embeddings.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(item_embeddings.values)
    df_final = pd.DataFrame(reduced, index=item_embeddings.index,
                             columns=[f"img_pca_{i+1}" for i in range(n_components)]).reset_index()
    return df_final


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_model(df, test_weeks=8):
    import lightgbm as lgb
    from sklearn.metrics import mean_absolute_error

    df = df.sort_values("week")
    numeric_feats = [c for c in ["lag_1w", "lag_2w", "lag_4w", "roll_mean_4w", "roll_mean_8w",
                                  "avg_price", "discount_pct", "week_of_year", "month",
                                  "on_hand_qty"] if c in df.columns]
    numeric_feats += [c for c in df.columns if c.startswith("img_pca_")]
    categorical_feats = [c for c in ["style", "color_code", "Color Family", "Fit", "Category",
                                      "Silhouette (Dress/Skirt)", "Neckline", "Sleeve Type",
                                      "Material Type"] if c in df.columns]

    model_df = df[numeric_feats + categorical_feats + ["qty_sold", "week"]].copy()
    for c in categorical_feats:
        model_df[c] = model_df[c].astype("category")

    cutoff = model_df["week"].max() - pd.Timedelta(weeks=test_weeks)
    train = model_df[model_df["week"] <= cutoff]
    test = model_df[model_df["week"] > cutoff]

    X_train, y_train = train[numeric_feats + categorical_feats], train["qty_sold"]
    X_test, y_test = test[numeric_feats + categorical_feats], test["qty_sold"]

    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31,
                               min_child_samples=20, random_state=42, verbosity=-1)
    model.fit(X_train, y_train, categorical_feature=categorical_feats)

    preds = np.clip(model.predict(X_test), 0, None)
    mae = mean_absolute_error(y_test, preds)
    wmape = np.sum(np.abs(y_test - preds)) / max(np.sum(y_test), 1) * 100

    baseline_preds = X_test["lag_1w"].fillna(0) if "lag_1w" in X_test.columns else np.zeros(len(y_test))
    baseline_mae = mean_absolute_error(y_test, baseline_preds)

    importance = pd.DataFrame({
        "feature": numeric_feats + categorical_feats,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    metrics = {"mae": mae, "wmape": wmape, "baseline_mae": baseline_mae,
               "n_train": len(train), "n_test": len(test),
               "train_end": train["week"].max(), "test_start": test["week"].min()}

    return model, importance, metrics, numeric_feats, categorical_feats


# =============================================================================
# STREAMLIT UI
# =============================================================================

st.markdown("<h1>Ming Wang</h1>", unsafe_allow_html=True)
st.caption("Sales Forecasting Studio")
st.markdown('<div class="mw-rule"></div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Upload & Process", "Train Model", "Predict"])

# ---- TAB 1: UPLOAD & PROCESS ----
with tab1:
    st.markdown("### Upload Your Files")

    col1, col2 = st.columns(2)
    with col1:
        order_files = st.file_uploader("Shopify order exports (CSV, TSV, XLS, or XLSX - can select multiple)",
                                        type=["csv", "tsv", "xls", "xlsx"], accept_multiple_files=True, key="orders")
        netsuite_file = st.file_uploader("NetSuite sales/attribute spreadsheet (CSV, XLS, or XLSX)",
                                          type=["csv", "xls", "xlsx"], key="netsuite")
    with col2:
        inventory_files = st.file_uploader("Monthly inventory snapshots (CSV, XLS, or XLSX - can select multiple)",
                                            type=["csv", "xls", "xlsx"], accept_multiple_files=True, key="inventory")
        image_features_file = st.file_uploader(
            "Image features CSV (from the Colab pipeline - recommended instead of raw photos)",
            type=["csv"], key="image_features_csv")
        with st.expander("Or upload raw photos instead (slower, processed in-app)"):
            image_files = st.file_uploader("Product photos (any common image format, can select multiple)",
                                            type=["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp", "gif", "heic", "heif"],
                                            accept_multiple_files=True, key="images")

    exclude_emails_raw = st.text_input(
        "Email addresses to exclude (comma-separated - e.g. internal/comp/PR orders)",
        placeholder="e.g. pr-samples@yourcompany.com, comp@yourcompany.com"
    )
    exclude_emails = [e.strip() for e in exclude_emails_raw.split(",") if e.strip()]

    if not IMAGE_DEPS_AVAILABLE and image_files:
        st.warning(
            "Image processing libraries (`rembg`, `transformers`, `torch`) aren't installed in this "
            "environment, so photo-based features will be skipped. Install with:\n\n"
            "`pip install rembg onnxruntime transformers torch`\n\n"
            "The rest of the pipeline (sales + attributes + inventory) will still run normally."
        )

    if st.button("Process Data", type="primary", disabled=not (order_files and netsuite_file)):
        with st.spinner("Cleaning order data..."):
            li = clean_order_files(order_files, exclude_emails)
            st.success(f"Loaded {len(li):,} valid order line items")

        with st.spinner("Building weekly sales panel..."):
            panel = build_weekly_panel(li)
            panel = add_lag_features(panel)
            st.success(f"Built weekly panel: {len(panel):,} item-week rows, {panel['item_key'].nunique():,} unique items")

        with st.spinner("Merging NetSuite attributes..."):
            panel = merge_netsuite_attrs(panel, netsuite_file)

        if inventory_files:
            with st.spinner("Merging inventory snapshots..."):
                panel = merge_inventory(panel, inventory_files)
        else:
            panel["on_hand_qty"] = -1
            panel["likely_in_stock"] = False

        if image_features_file is not None:
            with st.spinner("Merging pre-computed image features..."):
                img_feats = pd.read_csv(image_features_file)
                if "item_key" not in img_feats.columns:
                    st.error("This CSV needs an 'item_key' column (style_colorcode) to merge - "
                             "check that you ran the latest version of the Colab script.")
                else:
                    panel = panel.merge(img_feats, on="item_key", how="left")
                    st.session_state.image_features = img_feats
                    n_matched = panel["item_key"].isin(img_feats["item_key"]).sum()
                    st.success(f"Merged image features for {img_feats['item_key'].nunique()} items "
                               f"({n_matched:,} of {len(panel):,} rows matched)")
        elif image_files and IMAGE_DEPS_AVAILABLE:
            with st.spinner("Extracting image features (this can take a while)..."):
                img_feats = process_images(image_files)
                if img_feats is not None:
                    panel = panel.merge(img_feats, on="item_key", how="left")
                    st.session_state.image_features = img_feats
                    st.success(f"Extracted visual features for {len(img_feats)} items")

        st.session_state.modeling_table = panel
        st.success("Data processed and ready for training.")

    if st.session_state.modeling_table is not None:
        st.subheader("Preview")
        st.dataframe(st.session_state.modeling_table.head(50))
        st.caption(f"Full table: {len(st.session_state.modeling_table):,} rows, "
                   f"{st.session_state.modeling_table['item_key'].nunique():,} unique items")

# ---- TAB 2: TRAIN MODEL ----
with tab2:
    st.markdown("### Train The Model")

    if st.session_state.modeling_table is None:
        st.info('Process your data in "Upload & Process" first.')
    else:
        test_weeks = st.slider("Weeks to hold out for testing", 4, 16, 8)

        if st.button("Train Model", type="primary"):
            with st.spinner("Training LightGBM model..."):
                model, importance, metrics, num_feats, cat_feats = train_model(
                    st.session_state.modeling_table, test_weeks)
                st.session_state.model = model
                st.session_state.feature_importance = importance
                st.session_state.metrics = metrics
                st.session_state.num_feats = num_feats
                st.session_state.cat_feats = cat_feats

        if st.session_state.metrics:
            m = st.session_state.metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Model MAE", f"{m['mae']:.2f} units")
            col2.metric("Naive baseline MAE", f"{m['baseline_mae']:.2f} units")
            col3.metric("wMAPE", f"{m['wmape']:.1f}%")
            st.caption(f"Trained on {m['n_train']:,} rows through {m['train_end'].date()}, "
                       f"tested on {m['n_test']:,} rows from {m['test_start'].date()} onward")

            st.subheader("What's driving predictions")
            st.bar_chart(st.session_state.feature_importance.set_index("feature").head(15))

# ---- TAB 3: PREDICT ----
with tab3:
    st.markdown("### Look Up A Forecast")

    if st.session_state.model is None:
        st.info('Train a model in "Train Model" first.')
    else:
        df = st.session_state.modeling_table
        styles = sorted(df["style"].dropna().unique())
        selected_style = st.selectbox("Style #", styles)

        colors = sorted(df[df["style"] == selected_style]["color_code"].dropna().unique())
        selected_color = st.selectbox("Color code", colors) if colors else None

        item_key = f"{selected_style}_{selected_color or 'NA'}"
        item_rows = df[df["item_key"] == item_key].sort_values("week")

        if len(item_rows) == 0:
            st.warning("No history found for this item.")
        else:
            latest = item_rows.iloc[-1]
            feat_cols = st.session_state.num_feats + st.session_state.cat_feats
            X_pred = item_rows[feat_cols].iloc[[-1]].copy()
            for c in st.session_state.cat_feats:
                X_pred[c] = X_pred[c].astype("category")

            pred = max(0, st.session_state.model.predict(X_pred)[0])

            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Predicted units next week", f"{pred:.1f}")
                st.caption(f"Last actual (week of {latest['week'].date()}): {latest['qty_sold']:.0f} units")
                if "Category" in latest:
                    st.write(f"**Category:** {latest.get('Category', 'N/A')}")
                if "Color Family" in latest:
                    st.write(f"**Color:** {latest.get('Color Family', 'N/A')}")
                if "Fit" in latest:
                    st.write(f"**Fit:** {latest.get('Fit', 'N/A')}")
            with col2:
                st.line_chart(item_rows.set_index("week")["qty_sold"])
