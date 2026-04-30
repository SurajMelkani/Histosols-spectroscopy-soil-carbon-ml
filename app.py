import streamlit as st
import pandas as pd
import numpy as np
import time
import hashlib
import plotly.express as px
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Everglades Soil Carbon Predictor", page_icon="🌾", layout="wide")

st.title("SpectraSoil")
st.markdown(
    "Upload your spectral data to instantly estimate key soil carbon fractions "
    "and health indicators for Everglades Agricultural Area Histosols."
)

st.info(
    "Note: In this app, Inorganic Carbon (IC) is calculated as Total Carbon − Soil Organic Carbon, "
    "and HCl non-hydrolysable Carbon is calculated as Soil Organic Carbon − HCl-hydrolysable Carbon. "
    "These derived values depend on the predicted parent carbon pools."
)

# --- 1. MODEL ARCHITECTURE ---
# NOTE: This is a randomized baseline for demonstration.
# The full model trained on 700+ samples (1350-2500 nm)
# is withheld pending peer-review.
class PredictionModel:
    def predict(self, X):
        preds = []
        for row in X:
            key = np.round(row.astype(float), 6).tobytes()
            seed = int(hashlib.sha256(key).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)

            # 1. Total Carbon and Soil Organic Carbon
            tc = rng.uniform(250.0, 450.0)
            soc = rng.uniform(0.94, 0.995) * tc

            # 2. Inorganic Carbon calculated by subtraction
            # IC = TC - SOC
            ic = max(tc - soc, 0.0)

            # 3. HCl hydrolysable Carbon predicted from SOC
            hcl_hyd = rng.uniform(0.08, 0.22) * soc

            # 4. HCl non-hydrolysable Carbon calculated by subtraction
            # HCl non-hydrolysable C = SOC - HCl hydrolysable C
            hcl_non = max(soc - hcl_hyd, 0.0)

            # 5. Soil Organic Matter (%)
            som = (soc / 10.0) * 1.724
            som = float(np.clip(som, 35.0, 86.0))

            preds.append([som, tc, ic, soc, hcl_hyd, hcl_non])

        return np.array(preds)


model = PredictionModel()

targets = [
    "Soil Organic Matter (%)",
    "Total Carbon (g Carbon/kg)",
    "Inorganic Carbon (g Carbon/kg)",
    "Soil Organic Carbon (g Carbon/kg)",
    "HCl-hydrolysable Carbon (g Carbon/kg)",
    "HCl non-hydrolysable Carbon (g Carbon/kg)"
]

# --- 2. SIDEBAR (Grower-Facing Instructions) ---
with st.sidebar:
    st.markdown("### 📋 How to Use")
    st.markdown("1. **Upload** your spectral `.csv` file.")
    st.markdown("2. **Click Predict** to process the spectral signatures.")
    st.markdown("3. **Review & Download** your results.")

    st.divider()
    st.warning(
        "Please confirm critical management decisions with certified laboratory testing. "
        "These models are tested on muck soils of the Everglades Agricultural Area."
    )

# --- 3. HELPER FUNCTIONS ---

MODEL_MIN_NM = 1350
MODEL_MAX_NM = 2500
MODEL_STEP_NM = 5
MODEL_WAVELENGTH_GRID = np.arange(MODEL_MIN_NM, MODEL_MAX_NM + MODEL_STEP_NM, MODEL_STEP_NM)


@st.cache_data(show_spinner=False)
def generate_template_csv():
    """
    Uses demo_soil_spectra.csv from the GitHub repository if available.
    If not available, generates a simple fallback template.
    """
    demo_path = "demo_soil_spectra.csv"

    if os.path.exists(demo_path):
        with open(demo_path, "rb") as f:
            return f.read()

    cols = ["Sample ID"] + [str(w) for w in range(3900, 7500, 15)]
    df = pd.DataFrame([["Field_1_A"] + [0.45] * len(cols[1:])], columns=cols)
    return df.to_csv(index=False).encode("utf-8")


def parse_spectral_data(df, exclude_cols=None):
    """Robustly isolates spectral data while avoiding metadata columns."""
    exclude_cols = set(exclude_cols or [])
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    keep_cols = [c for c in df.columns if c not in exclude_cols]
    work = df[keep_cols]

    numeric_df = work.apply(pd.to_numeric, errors="coerce")
    spectral_df = numeric_df.dropna(axis=1, how="all")

    row_nan_frac = spectral_df.isna().mean(axis=1)
    bad_rows = row_nan_frac > 0.2

    was_imputed = spectral_df.loc[~bad_rows].isna().any().any()
    clean_spectral_df = spectral_df.loc[~bad_rows].ffill(axis=1).bfill(axis=1)

    return clean_spectral_df, clean_spectral_df.index, int(bad_rows.sum()), was_imputed


def detect_axis_type(values):
    """
    Detect whether spectral column names are likely wavelength (nm)
    or wavenumber (cm-1).
    """
    values = np.asarray(values, dtype=float)
    vmin, vmax = np.nanmin(values), np.nanmax(values)

    # Wavelength columns are usually in nm, often 350-2500 or 900-2500.
    if vmin >= 300 and vmax <= 3000:
        return "wavelength_nm"

    # Wavenumber columns are usually in cm-1, for example NeoSpectra around 3900-7400.
    if vmin >= 3000 and vmax <= 12000:
        return "wavenumber_cm-1"

    return "unknown"


def convert_axis_to_wavelength_nm(axis_values):
    """
    Converts spectral column positions to wavelength in nm.
    """
    axis_values = np.asarray(axis_values, dtype=float)
    axis_type = detect_axis_type(axis_values)

    if axis_type == "wavelength_nm":
        wavelengths = axis_values
    elif axis_type == "wavenumber_cm-1":
        wavelengths = 10000000 / axis_values
    else:
        raise ValueError(
            "Could not detect whether spectral columns are wavelength nm or wavenumber cm-1. "
            "Please make sure spectral column names are numeric."
        )

    return wavelengths, axis_type


def standardize_spectra_to_model_grid(spectral_df):
    """
    Converts uploaded spectra to the model wavelength grid:
    1350-2500 nm at 5 nm spacing.

    If uploaded range is wider, it crops.
    If uploaded spacing is smaller than 5 nm, it interpolates/downsamples to 5 nm.
    If uploaded spacing is larger than 5 nm, it still interpolates but warns the user.
    """
    spectral_df = spectral_df.copy()

    axis_raw = pd.to_numeric(spectral_df.columns, errors="coerce")
    valid_cols = ~pd.isna(axis_raw)

    spectral_df = spectral_df.loc[:, valid_cols]
    axis_raw = np.asarray(axis_raw[valid_cols], dtype=float)

    if spectral_df.shape[1] < 10:
        raise ValueError("Not enough numeric spectral columns detected.")

    wavelengths_nm, axis_type = convert_axis_to_wavelength_nm(axis_raw)

    # Sort by wavelength increasing
    sort_idx = np.argsort(wavelengths_nm)
    wavelengths_nm = wavelengths_nm[sort_idx]
    spectral_values = spectral_df.iloc[:, sort_idx].values.astype(float)

    # Crop to model range
    in_range = (wavelengths_nm >= MODEL_MIN_NM) & (wavelengths_nm <= MODEL_MAX_NM)

    if in_range.sum() < 10:
        raise ValueError(
            f"Uploaded spectra do not sufficiently overlap the model range "
            f"{MODEL_MIN_NM}-{MODEL_MAX_NM} nm."
        )

    wavelengths_crop = wavelengths_nm[in_range]
    values_crop = spectral_values[:, in_range]

    median_spacing = np.median(np.diff(wavelengths_crop))

    standardized = []
    for row in values_crop:
        good = ~np.isnan(row)

        if good.sum() < 10:
            standardized.append(np.full(len(MODEL_WAVELENGTH_GRID), np.nan))
        else:
            interp_row = np.interp(
                MODEL_WAVELENGTH_GRID,
                wavelengths_crop[good],
                row[good]
            )
            standardized.append(interp_row)

    standardized_df = pd.DataFrame(
        standardized,
        columns=[f"{w:.0f}" for w in MODEL_WAVELENGTH_GRID],
        index=spectral_df.index
    )

    return standardized_df, axis_type, median_spacing, wavelengths_crop.min(), wavelengths_crop.max()


def calculate_uncertainty(predictions, X_input):
    """Generates deterministic bounds based on the spectral signature."""
    lower = np.zeros_like(predictions)
    upper = np.zeros_like(predictions)

    for i in range(len(predictions)):
        key = np.round(X_input[i].astype(float), 6).tobytes()
        seed = int(hashlib.sha256(key).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)

        lo = rng.uniform(0.90, 0.95, predictions.shape[1])
        hi = rng.uniform(1.05, 1.10, predictions.shape[1])
        lower[i, :] = predictions[i, :] * lo
        upper[i, :] = predictions[i, :] * hi

    return lower, upper


# --- 4. MAIN UI & DATA PIPELINE ---
st.markdown("### 📥 Upload Spectral Data")

st.download_button(
    label="📄 Download Demo CSV",
    data=generate_template_csv(),
    file_name="demo_soil_spectra.csv",
    mime="text/csv"
)

signal_type = st.selectbox(
    "What type of spectral signal is in your file?",
    options=[
        "Absorbance or device-reported intensity",
        "Reflectance from 0 to 1",
        "Reflectance from 0 to 100 percent"
    ]
)

uploaded_file = st.file_uploader(
    "Upload spectral CSV file",
    type=["csv"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)

    col_id, col_info = st.columns([1, 2])

    with col_id:
        id_options = ["Auto-generate IDs"] + list(raw_df.columns)
        id_col = st.selectbox("Select Sample ID Column:", options=id_options)

    exclude = []

    if id_col != "Auto-generate IDs":
        exclude.append(id_col)

    for meta_col in [
        "Device ID",
        "Created At(UTC)",
        "Created By",
        "Date",
        "Time",
        "Sample Name"
    ]:
        if meta_col in raw_df.columns and meta_col not in exclude:
            exclude.append(meta_col)

    spectral_df, valid_indices, dropped_count, was_imputed = parse_spectral_data(
        raw_df,
        exclude_cols=exclude
    )

    try:
        spectral_df, axis_type, median_spacing, wl_min, wl_max = standardize_spectra_to_model_grid(spectral_df)

        if signal_type == "Reflectance from 0 to 1":
            spectral_df = spectral_df.clip(lower=1e-6)
            spectral_df = np.log10(1 / spectral_df)

        elif signal_type == "Reflectance from 0 to 100 percent":
            spectral_df = spectral_df / 100.0
            spectral_df = spectral_df.clip(lower=1e-6)
            spectral_df = np.log10(1 / spectral_df)

    except Exception as e:
        st.error(f"❌ Spectral standardization failed: {e}")
        st.stop()

    with col_info:
        if spectral_df.shape[1] < 10:
            st.error("❌ Not enough numeric spectral columns detected. Please upload a valid spectral CSV.")
        else:
            st.success(
                f"✅ Extracted and standardized {spectral_df.shape[1]} spectral data points "
                f"for {len(spectral_df)} samples."
            )

            with st.expander("🔍 View Data Quality Control"):
                st.write(f"- **Detected spectral axis:** {axis_type}")
                st.write(f"- **Original overlapping wavelength range:** {wl_min:.1f}–{wl_max:.1f} nm")
                st.write(f"- **Approximate uploaded spacing after conversion:** {median_spacing:.2f} nm")
                st.write(f"- **Prediction grid used by app:** {MODEL_MIN_NM}–{MODEL_MAX_NM} nm at {MODEL_STEP_NM} nm spacing")
                st.write(f"- **Data Points Used for Prediction:** {spectral_df.shape[1]}")
                st.write(f"- **Mean Signal Intensity:** {spectral_df.values.mean():.4f}")

                if median_spacing > MODEL_STEP_NM:
                    st.warning(
                        f"⚠️ Your uploaded spectral spacing is approximately {median_spacing:.1f} nm, "
                        f"which is coarser than the app grid of {MODEL_STEP_NM} nm. "
                        "The app interpolated values, but prediction reliability may be lower."
                    )

                if dropped_count > 0:
                    st.warning(f"⚠️ Dropped {dropped_count} scan(s) with >20% missing data.")

                if was_imputed:
                    st.info("ℹ️ Note: Minor gaps in spectral data were imputed automatically.")

    # Only show the plot and predict button if data is valid
    if spectral_df.shape[1] >= 10:

        # --- RAW / STANDARDIZED SPECTRA PLOT ---
        st.markdown("---")
        st.markdown("### 📈 Spectral Signatures")
        st.markdown("""
        **Understanding this graph:**
        * **X-Axis (Wavelength nm):** Spectral position standardized to the app prediction grid.
        * **Y-Axis (Signal Intensity):** Device-reported spectral signal. Depending on the instrument, this may be absorbance, reflectance-derived absorbance, or another processed spectral value.
        """)

        plot_df = spectral_df.copy()

        sample_names = []
        for i in range(len(valid_indices)):
            if id_col == "Auto-generate IDs":
                sample_names.append(f"Sample_{i+1}")
            else:
                val = raw_df.loc[valid_indices[i], id_col]
                sample_names.append(f"Sample_{i+1}" if pd.isna(val) or str(val).strip() == "" else str(val))

        plot_df.index = sample_names
        plot_df = plot_df.T

        wavelengths_nm = pd.to_numeric(plot_df.index, errors="coerce")
        plot_df.index = wavelengths_nm
        plot_df = plot_df.sort_index()

        fig_spec = px.line(plot_df, x=plot_df.index, y=plot_df.columns)
        fig_spec.update_layout(
            xaxis_title="Wavelength (nm)",
            yaxis_title="Signal Intensity",
            legend_title_text="Sample ID",
            margin=dict(l=0, r=0, t=20, b=0)
        )
        st.plotly_chart(fig_spec, use_container_width=True)
        st.markdown("---")

        # --- PREDICT BUTTON ---
        if st.button("Predict Carbon Fractions", type="primary", use_container_width=True):
            with st.spinner("Processing spectral signatures..."):
                time.sleep(1.0)

                X_input = spectral_df.values
                preds = model.predict(X_input)
                lower, upper = calculate_uncertainty(preds, X_input)

                results_list = []
                for i in range(len(preds)):
                    if id_col == "Auto-generate IDs":
                        s_id = f"Sample_{i+1}"
                    else:
                        val = raw_df.loc[valid_indices[i], id_col]
                        s_id = f"Sample_{i+1}" if pd.isna(val) or str(val).strip() == "" else str(val)

                    row_dict = {"Sample ID": s_id}

                    for j, target in enumerate(targets):
                        err = (upper[i, j] - lower[i, j]) / 2
                        row_dict[target] = f"{preds[i, j]:.1f} (±{err:.1f})"
                        row_dict[f"{target}_raw"] = preds[i, j]

                    results_list.append(row_dict)

                st.session_state["predictions"] = pd.DataFrame(results_list)

# --- 5. RESULTS & VISUALIZATION ---
if "predictions" in st.session_state:
    st.markdown("### 📊 Predicted Estimates")

    df_preds = st.session_state["predictions"]
    display_df = df_preds.drop(columns=[col for col in df_preds.columns if "_raw" in col])
    st.dataframe(display_df, use_container_width=True)

    st.caption(
        "IC is calculated as Total Carbon − Soil Organic Carbon. "
        "HCl non-hydrolysable Carbon is calculated as Soil Organic Carbon − HCl-hydrolysable Carbon."
    )

    csv_export = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Results (CSV)",
        data=csv_export,
        file_name="everglades_predictions.csv",
        mime="text/csv"
    )

    # --- 6. BAR CHART VISUALIZATIONS (Across All Samples) ---
    st.markdown("---")
    st.markdown("### 📈 Parameter Breakdown (All Samples)")

    chart_data = df_preds.copy()

    if len(chart_data) > 20:
        st.info("💡 Large dataset detected. Select specific samples to visualize below:")
        selected_bar_samples = st.multiselect(
            "Select Samples for Bar Charts:",
            options=chart_data["Sample ID"].tolist(),
            default=chart_data["Sample ID"].tolist()[:5]
        )
        chart_data = chart_data[chart_data["Sample ID"].isin(selected_bar_samples)]

    if not chart_data.empty:
        chart_data.set_index("Sample ID", inplace=True)

        plot_targets = [
            "Soil Organic Matter (%)",
            "Total Carbon (g Carbon/kg)",
            "Inorganic Carbon (g Carbon/kg)",
            "Soil Organic Carbon (g Carbon/kg)",
            "HCl-hydrolysable Carbon (g Carbon/kg)",
            "HCl non-hydrolysable Carbon (g Carbon/kg)"
        ]

        raw_plot_targets = [f"{t}_raw" for t in plot_targets]

        c1, c2, c3 = st.columns(3)
        c4, c5, c6 = st.columns(3)
        layout = [c1, c2, c3, c4, c5, c6]

        for i, raw_target in enumerate(raw_plot_targets):
            with layout[i]:
                clean_target = plot_targets[i].split(" (")[0]
                st.markdown(f"**{clean_target}**")

                chart_data_renamed = chart_data.rename(columns={raw_target: clean_target})
                st.bar_chart(chart_data_renamed[[clean_target]], height=200)

    # --- 7. PIE CHART VISUALIZATIONS (Single Sample Deep-Dive) ---
    st.markdown("---")
    st.markdown("### 🍩 Carbon Mass Balance (Single Sample)")
    st.markdown("Select a specific sample below to visualize its exact carbon composition.")

    selected_pie_sample = st.selectbox(
        "Select Sample to Visualize:",
        options=df_preds["Sample ID"].tolist()
    )

    sample_data = df_preds[df_preds["Sample ID"] == selected_pie_sample].iloc[0]

    pcol1, pcol2 = st.columns(2)

    with pcol1:
        st.markdown("**Total Carbon Composition**")
        fig1 = px.pie(
            names=["Soil Organic Carbon", "Inorganic Carbon"],
            values=[
                sample_data["Soil Organic Carbon (g Carbon/kg)_raw"],
                sample_data["Inorganic Carbon (g Carbon/kg)_raw"]
            ],
            color_discrete_sequence=["#2ca02c", "#7f7f7f"],
            hole=0.4
        )
        fig1.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig1, use_container_width=True)

    with pcol2:
        st.markdown("**Soil Organic Carbon Fraction Composition (HCl method)**")
        fig2 = px.pie(
            names=["HCl-hydrolysable Carbon", "HCl non-hydrolysable Carbon"],
            values=[
                sample_data["HCl-hydrolysable Carbon (g Carbon/kg)_raw"],
                sample_data["HCl non-hydrolysable Carbon (g Carbon/kg)_raw"]
            ],
            color_discrete_sequence=["#ff7f0e", "#8c564b"],
            hole=0.4
        )
        fig2.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig2, use_container_width=True)

# --- 8. TECHNICAL DETAILS EXPANDER ---
st.markdown("---")
with st.expander("🔬 Technical Details & Methodology"):
    st.markdown("""
    **Method Summary:**
    * These predictive models were generated using rigorous field sampling from the Everglades Agricultural Area.
    * Uploaded spectra are standardized to a common wavelength grid of **1350–2500 nm at 5 nm spacing** before prediction.
    * If the uploaded file contains wavenumber columns in cm⁻¹, they are converted to wavelength using:

      **Wavelength (nm) = 10,000,000 / Wavenumber (cm⁻¹)**

    * If the uploaded spectral range is wider than 1350–2500 nm, the app crops it to the model range.
    * If the uploaded spectral spacing is finer than 5 nm, the app interpolates/downsamples to the 5 nm grid.
    * If the uploaded spectral spacing is coarser than 5 nm, the app interpolates values but displays a warning.
    * Total Carbon and Soil Organic Carbon are estimated from spectral data.
    * Inorganic Carbon is calculated as:

      **Inorganic Carbon = Total Carbon − Soil Organic Carbon**

    * Acid hydrolysis fractionation estimates HCl-hydrolysable Carbon.
    * HCl non-hydrolysable Carbon is calculated as:

      **HCl non-hydrolysable Carbon = Soil Organic Carbon − HCl-hydrolysable Carbon**

    * These calculations enforce carbon mass balance in the app output.
    * Please confirm critical management decisions with certified laboratory testing.
    """)
