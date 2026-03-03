# 🔬 SpectraSoil: Soil Carbon & Health Analyzer

**A research-driven platform for rapid estimation of soil carbon fractions and soil health indicators using near-infrared spectroscopy.**

[Launch Live SpectraSoil](https://spectroscopy-rapid-soil-carbon-ml.streamlit.app/)

### Overview

SpectraSoil transforms near-infrared (NIR) spectral signatures into actionable soil carbon and soil health metrics.

The platform is designed specifically for organic soils (Histosols) and supports rapid, non-destructive analysis to assist researchers, agronomists, and soil scientists.

### How to Use

Prepare Your Data
Export spectral data from a handheld near-infrared instrument (such as NeoSpectra or similar devices) in .csv or .xlsx format.

Wavelength Requirements
The prediction models are optimized for the 1350–2500 nanometer spectral range.
Ensure your data covers this wavelength window.

Upload & Analyze
Upload your file to the dashboard.
The system will process spectral features and generate estimates for:

Soil organic carbon fractions

Active and stable carbon pools

Soil health indicators

Results are generated instantly through an integrated machine learning pipeline.

### No Data? Use the Demo Dataset
If you do not have raw spectral files available, you can download our **Reference Demo File** below to test the platform:
* 📥 **[Download demo_soil_spectra.csv](./demo_soil_spectra.csv)** *(Contains randomized spectral signatures for 10 Histosol samples to demonstrate the pipeline).*

---



### Technical Note
The core Machine Learning/Deep Learning models were trained on **700+ EAA samples**. To protect intellectual property pending peer-reviewed publication, this public version operates in **Demonstration Mode** using a representative baseline model.


