# Predicting manufacturing capacity
![hero_image](https://github.com/DillonGelman/production-capacity-predictive-models/blob/main/hero-image.png)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gelman-manufacturing-predictions.streamlit.app/)

This is a Streamlit app and group of 3 machine learning models to predict production capacity in a regulated manufacturing setting (medical devices).  The models are fully transparent for ISO and FDA audits, based on supervised Random Forest predictors.  The model explainability ensures compliance with the ISO 13485 and ISO 14971 standards.

The project is organised as a Python package named `qualitylab`.  It exposes a
command line interface for ingesting data and training the models.  Trained
models are stored under the `models/` directory and are automatically loaded by
the Streamlit dashboard. Input data is placed under `data/` and any dashboard
exports are written to `outputs/`.

## Setup

1. **Install the package and dependencies**

   ```bash
   pip install -e .
   ```

   This installs a `qualitylab` console script that provides commands for
   ingesting data and training the models. The package requires several
   additional libraries for model explainability and visualisations.  All
   dependencies are listed in `requirements.txt` so services like
   Streamlit Cloud can install them automatically.
   The pre-trained models were generated with `scikit-learn==1.6.1`, so
   ensure that this version is installed to avoid compatibility issues.

2. **Prepare your data**

   Gather your historical production spreadsheets (`.xlsx`, `.xls` or `.csv`)
   and optional downtime logs.  The file readers normalize column names and
   support both Excel and CSV formats.

## Training the models

1. **Ingest the production data**

   ```bash
   qualitylab ingest path/to/production_sheet1.xlsx
   ```

   The command concatenates the provided files and writes a
   `data/production.parquet` file inside the project directory.

2. **Ingest the downtime logs**

   ```bash
   qualitylab ingest-downtime path/to/downtime_log1.xlsx
   ```

   This writes a `data/downtime.parquet` file used for build quantity training.

3. **Train the build-time model**

   ```bash
   qualitylab train-build-time
   ```

   A timestamped model is saved under `models/`.

4. **Train the defect-count model**

   ```bash
   qualitylab train-defects
   ```

5. **Train the build-quantity model**
   ```bash
   qualitylab train-build-quantity
   ```

**OR you can train all models at once:**

   ```bash
   qualitylab train-all
   ```

After running these commands the `models/` directory will contain the latest
training artefacts.

## Running the dashboard

The app loads the most recent models from `models/` and provides an interface to
upload new spreadsheets, visualise predictions and export results.
