# Predicting manufacturing capacity
This is a Streamlit app and group of 3 machine learning models to predict production capacity in a regulated manufacturing setting (medical devices).  The models are fully transparent for ISO and FDA audits, based on supervised Random Forest classifiers with gradient boosters.  The model explainability ensures compliance with the ISO 13485 and ISO 14971 standards.

The `cli.py` file exposes a small command line interface used for ingesting data
and training the models.  The latest models are stored in the `models/`
directory and are automatically loaded by the Streamlit dashboard.

## Setup

1. **Install the package and dependencies**

   ```bash
   pip install -e .
   ```

   This installs a `qualitylab` console script that provides commands for
   ingesting data and training the models.

2. **Prepare your data**

   Gather your historical production spreadsheets (`.xlsx`, `.xls` or `.csv`)
   and optional downtime logs.  The file readers normalise column names and
   support both Excel and CSV formats.

## Training the models

1. **Ingest the production data**

   ```bash
   qualitylab ingest path/to/production_sheet1.xlsx path/to/production_sheet2.csv
   ```

   The command concatenates the provided files and writes a
   `data/production.parquet` file inside the project directory.

2. **Train the build-time model**

   ```bash
   qualitylab train-build-time
   ```

   A timestamped model is saved under `models/`.

3. **Train the defect-count model**

   ```bash
   qualitylab train-defects
   ```

4. **Train the build-quantity model**

   ```bash
   qualitylab train-build-quantity path/to/prod1.xlsx path/to/prod2.csv \
       --downtime-files downtime_log1.xlsx --downtime-files downtime_log2.csv
   ```

   Both production and downtime files are required for this step.

After running these commands the `models/` directory will contain the latest
training artefacts.

## Running the dashboard

Launch the Streamlit dashboard to explore predictions and model performance:

```bash
streamlit run streamlit_app.py
```

The app loads the most recent models from `models/` and provides an interface to
upload new spreadsheets, visualise predictions and export results.
