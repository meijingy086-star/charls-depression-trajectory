# Depressive Symptom Trajectory Predictor

A Streamlit interface for the exported R `randomForest` model. Python traverses the original R trees; the website does not retrain the model.

## Current model

- Exported on 22 October 2026.
- 500 trees; `mtry = 3`, `nodesize = 25`, `maxnodes = 30`.
- 15 original input variables and 20 encoded features.
- Development set: 1,867 participants; holdout set: 801 participants.
- Final fitting uses development-set up-sampling. Holdout participants are not up-sampled.
- The four-class holdout AUC recorded in the export is approximately 0.634. See `model_info.json` for the exported aggregate metrics and trajectories.
- Class display names were reviewed against the exported development-group means. Class numbers are model-specific and must be rechecked after refitting.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Use `app.py` as the Streamlit entry point. The model is loaded from `artifacts/model.json.gz`; its fingerprint is part of the cache key.

## Verify

```bash
python verify_model.py
python -m unittest discover -s tests -p "test_predictor.py" -v
python -m unittest discover -s tests -p "test_presentation.py" -v
```

The artifact contains synthetic verification profiles with their R encodings and prediction probabilities. Loading the model checks Python predictions against these reference values. No participant IDs or individual source records are included in those verification profiles.

## Export a later fitted R model

In the R session containing the fitted model, encoder, and development data:

```r
options(trajectory_export_dir = "PATH/TO/THIS/REPOSITORY")
# Optional: set this only after reviewing the newly fitted trajectories.
web_class_labels <- c(
  Class1 = "G1: Persistent Low Group",
  Class2 = "G2: Increasing Group",
  Class3 = "G3: U-Shaped Group",
  Class4 = "G4: Persistent High Group"
)
source("export_model.R", encoding = "UTF-8")
```

The exporter requires `rf_model`, `encoder`, `train_completed`, `train_set`, `model_feature_cols`, and `feature_names`. It verifies feature encoding and tree predictions before writing the artifact. Never copy class names to a refitted model without checking the new trajectories.

Run Python verification before committing a new artifact. Update `model_info.json` and class labels for the same release. The website SHAP explanation uses the exported development medians/modes as a single reference profile; the R cohort SHAP plots use sampled development backgrounds, so their numerical explanations need not coincide.
