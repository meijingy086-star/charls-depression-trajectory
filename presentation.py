"""Display names for model-defined inputs and explicitly verified classes."""

INPUT_LABELS = {
    "age": "Age", "gender": "Gender", "marry": "Marital status",
    "edu": "Education level", "retire": "Retirement status",
    "systo": "Systolic blood pressure", "diasto": "Diastolic blood pressure",
    "bmi": "BMI", "drinkl": "Alcohol consumption", "social": "Social participation",
    "chronic": "Chronic disease", "hibpe": "Hypertension", "diabe": "Diabetes",
    "arther": "Arthritis", "disability": "Disability", "hear": "Hearing",
    "eyesight_close": "Near vision", "eyesight_distance": "Distance vision",
    "sleep": "Sleep duration", "satlife": "Life satisfaction",
}

CATEGORY_LABELS = {
    "gender": {"1": "Female", "2": "Male"},
    "marry": {"1": "Other", "2": "Married"},
    "edu": {"1": "Less than primary school", "2": "Primary school",
            "3": "Middle school", "4": "High school or above"},
    "retire": {"1": "Not retired", "2": "Retired"},
    "hear": {"1": "Good", "2": "Fair", "3": "Poor"},
    "eyesight_close": {"1": "Good", "2": "Fair", "3": "Poor"},
    "eyesight_distance": {"1": "Good", "2": "Fair", "3": "Poor"},
    "satlife": {"1": "Poor", "2": "Fair", "3": "Good"},
}
BINARY_VARIABLES = {"drinkl", "social", "chronic", "hibpe", "diabe", "arther", "disability"}
UNITS = {"age": "years", "systo": "mmHg", "diasto": "mmHg",
         "bmi": "kg/m²", "sleep": "hours/night"}


def class_display_names(model_data, classes):
    configured = model_data.get("class_labels", {})
    if not isinstance(configured, dict):
        configured = {}
    return {name: configured.get(name) or name.replace("Class", "Class ")
            for name in classes}


def input_display_name(spec):
    return spec.get("label_en") or INPUT_LABELS.get(spec["name"], spec["name"].replace("_", " ").title())


def category_display_labels(spec):
    name = spec["name"]
    known = CATEGORY_LABELS.get(name, {"1": "No", "2": "Yes"} if name in BINARY_VARIABLES else {})
    return {str(option["value"]): option.get("label_en") or
            known.get(str(option["value"]), "Code " + str(option["value"]))
            for option in spec["options"]}
