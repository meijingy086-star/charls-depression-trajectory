"""Run after export: python verify_model.py"""
from pathlib import Path
import json
import sys

from predictor import TrajectoryModel

path = Path(__file__).absolute().parent / "artifacts" / "model.json.gz"
if not path.exists():
    sys.exit("尚未找到 artifacts/model.json.gz。请先在原 RStudio 会话运行 export_model.R。")
model = TrajectoryModel.load(path)
print(json.dumps({"status":"passed","trees":len(model.trees),"raw_inputs":len(model.inputs),
    "encoded_features":len(model.features),"r_python_cases":model.verify_export(),
    "r_side_checks":model.data.get("parity"),"sha256":model.digest},ensure_ascii=False,indent=2))
