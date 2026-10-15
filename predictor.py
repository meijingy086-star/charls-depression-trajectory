"""Inference for the existing R randomForest; no retraining or R runtime needed."""
import gzip
import hashlib
import json
import math
from pathlib import Path


class ModelError(ValueError):
    pass


def finite_number(value):
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _integer(value):
    return finite_number(value) and int(value) == float(value)


class TrajectoryModel:
    def __init__(self, data, digest=""):
        self.data = data
        self.digest = digest
        self._validate()
        self.classes = data["classes"]
        self.inputs = data["inputs"]
        self.features = data["features"]
        self.trees = data["trees"]

    @classmethod
    def load(cls, path):
        path = Path(path)
        with path.open("rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        with gzip.open(str(path), "rt", encoding="utf-8") as handle:
            data = json.load(handle)
        model = cls(data, digest)
        model.verify_export()
        return model

    def _validate(self):
        d = self.data
        if d.get("format") != "charls-rf-v1":
            raise ModelError("不支持的模型格式。请重新运行随附的 R 导出脚本。")
        classes = d.get("classes", [])
        if not isinstance(classes, list) or set(classes) != set("Class" + str(i) for i in range(1, 5)) or len(classes) != 4:
            raise ModelError("模型必须包含四个唯一的轨迹类别。")
        specs = d.get("inputs", [])
        features = d.get("features", [])
        trees = d.get("trees", [])
        if not specs or not features or not trees:
            raise ModelError("模型输入定义、特征或决策树为空。")
        names = [s.get("name") for s in specs]
        if len(set(names)) != len(names) or any(not n for n in names):
            raise ModelError("原始输入变量名重复或为空。")
        spec_map = dict((s["name"], s) for s in specs)
        for s in specs:
            if s.get("type") not in ("number", "category"):
                raise ModelError("不支持的输入类型。")
            if s["type"] == "category":
                options = s.get("options", [])
                codes = [o.get("value") for o in options]
                if not codes or len(set(codes)) != len(codes) or any(not isinstance(c, str) for c in codes):
                    raise ModelError("分类水平配置无效。")
        feature_names = [f.get("name") for f in features]
        if len(set(feature_names)) != len(feature_names):
            raise ModelError("编码特征名重复。")
        for f in features:
            if f.get("original") not in spec_map:
                raise ModelError("编码特征缺少原始变量。")
            s = spec_map[f["original"]]
            if f.get("type") != s["type"]:
                raise ModelError("原始变量与编码特征的类型不一致。")
            if f["type"] == "category":
                values = f.get("values", {})
                if set(values) != set(o["value"] for o in s["options"]) or not all(finite_number(v) for v in values.values()):
                    raise ModelError("分类编码规则不完整。")
        if d.get("n_trees") != len(trees):
            raise ModelError("决策树数量与元数据不一致。")
        for tree in trees:
            if not isinstance(tree, list) or not tree:
                raise ModelError("决策树为空。")
            for row in tree:
                if not isinstance(row, list) or len(row) != 6 or not all(finite_number(v) for v in row):
                    raise ModelError("决策树节点格式错误。")
                if not all(_integer(row[i]) for i in (0, 1, 2, 4, 5)):
                    raise ModelError("决策树节点索引必须为整数。")
                status = int(row[4])
                if status == -1:
                    if not 1 <= int(row[5]) <= len(classes):
                        raise ModelError("叶节点的类别编号无效。")
                elif status == 1:
                    if not 1 <= int(row[2]) <= len(features):
                        raise ModelError("树分裂变量编号无效。")
                    if not all(1 <= int(row[i]) <= len(tree) for i in (0, 1)):
                        raise ModelError("树子节点编号无效。")
                else:
                    raise ModelError("决策树节点状态无效。")
            # Reject cycles and shared children instead of risking an inference loop.
            pending, visited = [1], set()
            while pending:
                node = pending.pop()
                if node in visited:
                    raise ModelError("决策树包含循环或重复子节点。")
                visited.add(node)
                row = tree[node - 1]
                if int(row[4]) != -1:
                    pending.extend([int(row[0]), int(row[1])])

    def validate_inputs(self, raw):
        if not isinstance(raw, dict):
            raise ModelError("输入必须为字段映射。")
        cleaned, warnings = {}, []
        for s in self.inputs:
            name, label = s["name"], s.get("label", s["name"])
            value = raw.get(name)
            if value is None or value == "":
                raise ModelError("请填写：" + label)
            if s["type"] == "category":
                if isinstance(value, bool):
                    raise ModelError(label + "的类别无效。")
                code = str(value)
                if code not in [o["value"] for o in s["options"]]:
                    raise ModelError(label + "的类别不在开发集编码范围内。")
                cleaned[name] = code
            else:
                if not finite_number(value):
                    raise ModelError(label + "必须为有限数值。")
                value = float(value)
                cleaned[name] = value
                lo, hi = s.get("observed_min"), s.get("observed_max")
                if lo is not None and hi is not None and not float(lo) <= value <= float(hi):
                    warnings.append("{0}超出开发集观测范围（{1:g}–{2:g} {3}），请核对输入，模型在此范围外的表现未知。".format(label, float(lo), float(hi), s.get("unit", "")))
        return cleaned, warnings

    def encode(self, raw):
        cleaned, warnings = self.validate_inputs(raw)
        vector = []
        for f in self.features:
            value = cleaned[f["original"]]
            if f["type"] == "category":
                value = f["values"][value]
            vector.append(float(value))
        return vector, warnings

    def predict_encoded(self, vector):
        if len(vector) != len(self.features) or not all(finite_number(v) for v in vector):
            raise ModelError("编码后的特征长度或数值无效。")
        votes = [0] * len(self.classes)
        for tree in self.trees:
            node = 1
            for _ in range(len(tree) + 1):
                row = tree[node - 1]
                if int(row[4]) == -1:
                    votes[int(row[5]) - 1] += 1
                    break
                feature = int(row[2]) - 1
                node = int(row[0] if vector[feature] <= row[3] else row[1])
            else:
                raise ModelError("树预测未终止。")
        return [v / float(len(self.trees)) for v in votes]

    def predict(self, raw):
        vector, warnings = self.encode(raw)
        probabilities = self.predict_encoded(vector)
        top_index = max(range(len(probabilities)), key=lambda i: probabilities[i])
        return {"class": self.classes[top_index], "classes": list(self.classes),
                "probabilities": probabilities, "warnings": warnings,
                "tie": sum(abs(p - probabilities[top_index]) < 1e-12 for p in probabilities) > 1}

    def verify_export(self):
        cases = self.data.get("verification_cases", [])
        if not cases:
            raise ModelError("模型缺少与原 R 预测进行对照的核对样本。")
        for case in cases:
            vector, _ = self.encode(case["inputs"])
            expected_x = case.get("encoded", [])
            if len(expected_x) != len(vector) or any(abs(a - b) > 1e-10 for a, b in zip(vector, expected_x)):
                raise ModelError("网页编码与 R 编码核对失败。")
            got = self.predict_encoded(vector)
            expected = case.get("probabilities", [])
            if len(expected) != len(got) or any(abs(a - b) > 1e-12 for a, b in zip(got, expected)):
                raise ModelError("网页预测与原 R 模型核对失败。")
        return len(cases)
