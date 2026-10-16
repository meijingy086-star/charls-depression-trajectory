"""CHARLS four-class trajectory research application."""

import datetime
import html
import json
from pathlib import Path

import streamlit as st

from predictor import ModelError, TrajectoryModel


ROOT = Path(__file__).absolute().parent
MODEL_PATH = ROOT / "artifacts" / "model.json.gz"

with (ROOT / "study_info.json").open(encoding="utf-8") as handle:
    INFO = json.load(handle)


st.set_page_config(
    page_title=INFO["title"],
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
.block-container {
    max-width:1240px;
    padding-top:2.2rem;
    padding-bottom:3rem;
}
h1,h2,h3 {
    color:#172D37;
    letter-spacing:-.03em;
}
div[data-testid="stForm"] {
    background:white;
    border:1px solid #E1EAE5;
    border-radius:18px;
    padding:25px;
}
.eyebrow {
    color:#12796D;
    font-size:.76rem;
    font-weight:750;
    letter-spacing:.15em;
    margin-bottom:12px;
}
.hero {
    border-left:4px solid #12796D;
    padding:3px 0 3px 20px;
    margin-bottom:26px;
}
.hero h1 {
    font-size:2.1rem;
    margin-bottom:5px;
}
.hero p {
    color:#64747B;
    font-size:1rem;
    margin-bottom:5px;
}
.section-label {
    font-size:.8rem;
    color:#12796D;
    font-weight:750;
    letter-spacing:.1em;
    margin:14px 0 6px;
}
.result-box {
    background:#EAF4EF;
    border:1px solid #C5DED1;
    border-radius:16px;
    padding:22px 24px;
    margin-bottom:20px;
}
.result-box h2 {
    margin:6px 0;
    font-size:1.9rem;
}
.result-box p {
    color:#465E57;
    margin:0;
    font-size:.95rem;
}
.empty-box {
    background:white;
    border:1px dashed #C7D5CF;
    border-radius:18px;
    padding:35px 25px;
    min-height:160px;
    margin-top:2px;
}
.empty-box h3 {
    font-size:1.25rem;
}
.empty-box p {
    color:#6A7C76;
    line-height:1.8;
}
button[kind="primary"] {
    border-radius:9px;
    min-height:44px;
}
@media(max-width:700px) {
    .hero h1 {
        font-size:1.65rem;
    }
    .block-container {
        padding-top:1.2rem;
    }
    .hero {
        padding-left:12px;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="正在载入模型…")
def load_model(path, modified_at):
    return TrajectoryModel.load(path)


model = None
load_error = None

if MODEL_PATH.exists():
    try:
        model = load_model(
            str(MODEL_PATH),
            MODEL_PATH.stat().st_mtime_ns,
        )
    except (ModelError, OSError, ValueError, KeyError, TypeError) as exc:
        load_error = str(exc)


specs = model.inputs if model else INFO["preview_inputs"]
signature = model.digest if model else "preview"

if st.session_state.get("model_signature") != signature:
    for key in list(st.session_state):
        if key.startswith("profile_"):
            del st.session_state[key]

    st.session_state["result"] = None
    st.session_state["model_signature"] = signature


def class_label(name):
    return INFO["classes"].get(name, {}).get("label", name)


def example_profile():
    for spec in specs:
        st.session_state["profile_" + spec["name"]] = spec.get("default")

    st.session_state["result"] = None


def clear_profile():
    for spec in specs:
        st.session_state["profile_" + spec["name"]] = None

    st.session_state["result"] = None


def field(spec):
    key = "profile_" + spec["name"]

    if spec["type"] == "category":
        labels = {
            option["value"]: option["label"]
            for option in spec["options"]
        }

        return st.selectbox(
            spec["label"],
            options=list(labels),
            index=None,
            format_func=lambda code: labels[code],
            placeholder="请选择",
            key=key,
        )

    unit = spec.get("unit", "")
    label = spec["label"] + ("（" + unit + "）" if unit else "")

    return st.number_input(
        label,
        value=None,
        step=0.1,
        format="%.2f",
        placeholder="请输入实测值",
        key=key,
    )


st.markdown(
    '<div class="eyebrow">CHARLS · TRAJECTORY RESEARCH</div>',
    unsafe_allow_html=True,
)

subtitle = INFO.get("subtitle", "")
subtitle_html = (
    "<p>" + html.escape(subtitle) + "</p>"
    if subtitle
    else ""
)

st.markdown(
    '<div class="hero"><h1>'
    + html.escape(INFO["title"])
    + "</h1>"
    + subtitle_html
    + "</div>",
    unsafe_allow_html=True,
)

if model is None:
    st.error("模型尚未成功载入，暂时无法预测。")

    if load_error:
        with st.expander("查看加载错误"):
            st.code(load_error, language=None)


left, right = st.columns([1.15, 1], gap="large")

with left:
    st.subheader("输入基线信息")

    b1, b2 = st.columns(2)

    b1.button(
        "填入参考示例",
        on_click=example_profile,
        disabled=model is None,
    )

    b2.button(
        "清空输入",
        on_click=clear_profile,
    )

    with st.form("prediction_form"):
        groups = [
            (
                "基本情况",
                ["age", "gender", "marry", "edu", "retire"],
            ),
            (
                "健康状况",
                [
                    "chronic",
                    "hibpe",
                    "diabe",
                    "arther",
                    "disability",
                    "hear",
                    "eyesight_close",
                    "eyesight_distance",
                ],
            ),
            (
                "生活方式与感受",
                ["drinkl", "social", "sleep", "satlife"],
            ),
            (
                "体格测量",
                ["systo", "diasto", "bmi"],
            ),
        ]

        lookup = {spec["name"]: spec for spec in specs}
        ordered = []
        raw = {}

        for title, names in groups:
            current = [
                lookup[name]
                for name in names
                if name in lookup
            ]

            if not current:
                continue

            st.markdown(
                '<div class="section-label">'
                + html.escape(title)
                + "</div>",
                unsafe_allow_html=True,
            )

            cols = st.columns(2)

            for i, spec in enumerate(current):
                with cols[i % 2]:
                    raw[spec["name"]] = field(spec)

                ordered.append(spec["name"])

        for spec in specs:
            if spec["name"] not in ordered:
                raw[spec["name"]] = field(spec)

        submitted = st.form_submit_button(
            "预测轨迹类别",
            type="primary",
            width="stretch",
            disabled=model is None,
        )

    if submitted and model:
        st.session_state["result"] = None

        try:
            result = model.predict(raw)
            result["inputs"] = dict(raw)
            result["model_sha256"] = model.digest
            result["created_at"] = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()

            st.session_state["result"] = result

        except ModelError as exc:
            st.error(str(exc))


with right:
    st.subheader("预测结果")

    result = st.session_state.get("result")

    if result:
        label = class_label(result["class"])

        st.markdown(
            '<div class="result-box">'
            "<p>模型最可能分配的轨迹类别</p>"
            "<h2>"
            + html.escape(label)
            + "</h2></div>",
            unsafe_allow_html=True,
        )

        for warning in result.get("warnings", []):
            st.warning(warning)

        st.download_button(
            "下载本次预测结果",
            data=json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8"),
            file_name="trajectory_prediction.json",
            mime="application/json",
        )

    else:
        st.markdown(
            '<div class="empty-box">'
            "<h3>等待预测</h3>"
            "<p>填写左侧信息后，点击“预测轨迹类别”查看结果。</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        
