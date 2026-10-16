"""CHARLS four-class trajectory research application."""
import datetime
import html
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from predictor import ModelError, TrajectoryModel

ROOT = Path(__file__).absolute().parent
MODEL_PATH = ROOT / "artifacts" / "model.json.gz"
with (ROOT / "study_info.json").open(encoding="utf-8") as handle:
    INFO = json.load(handle)

st.set_page_config(page_title=INFO["title"], page_icon="🌿", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown("""
<style>
.block-container {max-width:1240px;padding-top:2.2rem;padding-bottom:3rem}
h1,h2,h3 {color:#172D37;letter-spacing:-.03em}
div[data-testid="stForm"] {background:white;border:1px solid #E1EAE5;border-radius:18px;padding:25px}
div[data-testid="stMetric"] {background:#FFF;border:1px solid #E1EAE5;border-radius:14px;padding:14px 18px}
.eyebrow {color:#12796D;font-size:.76rem;font-weight:750;letter-spacing:.15em;margin-bottom:12px}
.hero {border-left:4px solid #12796D;padding:3px 0 3px 20px;margin-bottom:26px}
.hero h1 {font-size:2.1rem;margin-bottom:5px}
.hero p {color:#64747B;font-size:1rem;margin-bottom:5px}
.section-label {font-size:.8rem;color:#12796D;font-weight:750;letter-spacing:.1em;margin:14px 0 6px}
.result-box {background:#EAF4EF;border:1px solid #C5DED1;border-radius:16px;padding:22px 24px;margin-bottom:12px}
.result-box h2 {margin:6px 0;font-size:1.9rem}
.result-box p {color:#465E57;margin:0;font-size:.95rem}
.empty-box {background:white;border:1px dashed #C7D5CF;border-radius:18px;padding:35px 25px;min-height:210px;margin-top:2px}
.empty-box h3 {font-size:1.25rem}.empty-box p {color:#6A7C76;line-height:1.8}
.footnote {font-size:.8rem;color:#76847F;line-height:1.8;margin-top:30px;border-top:1px solid #DCE5DF;padding-top:16px}
button[kind="primary"] {border-radius:9px;min-height:44px}
@media(max-width:700px) {.hero h1{font-size:1.65rem}.block-container{padding-top:1.2rem}.hero{padding-left:12px}}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="正在载入研究模型并核对预测…")
def load_model(path, modified_at):
    return TrajectoryModel.load(path)


model, load_error = None, None
if MODEL_PATH.exists():
    try:
        model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
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


def class_color(name):
    return INFO["classes"].get(name, {}).get("color", "#12796D")


def example_profile():
    for s in specs:
        st.session_state["profile_" + s["name"]] = s.get("default")
    st.session_state["result"] = None


def clear_profile():
    for s in specs:
        st.session_state["profile_" + s["name"]] = None
    st.session_state["result"] = None


def field(spec):
    key = "profile_" + spec["name"]
    if spec["type"] == "category":
        labels = dict((o["value"], o["label"]) for o in spec["options"])
        return st.selectbox(spec["label"], list(labels), index=None,
                            format_func=lambda code: labels[code], placeholder="请选择", key=key)
    unit = spec.get("unit", "")
    label = spec["label"] + ("（" + unit + "）" if unit else "")
    return st.number_input(label, value=None, step=0.1, format="%.2f", placeholder="请输入实测值", key=key)


st.markdown('<div class="eyebrow">CHARLS · TRAJECTORY RESEARCH</div>', unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>' + html.escape(INFO["title"]) + '</h1><p>' +
            html.escape(INFO["subtitle"]) + '</p></div>', unsafe_allow_html=True)
if model:
    st.caption("原研究随机森林 · {0}项基线信息 · 四类输出 · 内部验证".format(len(specs)))
else:
    st.info("模型尚未载入，当前可以查看页面；预测功能将在真实研究模型载入后启用。")
    if load_error:
        st.error("模型未通过完整性或预测一致性核对，已停止预测。")
        with st.expander("查看加载错误"):
            st.code(load_error, language=None)

predict_tab, research_tab = st.tabs(["个体预测", "轨迹与研究表现"])
with predict_tab:
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.subheader("输入基线信息")
        st.caption("填写同一基线时点的信息。请完成全部字段；不确定的信息请核实后再预测。")
        b1, b2 = st.columns(2)
        b1.button("填入参考示例", on_click=example_profile, disabled=model is None,
                  help="由开发集的中位数和众数组成，不代表真实受试者。")
        b2.button("清空输入", on_click=clear_profile)
        with st.form("prediction_form"):
            groups = [
                ("基本情况", ["age", "gender", "marry", "edu", "retire"]),
                ("健康状况", ["chronic", "hibpe", "diabe", "arther", "disability", "hear", "eyesight_close", "eyesight_distance"]),
                ("生活方式与感受", ["drinkl", "social", "sleep", "satlife"]),
                ("体格测量", ["systo", "diasto", "bmi"])
            ]
            lookup = dict((s["name"], s) for s in specs)
            ordered = []
            raw = {}
            for title, names in groups:
                current = [lookup[n] for n in names if n in lookup]
                if not current:
                    continue
                st.markdown('<div class="section-label">' + title + '</div>', unsafe_allow_html=True)
                cols = st.columns(2)
                for i, s in enumerate(current):
                    with cols[i % 2]:
                        raw[s["name"]] = field(s)
                    ordered.append(s["name"])
            for s in specs:
                if s["name"] not in ordered:
                    raw[s["name"]] = field(s)
            submitted = st.form_submit_button("预测轨迹类别", type="primary", width="stretch", disabled=model is None)
        if submitted and model:
            st.session_state["result"] = None
            try:
                result = model.predict(raw)
                result["inputs"] = dict(raw)
                result["model_sha256"] = model.digest
                result["created_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                st.session_state["result"] = result
            except ModelError as exc:
                st.error(str(exc))
    with right:
        st.subheader("预测结果")
        result = st.session_state.get("result")
        if result:
            top = max(result["probabilities"])
            label = class_label(result["class"])
            st.markdown('<div class="result-box"><p>模型最可能分配的轨迹类别</p><h2>' +
                        html.escape(label) + '</h2><p>该类投票比例：{0:.1%}</p></div>'.format(top), unsafe_allow_html=True)
            st.caption("下方为随机森林对四类的投票比例，总和为100%。这些分数尚未经概率校准。")
            fig = go.Figure(go.Bar(
                x=result["probabilities"], y=[class_label(c) for c in result["classes"]],
                orientation="h", marker_color=[class_color(c) for c in result["classes"]],
                text=["{0:.1%}".format(p) for p in result["probabilities"]], textposition="outside",
                hovertemplate="%{y}<br>投票比例 %{x:.1%}<extra></extra>"))
            fig.update_layout(height=290, margin=dict(l=5,r=45,t=10,b=15),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(size=14,color="#172D37"),
                              xaxis=dict(range=[0,1.15],tickformat=".0%",title="投票比例",gridcolor="#E4ECE7"),
                              yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            if result["tie"]:
                st.info("存在最高分并列，按 Class 列顺序选择第一类，与研究代码的规则一致。")
            for warning in result["warnings"]:
                st.warning(warning)
            st.info("轨迹类别是对多年症状变化模式的模型预测，不能用于诊断抑郁症或决定治疗。")
            st.download_button("下载本次预测结果", json.dumps(result,ensure_ascii=False,indent=2).encode("utf-8"),
                               file_name="trajectory_prediction.json", mime="application/json")
        else:
            st.markdown('<div class="empty-box"><h3>了解可能的症状变化模式</h3><p>完成左侧基线信息后，点击“预测轨迹类别”。<br>结果将显示最可能的类别，以及四类的模型得分。</p></div>', unsafe_allow_html=True)
        st.caption("提交仅用于本次会话计算；应用不写入个体输入记录或建立个人档案。")

with research_tab:
    st.subheader("这项研究预测什么？")
    st.write("根据基线人口学、健康及生活方式信息，预测受试者在2011—2020年抑郁症状随访中最可能归属的轨迹类别。四类编号沿用开发集轨迹模型，不表示严重程度从低到高排序。")
    dn = model.data.get("development_n") if model else INFO["development_n"]
    vn = model.data.get("validation_n") if model else INFO["validation_n"]
    c1,c2,c3 = st.columns(3)
    c1.metric("开发集", "{0:,} 人".format(dn) if dn else "未提供")
    c2.metric("验证集", "{0:,} 人".format(vn) if vn else "未提供")
    c3.metric("轨迹类别", "4 类")
    trajectory_data = model.data.get("trajectories", []) if model else []
    if trajectory_data:
        fig = go.Figure()
        for cls in model.classes:
            points = sorted([p for p in trajectory_data if p["class"] == cls],key=lambda p:p["year"])
            if points:
                fig.add_trace(go.Scatter(x=[p["year"] for p in points],y=[p["mean"] for p in points],
                    name=class_label(cls),mode="lines+markers",line=dict(color=class_color(cls),width=2.5,dash="dash")))
        fig.update_layout(title="开发集各轨迹类别的实际平均抑郁评分",height=360,
            xaxis=dict(title="随访年份",tickmode="array",tickvals=[2011,2013,2015,2018,2020]),
            yaxis=dict(title="CES-D-10 平均得分"),legend=dict(orientation="h",y=1.12),
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#FFF",margin=dict(l=15,r=15,t=70,b=20))
        st.plotly_chart(fig,width="stretch")
        st.caption("本图为按开发集轨迹归属分组的实际均值，不是对本次输入个体未来各年的评分预测。")
    metrics = model.data.get("metrics", []) if model else []
    if not metrics and (ROOT / "validation_metrics.csv").exists():
        metrics = pd.read_csv(ROOT / "validation_metrics.csv").to_dict("records")
    test_metrics = [r for r in metrics if r.get("Dataset") == "Test"]
    st.subheader("内部验证表现")
    if test_metrics:
        m = next((r for r in test_metrics if r.get("Model") == "RF"),None)
        if m:
            cols = st.columns(3)
            cols[0].metric("RF 准确率", "{0:.1%}".format(m["Accuracy"]))
            cols[1].metric("RF 宏平均 F1", "{0:.3f}".format(m["F1"]))
            cols[2].metric("RF 多分类 AUC", "{0:.3f}".format(m["AUC"]))
        frame = pd.DataFrame(test_metrics)
        frame = frame[[c for c in ["Model","Accuracy","Sensitivity","Precision","F1","AUC"] if c in frame]]
        frame = frame.rename(columns={"Model":"模型","Accuracy":"准确率","Sensitivity":"宏平均召回率",
                                      "Precision":"宏平均精确率","F1":"宏平均 F1","AUC":"多分类 AUC"})
        st.dataframe(frame.round(3),hide_index=True,width="stretch")
    if model:
        dev_counts = model.data.get("development_class_counts",{})
        val_counts = model.data.get("validation_class_counts",{})
        if dev_counts and val_counts:
            majority = max(dev_counts,key=dev_counts.get)
            baseline = val_counts.get(majority,0) / float(sum(val_counts.values()))
            st.caption("开发集最大类为{0}；始终预测该类的验证集准确率为{1:.1%}。基线应结合宏平均指标一起解释。".format(class_label(majority),baseline))
    st.warning("当前模型的验证表现有限，部分小类识别能力较弱。本平台用于研究展示，尚不适合作为临床筛查或个体治疗决策工具。")
    with st.expander("建模方法与使用边界"):
        st.write("按受试者划分开发集与验证集；在开发集中建立四类轨迹模型，再用固定模型确定验证集轨迹归属。预测模型采用开发集筛选的基线变量，并在开发集内调参和上采样。")
        st.write("网页直接使用已有随机森林的决策树及原编码规则，不重新训练、不改变模型，也不对投票比例作事后调整。最终拟合使用了上采样，因此输出不应被解释为已校准的绝对风险。")
        st.write("原研究代码从多重插补结果中使用第一份完整数据训练模型，尚未合并插补不确定性；开发集交叉验证也未逐折重做全部插补及筛选。网页不会补足这些方法学限制。")
        st.write("本页面要求入选变量完整输入，不对新输入重新运行多重插补。验证集属于随机划分的内部验证。")
        if model:
            st.caption("模型版本：" + model.digest[:16] + " · 已通过随附 R 对照样本核对")

st.markdown('<div class="footnote">CHARLS 轨迹研究原型 · 随机森林四分类<br>预测结果仅供科研交流，不能替代专业评估。请勿输入姓名、身份证号或其他身份标识。</div>',unsafe_allow_html=True)
