# 老年人抑郁症状四类轨迹预测平台

本项目使用你已训练的 **R `caret` + `randomForest` 模型**。它不在 Python 中重训随机森林，而是导出原决策树、类别顺序及 `dummyVars` 编码规则，直接计算相同的投票比例。

## 1. 在当前 RStudio 会话导出真实模型

保留已经完成分析的 RStudio 会话。在该会话运行：

```r
source("C:/Users/29183/Documents/Codex/2026-10-15/z/outputs/trajectory_streamlit/export_model.R", encoding = "UTF-8")
```

若提示缺少包，按提示安装 `jsonlite`，再运行导出脚本。无需重新拟合模型。

需要当前会话中已有：`rf_model`、`encoder`、`train_completed`、`train_set`、`model_feature_cols`、`feature_names`。脚本还会尽可能从 `metrics_summary_fixed`、`test_set`、`df_long_dev` 和 `m4_lcgm` 提取研究的汇总指标和轨迹均值。

成功后会生成 `artifacts/model.json.gz`。导出器检查原编码器与便携编码在整个开发集上的一致性，并核对人工合成资料及树分裂边界的预测；网页加载时还会检查 Python 预测是否与保存的 R 对照结果一致。核对失败会停止预测。

没有模型文件时，网页允许预览界面，但不会生成虚构的预测。

## 2. 本地运行

使用 **Python 3.12**，在本项目文件夹打开终端：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe verify_model.py
.venv\Scripts\python.exe -m streamlit run app.py
```

浏览器打开终端显示的地址，通常为 `http://localhost:8501`。

本机的 `work/python312/python.exe` 是本次在工作目录准备的便携解释器，不随项目 ZIP 分发。本地依赖下载受到连接中断影响，尚未装齐；使用前仍需为所选解释器安装 `requirements.txt` 中的依赖。云端已完成完整测试，详见 [VERIFICATION.md](VERIFICATION.md)。

## 3. 发布为 Streamlit 网站

1. 项目已上传至 [meijingy086-star/charls-depression-trajectory](https://github.com/meijingy086-star/charls-depression-trajectory)，包含真实模型文件，无需再手动上传。
2. 若迁移到其他仓库，上传本文件夹内的项目文件，包括隐藏目录 `.streamlit` 和 `artifacts/model.json.gz`。不要再嵌套多一层项目目录。
3. 不上传原始 Excel、受试者逐人数据、完整 `.RData`、论文或 Word 文档；这些都不是网站运行的依赖。
4. 登录 [Streamlit Community Cloud](https://share.streamlit.io/)，选择 Create app，选择部署 GitHub 仓库。
5. Repository 填 `meijingy086-star/charls-depression-trajectory`，Branch 填 `main`，Main file path 填 `app.py`；Python 版本选择 `3.12`，点击 Deploy。
6. 部署成功后，以平台实际返回的 `https://….streamlit.app/` 地址为准；本项目不预设或伪造公开网址。

官方流程：[部署应用](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app)，[依赖管理](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)。

推理仅需 Python，不需要在 Streamlit Cloud 安装 R。通常不需要 `packages.txt`。

## 页面内容

- 基线信息表单：实际所需变量从导出模型读取，避免手填错列或把哑变量当作用户输入。
- 四类结果：最高分组、各类投票比例及本次结果下载。
- 开发集四类实际平均轨迹图：仅为开发集描述，不是个体未来评分预测。
- 验证集四模型比较，以及开发集最大类基线的验证集准确率。
- 方法说明、适用边界和与原 R 输出的一致性核对。

当前代码保存的筛选结果为15个原始变量、20个编码特征：性别、婚姻状况、教育程度、退休、收缩压、BMI、饮酒、社会参与、慢性疾病、关节炎、残疾、听力、近距离视力、睡眠、生活满意度。以最终导出模型为准。

## 类别名称

`study_info.json` 中的 `classes` 可把 `Class 1` 等显示名改为论文最终组名。**必须按本次开发集模型的编号核对**，编号本身不保证按严重程度排序。修改显示名不会改变模型输出。

网页不会对新输入重新拟合 MICE；全部入选变量必须填写。遇到不清楚的测量，应先核实，不能把未知值默认成“无”。“填入参考示例”明确使用开发集中位数/众数组合，不是某个真实受试者。

## 研究解释

最终随机森林在开发集上采样后拟合。输出是未校准的投票比例，应避免称为已经校准的个体绝对风险或“诊断概率”。本次验证 RF 准确率约0.496、宏平均 F1 约0.334、多分类 AUC 约0.615，部分小类识别能力有限。平台用于研究展示，不能替代诊断或治疗决策。

原主分析只使用第一份插补数据，未合并多重插补不确定性；CV 也未逐折重做全部预处理与筛选。本项目保留现有模型，不把上述限制包装成已解决。

应用不把个体输入写入磁盘或数据库，模型资源缓存也不缓存个人预测。托管服务仍按自身政策处理连接及访问日志。若下载结果，请自行妥善保管。

## 验证

```powershell
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
python verify_model.py
```

测试使用的少量人工决策树仅存在于 `tests/`，不会被网页当作真实模型加载。`verify_model.py` 则必须使用 R 导出的真实模型。

仓库中的 GitHub Actions 会验证真实模型对照结果，并测试页面首次载入、缺失输入拦截、填入参考示例、预测和清空表单等行为。

参考论文：A machine-learning-derived online prediction model for depression risk in COPD patients: A retrospective cohort study from CHARLS，DOI：[10.1016/j.jad.2025.02.063](https://doi.org/10.1016/j.jad.2025.02.063)。参考网站和其 GitHub 链接在本次访问时不可用，本项目未复制其源代码。
