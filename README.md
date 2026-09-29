# GRCF

Grey-Relational Contrastive Fusion，用于可检查的在线课程推荐。

本仓库提供论文的模型与实验代码、XuetangX（XTX）和 MARS 数据副本，以及现有主实验和消融结果。论文框架图及其复现材料不包含在本仓库中。

## 目录

```text
reproducibility/
  code/grcf_lib.py                 数据读取、特征、EW-GRA、模型和训练
  code/run_experiments.py          实验入口
  code/build_integrated_assets.py  从保存结果生成三张 LaTeX 表
  data/raw/xuetangx/               XTX 原始数据及来源 Readme
  data/raw/mars_*.tab              MARS 原始数据文件
  results/main.json                2 数据集 × 3 种子 × 9 方法
  results/ablation.json            8 配置 × 2 种子
DATA_SOURCES.md                    数据来源及引用
ARTIFACTS_SHA256.json              文件完整性清单
verify_artifacts.py                校验文件与结果覆盖范围
```

## 环境

建议 Python 3.10，新建独立环境后安装：

```bash
python -m pip install -r requirements.txt
python verify_artifacts.py
```

依赖版本来自本地实际可运行环境。训练使用 CPU；默认 2 线程，可通过 `GRCF_THREADS` 修改。

## 读取保存结果

```bash
python reproducibility/code/build_integrated_assets.py
```

输出至 `tables/`。`main.json` 为补充材料中的完整汇总结果，并非本次上传时重新跑出的全部实验。主实验 XTX 的 GRCF 平均 AUC 为 0.9257；MARS 为 0.7465。

## 重新运行实验

实验脚本支持断点续跑，已存在的结果会跳过。为重新计算且保留随仓库提供的结果，请给 `GRCF_WORK` 设置新的输出目录。

Linux/macOS：

```bash
GRCF_WORK="$PWD/runs/reproduction" python reproducibility/code/run_experiments.py main
GRCF_WORK="$PWD/runs/reproduction" python reproducibility/code/run_experiments.py ablation
```

Windows PowerShell：

```powershell
$env:GRCF_WORK = Join-Path (Get-Location) 'runs/reproduction'
python reproducibility/code/run_experiments.py main
python reproducibility/code/run_experiments.py ablation
```

原始数据始终从 `reproducibility/data/raw` 读取；输出目录中建立 `cache/` 和 `results/`。XTX 使用 8:1:1 随机交互划分，每个正交互配两个未观察负样本；MARS 将评分 >=6 视为正标签。

`run_experiments.py` 还保留补充材料中的其他实验入口；本仓库随附的完整结果仅覆盖 `main` 和 `ablation`。这些其他入口不代表已完成的结果。

## 数据与权利

请引用原始数据作者，详细链接见 [DATA_SOURCES.md](DATA_SOURCES.md)。第三方数据保留原权利与原始使用条件，本仓库未为其授予新的许可。课程文本、匿名交互与数值结果用于本研究复现。
