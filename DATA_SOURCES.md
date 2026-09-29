# 数据来源

## XuetangX（XTX）

- 来源：[Course Recommendation / MOOCData](http://moocdata.cn/data/course-recommendation)。
- 原作者项目：[jerryhao66/HRL](https://github.com/jerryhao66/HRL)。该项目明确将上述 MOOCData 页面列为数据入口。
- 对应论文：Jing Zhang, Bowen Hao, Bo Chen, Cuiping Li, Hong Chen, Jimeng Sun. *Hierarchical Reinforcement Learning for Course Recommendation in MOOCs*. AAAI 2019.
- 本地原始说明保留在 `reproducibility/data/raw/xuetangx/Readme`。
- 本实现使用 `data.csv`，GBK 编码；字段为学习者匿名编号、选课时间、课程编号、课程名、类别及类别编号。加载后去除缺失类别并对交互去重。
- 数据包中的 `mooc.*` 文件同时保留；当前实验入口使用 `data.csv` 并自行划分，不读取这些预划分文件。

## MARS

- 来源：[Mandarine Academy Recommender System dataset](https://doi.org/10.7910/DVN/BMY3UD)，Harvard Dataverse。
- 当前模型读取 `mars_explicit_fr.tab`、`mars_items_fr.tab` 和 `mars_users_fr.tab`。
- `mars_implicit_en.tab` 是补充包内附带文件，当前主实验未使用。
- 明确反馈为 0--10 评分，本实现以 6 为正负标签分界。

## 文件一致性

这里的数据文件来自本论文补充包，上传副本的 SHA-256 与本地源文件逐项核对。`ARTIFACTS_SHA256.json` 记录实际上传文件的大小和摘要；运行 `python verify_artifacts.py` 可核验。

数据的原始作者、来源与使用条件不因本仓库托管而改变。本仓库没有为第三方数据新增或变更许可；后续使用者应查阅原始发布方的现行条件。
