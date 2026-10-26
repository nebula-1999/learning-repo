# Interface-Induced Trajectory Censoring：学习资料

论文 *Interface-Induced Trajectory Censoring: When Tool-Call Interfaces Distort Evaluation and RL Experience* 的配套学习资料。

- 代码与实验仓库：[nebula-1999/Interface-Induced-Trajectory-Censoring](https://github.com/nebula-1999/Interface-Induced-Trajectory-Censoring)
- 文中出现的 `analysis/…`、`p3/…`、`runs/…` 等路径都指代码仓库里的文件；可点击的链接已指向代码仓库的 `main` 分支。

## 文档

| 文档 | 组织方式 | 什么时候读 |
|---|---|---|
| [PAPER_STUDY_GUIDE.md](PAPER_STUDY_GUIDE.md) | 按论文逐节、逐表、逐数字（对照 arXiv v2） | 要把论文本身讲清楚、查某个数字的口径和出处 |
| [PROJECT_LEARNING_GUIDE.md](PROJECT_LEARNING_GUIDE.md) | 按系统、代码路径、实验、工程教训 | 想弄懂项目是怎么做出来的、要接手代码或再开机跑实验 |
| [ORIGINAL_CODING_AGENT_GUIDE.md](ORIGINAL_CODING_AGENT_GUIDE.md) | 最初那个 coding agent 的工作方式 | 先看"写代码→测→改"的原始 agent 长什么样 |
| [PROJECT_RETROSPECTIVE.md](PROJECT_RETROSPECTIVE.md) | 时间线复盘日志 | 看某次翻车的原始经过（部分解释已过时，以前两份为准） |
| [learning_lab.py](learning_lab.py) | 不用 GPU 的小练习 | `python3 -B learning_lab.py` |

## 建议顺序

1. `PROJECT_LEARNING_GUIDE.md` §1–§5：项目全景、工具调用链路、四条代码路径、指标。
2. `PAPER_STUDY_GUIDE.md` §1–§3：论文主张、前置知识（含统计）、正文逐节。
3. `PROJECT_LEARNING_GUIDE.md` §6–§10：实验逐个读、RL、P3 设计、工程与元教训。
4. `PAPER_STUDY_GUIDE.md` §4–§8：附录、数字速查、误读对照、自测题。
5. 跑 `learning_lab.py`，做两份文档里的练习。
