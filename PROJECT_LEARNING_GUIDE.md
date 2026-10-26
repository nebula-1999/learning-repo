# 项目学习指南：从一条说不通的训练曲线，到"接口也是测量仪器"

> **写给谁**：项目负责人本人——有算法和模型训练背景，但对 LLM Agent 的服务栈、工具调用、agentic RL 框架还不熟。
>
> **这份文档管什么**：按"**项目本身**"组织——系统是怎么搭的、代码里每条路径谁在干什么、每个实验为什么要做、我们在哪些地方翻过车、学到了什么。
>
> **和其他文档的分工**：
>
> | 文档 | 组织方式 | 什么时候读 |
> |---|---|---|
> | **本文** | 按系统、代码、实验、教训 | 想真正弄懂"这个项目是怎么做出来的"、要接手代码或再开机跑实验 |
> | [PAPER_STUDY_GUIDE.md](PAPER_STUDY_GUIDE.md) | 按论文逐节、逐表、逐数字 | 要把论文本身讲清楚、查某个数字的口径和出处 |
> | [ORIGINAL_CODING_AGENT_GUIDE.md](ORIGINAL_CODING_AGENT_GUIDE.md) | 最初那个 coding agent 的工作方式 | 想先看"写代码→测→改"的原始 agent 长什么样 |
> | [PROJECT_RETROSPECTIVE.md](PROJECT_RETROSPECTIVE.md) | 时间线复盘日志 | 想看某次翻车的原始经过（部分解释已过时，以本文为准） |
>
> **状态**：截至 2026-10。P3 正式对照和共同原生 FC 评测都已完成，论文已投 ICLR 2027 并发布 arXiv 预印本。旧版本文档里"待跑""第 19 章准备阶段"的内容都已过时，本文全部按最终结果改写。

---

## 目录

- [0. 怎么读这份文档](#0-怎么读这份文档)
- [1. 项目一页纸：问题、转折、结论、时间线](#1-项目一页纸问题转折结论时间线)
- [2. 用你熟悉的概念理解 Agent](#2-用你熟悉的概念理解-agent)
- [3. 一次工具调用的完整过程](#3-一次工具调用的完整过程)
- [4. 项目代码里的四条路径（核心章）](#4-项目代码里的四条路径核心章)
- [5. 数据、测试、奖励与指标](#5-数据测试奖励与指标)
- [6. 实验逐个读：问题 → 设计 → 结果 → 边界](#6-实验逐个读问题--设计--结果--边界)
- [7. 从梯度下降理解 agentic RL](#7-从梯度下降理解-agentic-rl)
- [8. P3 是怎么设计出来的](#8-p3-是怎么设计出来的)
- [9. 工程：显存、LoRA、checkpoint、磁盘、Ray](#9-工程显存loracheckpoint磁盘ray)
- [10. 全项目的元教训（每条都对应一次真实事故）](#10-全项目的元教训每条都对应一次真实事故)
- [11. 证据边界：哪些能讲，哪些不能讲](#11-证据边界哪些能讲哪些不能讲)
- [12. 实验设计与统计（精简版）](#12-实验设计与统计精简版)
- [13. 按源码学习：五次阅读路线](#13-按源码学习五次阅读路线)
- [14. 再开机之前：验收清单](#14-再开机之前验收清单)
- [15. 练习（附答案）](#15-练习附答案)
- [16. 术语表与原始资料](#16-术语表与原始资料)

---

## 0. 怎么读这份文档

### 0.1 读完应该达到什么水平

拿到任意一条原始轨迹，你能回答五个问题：

1. **谁产生了这段文本？**（哪个 checkpoint、什么 prompt、什么解码参数）
2. **谁解析了它？**（vLLM 服务端 parser？verl 的 ToolParser？评测器自己的正则？）
3. **谁执行了测试？**（模型发起的工具调用？评测器无条件执行？奖励函数打分？）
4. **反馈有没有真的回到模型？**（以什么角色——tool 消息还是 user 消息？）
5. **最终分数是谁算的？**（训练奖励？EvalPlus script 模式？BFCL 官方 scorer？τ-bench 环境状态？）

能稳定答出这五个问题，就说明你理解了这个项目；背下多少数字反而是次要的。

### 0.2 建议路线

| 目标 | 读哪里 | 产出 |
|---|---|---|
| 先知道项目在干什么 | §1、§2、§3 | 能画出一次工具调用的完整路径 |
| 理解系统 | §4、§5 | 能画出四条代码路径，说清每条路径里 parser 在哪、谁跑测试、谁看到反馈 |
| 理解实验 | §6 | 每个实验能用"问题—改动—结果—边界"讲两分钟 |
| 理解 RL 与 P3 | §7、§8 | 能解释"工具通道恢复了，但学习收益没测到"为什么不矛盾 |
| 能自己跑实验 | §9、§10、§14 | 能写一张 run card，知道开机前要验哪些东西 |
| 能判断结论是否过头 | §11、§12 | 能指出一句话的分母、判据、替代解释 |
| 巩固 | §13、§15 | 读过关键源码，做完练习 |

### 0.3 本文的三种表述

- **【原理】**：通用知识（GRPO 是什么、LoRA 怎么算），不是我们的发现。
- **【项目记录】**：代码、日志、实验表里已有的事实，带来源。
- **【边界】**：明确不能推出的东西。这一类和正面结论一样重要。

所有小代码片段如果不是标了文件名的真实源码，都是教学构造，不是实验数据。

---

## 1. 项目一页纸：问题、转折、结论、时间线

### 1.1 一段话说清项目

> 我们最初想用 RL 训练一个会多轮调试的代码 agent。训练后分数在涨，但拆开看，涨的几乎全是第一轮初稿，多轮"救回"一点都没动。顺着这个异常往下查，发现工具在训练中**从来没有被成功调用过**：Qwen2.5-Coder 输出的调用格式和服务端解析器期待的格式对不上，服务端返回 HTTP 200 加空的 `tool_calls`，看上去和"模型不想调用工具"一模一样。于是项目从"训练一个 agent"转成了"测量接口如何扭曲 agent 的评测和训练经验"：在 BFCL 和 τ-bench 上复现了这个机制，做了 template×parser 的 2×2、规模阶梯、Llama 的反向案例，最后做了一个只换 verl parser 的 7B RL 对照——工具执行从 10 次恢复到 16,844 次，但 held-out 成功率没有检测到提升。

### 1.2 三次关键转折

| 转折 | 之前以为 | 之后知道 | 是什么让我们改主意的 |
|---|---|---|---|
| **转折 1**（08-28 → 08-29） | RL 提升了模型，只是 1.5B 学不会多轮修复 | 训练全程工具调用数为 0，多轮信号从来不存在 | `timing_s/agent_loop/tool_calls` 恒为 0.0（而它其实是秒数）；自己埋的计数器 0 行；`num_turns` 钉在最小值 2 |
| **转折 2**（08-30） | "FC 协议本身压制模型发起调用，ReAct 不压制" | 这个结论不成立：之前的 FC 实验漏了服务端参数；补齐后 Llama 发起率 97%，效应是**家族依赖**的；第一版 Llama 结果的显著性靠 26 条无法归因的空代码撑着 | 一次系统审计找出 8 个洞（§6.2） |
| **转折 3**（09-01） | "parser 静默丢弃了合法调用" | 全档案去重后 4,254 个首轮里，"parser 按自己规则本该接受却没接受"的情况 **0 例**；真正的机制是模型**不产出 parser 要的包装层**（契约不匹配） | 逐行重放 vLLM hermes parser 的失败层分解（§6.4） |

这三次转折是理解整个项目的钥匙：**每一次都是"测量本身出了问题"**——而这恰好就是论文最后的主题。

### 1.3 最终结论（一句话版）

> 工具调用率测的不是模型，而是 **模型 × 模板 × 解析器 × 执行环境** 的组合。接口错配能把评测分数从 0.96 压到 0.00，也能让 RL 训练几乎采不到工具经验；但修好接口恢复的是**机制**（解析、执行、观测），在我们的 7B 单种子设定下**没有检测到**任务成功率的提升。

详细的数字和口径见 [PAPER_STUDY_GUIDE.md §1、§5](PAPER_STUDY_GUIDE.md)。

### 1.4 时间线

| 时间（2026） | 事件 | 产物 |
|---|---|---|
| 08-26 | 项目启动：1.5B 多轮代码 agent 的 GRPO 训练 + 归因分析 | `plan-code-agent.md` |
| 08-28 | 150 步 GRPO 跑完；pass 涨 2.6–2.8 分，91–94% 来自 turn-1；补跑 GRPO seed 1 和 RLOO | `RESULTS.md`、`analyze_all.py`、`runs/step3/` |
| 08-29 | 诊断：训练全程工具调用 0 次（**转折 1**）；加 `guard_tool_calls.sh` 防线 | `code_tool.py` 里的计数器和凭据文件 |
| 08-30 | 审计第一版跨家族实验，找出 8 个洞，推翻"FC 压制发起"（**转折 2**） | `fixes/AUDIT.md` |
| 08-31 | 修 `max_tokens` 记录错误（后来发现修复没生效，见 §10）；下载 hanXen 社区 parser | `runs/final/ERRATA.md`、`runs/final/plugin/` |
| 09-01 | 全档案失败层分解：parser loss = 0（**转折 3**） | `analysis/failure_layer.py` |
| 09-01 → 09-03 | 第二轮外部验证：BFCL v4 clean run + 2×2、τ-bench retail、Qwen3 预注册阶梯、Granite rescue、7B rollout 探针 | `p1/`、`runs/p1_2x2/`、`p5/`、`runs/p4/`、`p3/rollout_probe.py` |
| 09-03 | P3 预注册修正 | `p3/PREREG_AMENDMENT_20260903.md` |
| 09-15 → 09-19 | P3 正式 7B 对照：GPU 验收、两臂训练、repaired 臂中断与恢复、最终证据审计 | `p3/ACCEPTANCE_20260915.md`、`p3/FINAL_RESULT_20260918.md`、`p3/FINAL_EVIDENCE_AUDIT_20260919.md` |
| 09-23 → 09-26 | 审稿后加做共同原生 FC 评测：CPU 准备 → GPU 冒烟 → 全量 | `p3/NATIVE_FC_*.md`、`paper/NATIVE_FC_UPDATE_20260926.md` |
| 之后 | 论文投 ICLR 2027；arXiv 预印本（v2 与投稿版内容一致） | `paper/` |

---

## 2. 用你熟悉的概念理解 Agent

### 2.1 从一次预测变成一个反馈循环

**【原理】** 监督学习：输入 x → 预测 f_θ(x) → 和标签比 → 损失。普通代码生成也是这样：给题目，生成一份程序，结束。

工具型 agent 多了一条**环境反馈回路**：

```text
题目 → 模型生成初稿 → 请求运行测试 → 执行器运行测试
                         ↑                  ↓
                      修改代码 ← 模型读取真实测试反馈
                                           ↓
                                     提交最终答案
```

一个模型 + 一个工具（`run_tests`）+ 一个循环，就是我们研究的 agent。不需要多模型、规划器、记忆系统。

**控制论类比**：模型发出控制指令，系统把它送到执行器，执行器返回观测。如果通信协议让执行器**读不到**指令，控制器再聪明，闭环也转不起来——而且从外面看，就像控制器什么都没发。这就是本项目的核心现象。

### 2.2 四个常被混用的词

| 名称 | 在本项目中的含义 | 不能等同于 |
|---|---|---|
| 模型 model | 权重 + 架构 + tokenizer 定义的文本生成器 | 整个 agent 系统 |
| 策略 policy | 给定当前历史，生成下一段输出的分布 π_θ(·\|h) | 固定的 Python 工作流 |
| Agent | 策略 + 交互循环 + 工具接口 + 停止规则 | 单独一套权重 |
| 工作流 workflow | 程序预先安排的调用和分支 | 模型自主决定的动作 |

判断"模型是否自主使用工具"，不看有没有 while 循环，看**谁决定调用工具、谁提供反馈**。这一点在 §4 区分四条代码路径时非常关键：我们的程序化反馈评测器是"评测器无条件跑测试、把结果塞回去"——那不是模型自主调用。

### 2.3 三种不同的"变化"

```text
读报错后改代码      ：历史 h 变了，θ 不变          （推理时的上下文变化）
RL 训练后再做题     ：θ 变了，策略 π_θ 变了          （学习）
换 parser 后执行恢复：接口 I 变了，θ 可能完全没变    （测量/动作通道的变化）
```

三者都能让最终成功率上升。所以"它最后修好了"本身**不能**说明"它通过 RL 学会了修复"。整个项目的很多误读，都来自把这三种变化混为一谈。

### 2.4 用 MDP 的语言

| RL 概念 | 在本项目中 |
|---|---|
| 状态 | 题目 + 对话历史（模型看不到环境完整状态，更接近部分可观测；τ-bench 里要调 API 才知道订单信息） |
| 动作 | 模型生成的 token 序列；其中"工具调用"是能影响环境的那部分 |
| 观测 | 工具返回的测试结果 |
| 奖励 | 最终代码的测试通过率（训练）/ 是否全通过（评测） |
| episode | 从题目开始，到提交最终答案、达到最大轮数或预算耗尽 |

**关键洞察**：工具动作之后的新观测会改变未来的历史。所以一次解析失败不只是"少记了一个计数"——它让之后的报错、修复、成功路径**根本不会被采样出来**。这就是标题里"trajectory censoring"中 trajectory 的含义。

### 2.5 token 预算与两种截断

| 参数 | 影响什么 | 本项目中的风险 |
|---|---|---|
| temperature | 采样分布的尖锐程度 | 0 = 贪心，但**不保证**跨硬件、跨 batch 组合逐位可复现（BFCL repaired 臂两次运行差了 1 个 case） |
| top-p / top-k | 候选范围 | 训练用 top-p 1、top-k −1 |
| max_tokens | 单次生成预算 | 调用还没闭合就被截断；FC 把代码放进 JSON 字符串，换行要转义，**同一份代码在 FC 下更费 token** |
| context window | 上下文容量 | 多轮报错累积后超限 |
| stop string | 何时停 | ReAct 循环停在 `Observation:`，否则模型会自己编造工具结果 |

**两种截断一定要分清**：
1. **生成截断**：模型达到输出预算，停止生成——改变了模型产出的结果。
2. **存储截断**：日志只保存前 4,000 字符——模型生成了完整内容，但我们事后看不到。

前者改变结果，后者破坏观测。修日志长度补不回第一次没生成的尾巴；反过来，存储截断造成的"看不到调用"也不代表模型没生成调用（这就是规模阶梯 32B 那档的不确定性来源）。

---

## 3. 一次工具调用的完整过程

### 3.1 用一道题走一遍

教学题：实现 `add(a, b)`。工具声明（简化自 `code_tool.py` 的 `SCHEMA`）：

```json
{
  "type": "function",
  "function": {
    "name": "run_tests",
    "description": "把你写的 Python 代码交给测试运行，返回通过情况与报错。",
    "parameters": {
      "type": "object",
      "properties": {"code": {"type": "string", "description": "完整的 Python 代码"}},
      "required": ["code"]
    }
  }
}
```

**先分清两个函数**：

```text
add(a, b)        题目要求模型"实现"的函数
run_tests(code)  系统允许模型"调用"的工具
```

如果模型输出 `{"name": "add", "arguments": {"a": 1, "b": 2}}`，它是在**调用**题目函数，而不是把 `add` 的实现提交去测试。这就是 Llama 23% 的"角色混淆"（§6.3），也是 1.5B 训练模型 52 个可恢复调用里 0 个点名 `run_tests` 的原因（§6.9）。

### 3.2 模型输出的只是 token

模型可能生成：

```text
<tool_call>{"name":"run_tests","arguments":{"code":"def add(a, b):\n    return a + b"}}</tool_call>
```

也可能生成（Qwen2.5-Coder 的习惯）：

````text
```json
{"name":"run_tests","arguments":{"code":"def add(a, b):\n    return a + b"}}
```
````

对人来说两者等价；但 hermes parser 必须先找到 `<tool_call>`，第二种就不会变成结构化调用。**模型没有亲自启动 Python**——外部程序必须先把文本识别成调用，才可能执行。

### 3.3 API 响应和模型原文不是同一层

服务端解析成功后，返回大致是：

```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [{
    "id": "call_001",
    "type": "function",
    "function": {"name": "run_tests",
                 "arguments": "{\"code\":\"def add(a, b):\\n    return a + b\"}"}
  }]
}
```

注意两点：
- `arguments` 通常是一段 **JSON 字符串**，要再解码一次——日志里一堆反斜杠就是这么来的。外层合法 ≠ 内层合法。
- 解析成功后原文被"挪走"了，`content` 变空。所以**对已解析的臂，事后无法从 content 重新分类原文**——论文表里的 N/A 就是这个意思。

没解析成功时，整段文本留在 `content`，`tool_calls` 为空——**也是 HTTP 200**。HTTP 200 只说明请求在 API 层被处理了。

### 3.4 一次完整工具交互的六个事件

```text
① 原始输出里出现调用候选           （emission）
② parser 接受                     （parse）
③ 工具名和参数通过校验             （validate）
④ 执行器真正运行                   （execute）
⑤ 产生 observation                 （observe）
⑥ observation 进入模型的下一次输入  （feedback delivered）
```

每一步都可能断。"有工具输出"和"模型收到了工具输出"也要分开：消息拼接错误、上下文截断都可能让第 ⑥ 步失败。

**检查表**：

| 你看到的证据 | 能说什么 | 还不能说什么 |
|---|---|---|
| HTTP 200 | API 请求完成 | 有工具调用 |
| 原文里出现 `run_tests` | 有相关文本 | 一定是调用（可能是复述 schema、讨论工具） |
| JSON 可解码 | 语法通过 | 工具名合法、参数合规、程序正确 |
| `tool_calls` 非空 | 服务端识别出调用 | 工具已执行 |
| 执行计数增加 | 到达被计数的执行入口 | 执行成功、反馈送达 |
| 下一轮输入含 observation | 模型拿到了反馈 | 模型利用了反馈 |
| 最终通过测试 | 满足所用测试 | 对所有输入正确、具有通用修复能力 |

### 3.5 模板、解析器、schema、约束各管什么

| 组件 | 管什么 | 不管什么 | 本项目的例子 |
|---|---|---|---|
| **chat template** | 把 messages + tools 渲染成模型输入文本；规定/示范调用格式 | 模型实际按不按这个格式输出 | Qwen2.5-Coder 继承 Qwen2.5 模板，**写着**用 `<tool_call>`，但 Coder 不按这个输出 → "模板检查假阳性" |
| **parser** | 规定哪些文本算调用、怎么抽出 name/arguments | 工具名是否在白名单（取决于实现）、代码是否正确 | hermes 只认 `<tool_call>`；hanXen 认 `<tools>`；我们的 verl parser 认三种并校验名字和非空 code |
| **schema** | 参数形状（字段、类型、必需） | 程序正确性 | `{"code":"return 123"}` 合法且符合 schema，但程序是错的 |
| **`strict: true`** | 在生成时约束输出符合 schema（约束解码） | 参数语义正确 | Llama 错调用 22→0 |

**模板是模型输入的一部分**。所以换模板可能改变首轮代码内容、调用概率、输出长度——它**不是**"只影响第一轮之后"的东西。项目早期写过"adapter 只影响第一轮之后，所以 53 vs 53 识别了失败层"——这句话不对（adapter 包含模板），论文最终把它降级为"一致性检查"。

### 3.6 `auto`、`required`、指定工具、`strict` 是四个不同的开关

| 设置 | 控制什么 | 注意 |
|---|---|---|
| `tool_choice: "auto"` | 模型自己决定调不调 | 没调用可能是合理选择，也可能是没被识别 |
| `tool_choice: "required"` | 必须产生一个调用 | 可能走不同的约束生成路径；**成功不能证明 auto 下 parser 正常**，它只是一条阳性对照 |
| 指定工具名 | 必须调这个工具 | 不再测试自主工具选择 |
| `strict: true` | 加强 schema 约束 | 支持范围依赖 parser、后端、版本 |

**一个项目记录**：08-29 诊断时，1.5B 在 `auto` 下 0/10 调用，`required` 下 6/6 成功；但把失败反馈喂回去之后 0/6 再次调用。`required` 能逼出调用，不代表模型在 `auto` 下"会用"工具。

### 3.7 FC vs ReAct

- **FC（function calling）**：走 API 的结构化 tools 通道，依赖 template + 服务端 parser。
- **ReAct**：模型在纯文本里写 `Thought / Action / Action Input`，由**宿主程序自己的正则**抽代码（见 `react_agent_loop.py` 的 `_ACTION`、`_CODE`）。

```text
Thought: I should check the implementation.
Action: run_tests
Action Input: ```python
def add(a, b): return a + b
```
```

ReAct 绕开的是**服务端 FC parser**，没有绕开"一切解析规则"——它有自己的正则，也可能被模型输出格式打败。在论文里 ReAct 是**参照臂**。

"FC 比 ReAct 差"可能来自：提示词、模板、约束、解析覆盖、token 预算、错误处理、停止规则。只有控制住这些，才能谈"协议效应"——这也是转折 2 推翻早期结论的原因。

### 3.8 两个"adapter"，别混

- **serving adapter**：parser + chat template（+ few-shot）这套服务配置。论文里"adapter repair"大多指这个。**不改权重**。
- **LoRA adapter**：训练时加入的低秩可训练参数。**就是权重增量**。

看到"adapter"先问是哪一个。

---

## 4. 项目代码里的四条路径（核心章）

这是全文最重要的一章。项目里有**四条看起来相似、实际上完全不同**的代码路径。它们都"让模型写代码、跑测试"，但 parser 在不同的地方、测试由不同的人发起、反馈以不同的角色送回。几乎所有误读都来自把它们混为一谈。

### 4.0 一张总表先记住

| 路径 | 入口 | 谁生成 | 谁解析调用 | 谁决定跑测试 | 反馈以什么角色回到模型 | 用在哪 |
|---|---|---|---|---|---|---|
| **A. 服务端 FC 探针** | `runs/final/probe_react_full.py` | vLLM 服务 | **vLLM 服务端 parser**（`--tool-call-parser`） | 模型（发出调用且被解析才跑） | tool 消息 | 50 个 KodCode 臂、规模阶梯、Llama 对照、评测时修复 |
| **B. 训练 rollout** | `launch_ppo.py` → verl | verl rollout（vLLM 后端） | **verl 自己的 ToolParser**（`multi_turn.format`） | 模型（被 verl parser 接受才跑） | tool 消息 | 历史 1.5B 训练、P3 两臂 |
| **C. 程序化反馈评测** | `code_eval_hook.py` → `eval_decompose.py` | verl native rollout RPC（当前 LoRA） | **不解析工具调用**：直接从文本抽代码 | **评测器无条件跑** | **user 消息**（"测试结果：…请修正代码。"） | P3 每个 checkpoint 的 held-out 评测、历史 1.5B 曲线 |
| **D. 原生 FC 评测** | `p3/native_fc_eval.py` | Transformers + PEFT（顺序推理） | 修好的 verl 抽取逻辑（`qwen_tools_parser.extract_calls`） | 模型（只有被接受的 `run_tests` 调用才跑） | tool 消息 | 审稿后的共同原生 FC 评测 |

再加一个**不是路径、但最容易混进来的东西**：

| | 入口 | 作用 |
|---|---|---|
| **E. 训练奖励** | `reward_code.compute_score` | 训练器给最终代码打分。它也调用 `sandbox.run_tests`，但**模型看不到这个结果**——它只进优化器。 |

### 4.1 路径 A：服务端 FC 探针（`runs/final/probe_react_full.py`）

```text
probe_react_full.py
  → HTTP 请求：messages + tools + tool_choice="auto"         （gen_fc）
  → vLLM 用 chat template 渲染、生成
  → vLLM 的 tool parser 把文本变成 tool_calls（或失败，留在 content）
  → 探针检查工具名、参数（wrong-tool 分支不执行）
  → sandbox.run_tests 执行
  → 把 observation 作为 tool 消息拼回 messages，进入下一轮
  → 轨迹落盘（带 provenance）
```

几个值得看的设计（都是翻车之后加上的）：
- **请求错误计数 `n_err`**：非零时打印"本组结果不可用"，并以**退出码 2** 结束（文件末尾 `EXIT_DIRTY = 2 if n_err else 0`）。第一版只打警告、返回码仍是 0，结果错误数据被填进了对比表（§6.2 洞 4）。
- **provenance 字段**：`model / protocol / adapter / max_tokens / clean_index / seed / temperature / fc_schema`。第一版只存了 `protocol / strength`，消融的两个臂除了文件名无法区分。
- **三种 schema**：`FC_TOOLS`（terse）、`RICH_FC_TOOLS`、`STRICT_FC_TOOLS`。
- **ReAct 侧**用 `_ACTION`、`_CODE`、`_FINAL` 正则自己解析，不经过服务端 parser。
- **`parser_adapter()`** 决定 ReAct 输出怎么判定有效。第一版对 Qwen 用 `legacy`、对别的家族用 `cross_family`，同一个行为在不同家族上被判成不同结果（§6.2 洞 5）；后来的臂统一记录为 `cross_family`（论文附录 E 的 Adapter 列；最早一批 `x2_*` 臂没有记录这个字段，显示为 `---`）。

**这条路径里，服务端 parser 是动作通道的一部分**。改 vLLM 的 `--tool-call-parser` 会直接改变执行率。

### 4.2 路径 B：训练 rollout（verl）

```text
launch_ppo.py
  → 先 import code_patch（挂评测钩子）→ 再进 verl.trainer.main_ppo
  → RayPPOTrainer 调度；TaskRunner 是独立的 Ray actor 进程
  → 根据数据行的 extra_info["agent_name"] 选 AgentLoop：tool_agent（FC）或 react_agent（ReAct）
  → rollout 后端生成 token
  → [FC] verl 自己的 ToolParser.extract_tool_calls（multi_turn.format 选哪个）
  → ToolAgentLoop._call_tool → CodeTool.execute（code_tool.py）→ sandbox.run_tests
  → 工具响应作为 tool 消息拼回上下文（response_mask = 0，不计入 loss）
  → 轨迹结束后 reward_code.compute_score 给最终代码打分
  → advantage / loss / update
```

**这条路径里最容易忽略的一层：verl 的 parser 和 vLLM 的 parser 是两个独立的实现。**
- verl 0.9.0 在 `verl/experimental/agent_loop/tool_parser.py` 有自己的 `ToolParser` 注册表；其中 `hermes` 是从 vLLM v0.9.1 移植的。
- `actor_rollout_ref.rollout.multi_turn.format=hermes` 选的是**verl 的**那个。
- 所以给 vLLM 服务端装一个 parser 插件，**对训练链路完全无效**。两边都叫 `hermes`，不代表注册表、正则、返回类型相同。

**我们怎么把自己的 parser 装进去**（这段工程值得学）：

1. `qwen_tools_parser.py` 定义 `extract_calls`（依次尝试 `<tools>` 标签、```json 代码块、裸 JSON；同一段文本只按**第一种命中的格式**解析，避免重复计数）和 `register_into_verl()`（用 `@ToolParser.register("qwen2_5_coder")` 注册一个类）。
2. 注册的类在返回前**必须**校验：工具名在传入的 tools 白名单里、`arguments` 能解码、`code` 是非空字符串。注释里写了原因：1.5B 在 mandatory prompt 下 52/52 调用的都是题目函数，不校验就会被当成有效调用（下游 KeyError，或更糟——被统计成一次成功发起）。
3. **注册必须发生在每一个 Ray actor 进程里**。rollout 跑在 Ray actor 中，只在 driver 进程里 patch **完全无效**——症状是日志里打印"已挂载"，却一条数据都不产出，训练看起来一切正常。
4. 所以用 **`sitecustomize.py`**：Python 解释器启动时会自动 import 它（只要在 `PYTHONPATH` 上），Ray worker 也是普通 Python 进程。它包住 `builtins.__import__`，**精确等到目标模块本身导入完成**再动手——在父包阶段动手会撞循环导入。
5. P3 有自己的 `p3/sitecustomize.py`，等 `verl.experimental.agent_loop.tool_parser` 导入完成后调用 `rollout_probe.install()`（里面调 `register_into_verl()` 并给每个 parser 的 `extract_tool_calls` 包一层记录事件），等 `tool_agent_loop` 导入完成后调 `install_agentloop()`。**两段要分开**：`tool_parser` 导入完成的那一刻，`tool_agent_loop` 还是 partially initialized。
6. **接力**：把 `p3/` 放在 `PYTHONPATH` 最前面会让 Python 只加载 p3 的 sitecustomize，原来根目录那个（装评测钩子的）被静默挡掉。所以 `p3/sitecustomize.py` 末尾显式按路径加载原来那个（`_chain_original()`）。注释原话："宁可多写十行，也不要又制造一个静默失效。"

**两个 verl 的坑**：
- **`BaseTool.calc_reward` 是死接口**：在 verl 0.9.0 里只出现在 docstring 中，没有任何代码调用它。把奖励写在那里，训练会**全程拿 0 分且日志无异常**。奖励必须由 `reward.custom_reward_function.path` 指定的函数给出（就是 `reward_code.compute_score`）。
- **`timing_s/agent_loop/tool_calls` 是秒数，不是次数**。verl 的 step 指标里没有工具调用计数。08-28 那次 150 步训练里，这个值恒为 0.0 是唯一的破绽（`code_tool.py` 注释："4 小时 ¥20 全程没有任何报警"）。

**计数器放在哪**（`code_tool.py`、`react_agent_loop.py`）：
- FC 路径：`CodeTool.execute` 开头往 `TOOL_CALL_COUNTER` 文件追加一行 `codetool`。
- ReAct 路径：`ReActAgentLoop._run_tests` 追加一行 `react`。
- 为什么写文件：CodeTool 跑在 Ray worker 里，进程内计数器传不出去，文件系统是同机共享的。
- **不能放在 `sandbox.run_tests`**：奖励函数打分也走它，会把"奖励评估"混进"工具调用"。沙箱自己的计数器改名为 `sandbox_exec_count`，防止再混。
- 还有一个**凭据文件** `TOOL_CALLED_FLAG`：工具一旦真被执行就 touch 它；`guard_tool_calls.sh` 定期检查，长时间不出现就报警/停止训练。

**ReAct 的 AgentLoop**（`react_agent_loop.py`，注册名 `react_agent`）：
- 解析 `Action: run_tests` + ```python 代码块```，执行后把 Observation 拼回序列。
- stop 串设为 `"Observation:"`，否则模型会自己编造工具结果。
- **mask 是关键**：环境注入的 token `response_mask += [0] * len`——进入上下文，但**不参与 loss**，否则模型会去学"预测工具输出"。
- 注意它的 docstring 写着"Qwen2.5-Coder 全系列在 auto 下从不发起 function call（0/1000）"——这是**被推翻的旧结论**（真正原因是格式不匹配，不是模型不发起）。**源码注释也会过时**，读代码时要用控制流和产物验证注释。

### 4.3 路径 C：程序化反馈评测（`code_eval_hook.py` → `eval_decompose.py`）

```text
code_patch.py 包住 RayPPOTrainer._validate
  → 先跑 verl 自带验证，再跑 CodeEvalHook.run(step)
  → 策略：NativeRolloutPolicy（P3，走 verl native rollout RPC）或 ChatPolicy（历史，走 OpenAI HTTP）
  → eval_decompose.evaluate：
       每轮把所有还没通过的题攒成一个 batch 交给 policy
       从生成文本抽代码（extract_submitted_code：先找 <tool_call> 里的 code，没有就取最后一个代码块）
       评测器无条件跑 EvalPlus script 模式测试（30 秒超时）
       通过就停；没通过且没到 max_turns 就追加
         assistant: <原文>
         user:      "测试结果：\n{obs}\n\n请修正代码。"
  → summarize → scalars（code/final_pass, code/turn1_pass, code/repair_rate …）
```

要点：
- **请求里不带 `tools`**，评测器**不解析工具调用**来决定是否执行——它总是执行。所以这条路径测的是"**在被动获得测试反馈的条件下，模型会不会修**"，不是"模型会不会主动发起调用"。
- 这就是为什么 P3 的 broken 训练臂在评测里也有 12 个 rescue（论文 §5.5）。
- **通过就停**，所以不可能出现"turn-1 过了、final 没过"的回退 → 这条路径上 `rescue = final − turn1`。
- 两个通道：
  - **multi**：542 道 EvalPlus（164 HumanEval+ + 378 MBPP+）从零写。
  - **repair**：固定的 buggy 初稿 + 真实报错（`probes_repair.jsonl`），测"修已知错误"的能力。分母固定、两个 checkpoint 拿到逐字相同的题——为了避开"自然失败集随模型强弱变化"的问题。
- MBPP+ 的 prompt 只有一句自然语言，必须附上示例断言（`question_of`），否则模型不知道函数名，pass 率会因为与能力无关的原因偏低。
- **`NativeRolloutPolicy` 为什么必要**（`chat_policy.py` 的注释）：verl 加载的 LoRA 是通过 `server.generate` 以 LoRARequest 注入的；OpenAI HTTP 端点只知道 base model 的注册表。**用普通 HTTP 请求评测，可能静默地评测了 base model 而不是当前 LoRA**。所以 P3 评测走 native RPC，并且每个请求都核验返回的 `global_steps` 必须等于期望的权重步（不等就抛错）。`CODE_EVAL_REQUIRE_NATIVE=1` 时不满足条件直接拒绝启动。

### 4.4 路径 D：原生 FC 评测（`p3/native_fc_eval.py`）

审稿人问："你的评测不测自愿发起，对 FC 训练不公平。"于是加了这条路径：

```text
evaluate_task(task_id, question, system, generate, test, max_turns=4):
  for turn in 1..4:
     raw = generate(messages)                        # HFPolicy：Transformers + PEFT 加载 LoRA，贪心
     parsed, accepted = accepted_calls(raw)          # qwen_tools_parser.extract_calls，再要求 name=='run_tests' 且 code 非空
     if not accepted:
         若 raw 里有可识别的最终代码 → 作为最终提交；结束
     else:
         只执行第一个被接受的调用；结果以 tool 消息回给模型
         rescue 定义：第一次"实际工具测试"失败，之后某次工具测试通过
  对话结束后，最终代码独立评分一次（不返回反馈，不算工具调用）
```

和路径 C 的区别：**要模型自己发起调用**；反馈是 **tool 角色**；评分是终端独立测试。和训练的区别：不是逐字节重放 AgentLoop（控制器每轮重渲染历史、script 模式测试而不是 pytest 部分得分）。

### 4.5 同一个 sandbox，两种含义

`CodeTool.execute` 调 `sandbox.run_tests` → 是**模型动作带来的环境交互**，结果给模型看。
`reward_code.compute_score` 调 `sandbox.run_tests` → 是**训练器给最终答案打分**，结果只给优化器。

即使模型一次工具都没用，奖励计算照样会跑很多次测试。所以读任何"测试次数"时都要问：**这是给模型看的反馈，还是只给优化器看的奖励？**

### 4.6 练习：画出四条路径

不看文档，在纸上画出 A/B/C/D 四条路径，每条标出：parser 在哪、谁跑测试、谁看到反馈、反馈角色。画对了，§6 和 §8 的大部分内容就是自然推论。

---

## 5. 数据、测试、奖励与指标

### 5.1 四个对象的分工

| 对象 | 功能 | 本项目例子 |
|---|---|---|
| Dataset | 提供题目、参考信息、测试 | KodCode、EvalPlus、BFCL cases、τ-bench tasks |
| Harness | 组织请求、执行、记录、评分 | `probe_react_full.py`、BFCL OpenAI handler、τ-bench runner、`eval_decompose.py` |
| Executor | 真正运行程序或环境动作 | `sandbox.py` 子进程、τ-bench retail API |
| Verifier / scorer | 判断是否达标 | pytest 通过率、EvalPlus script 全通过、BFCL AST/state checker、τ-bench 终态比对 |

用了官方数据 ≠ 用了完整官方评测协议；用了官方 scorer ≠ 我们的路由没有影响。我们在 BFCL 上走的是 OpenAI 兼容 handler（为了让请求经过服务端 parser），而 BFCL 某些本地模型 handler 会绕过这条路径——所以我们研究的是**指定服务路径**的效应。

### 5.2 每个数据集在项目里的角色

| 数据集 | 用途 | 规模 | 注意 |
|---|---|---|---|
| KodCode（去污染池） | 训练数据 | 7,669 题 | 对 EvalPlus 做 n-gram containment 去污染（题面 8/5、解答 10/6、阈值 0.10）；不排除语义重叠或预训练污染 |
| KodCode `clean[:100]` | 50 个受控探针臂共享的题集 | 100 | **不是随机抽样**，但所有臂严格配对 |
| KodCode 随机 300（seed 20260901） | 检验 `clean[:100]` 是否有偏 | 300 | `analysis/p0_random_300.json` |
| EvalPlus multi | held-out 评测 | 542（164 + 378） | 历史曲线排除 2 个缺陷题用 540；P3 保留 542 |
| EvalPlus repair | 固定 buggy 初稿修复 | 454（155 + 299） | 分母与模型强弱无关 |
| BFCL v4 | 外部基准 | `simple_python` 100 + `multi_turn_base` 100 | Gorilla commit `6ea5797` |
| τ-bench retail | 非代码、有状态的外部基准 | 115 | 用户模拟器是本地 Llama-3.1-8B |

**训练集、验证集、报告集要分开**；"测试多"也不等于"没有泄漏"。

### 5.3 单位不要混

```text
task        题目身份
trial       对同一题的一次独立尝试
rollout     一次采样得到的交互轨迹
turn        轨迹中的一轮（定义随实现而变！）
generation  一次模型生成调用
test case   一个测试断言
```

仓库里出现过 540、542、454、996（542+454）、1108 等分母。**1108 来自已作废的旧 step-0 记录**（20/1108 那个 baseline 已撤销；正式重跑改用 542 个唯一配对题，逐 checkpoint 记录 run/channel/task/step 身份、内容哈希和 native weight step，不从历史 append 文件恢复）。不要靠记忆把它们拼成一个故事，每个数都回到产生它的那次 run 的 manifest 去确认。

P3 的事件计数单位是 **generation 事件**，不是题数：broken 19,203 个、repaired 33,367 个。repaired 能继续多轮，自然产生更多 generation——所以 generation 总数是**干预之后的变量**，不能当共同分母。

### 5.4 sandbox 的两种模式（`sandbox.py`）

- `mode="pytest"`：训练用（KodCode 的 pytest 测试）。解析 summary 行，得到 passed/failed/errors/skipped/xfailed/xpassed，给**部分得分**。
- `mode="script"`：评测用（EvalPlus）。`build_evalplus` 拼装测试脚本（EvalPlus 的 `check(candidate)` 自己不调用，末尾要补调用），看进程结果得到**严格全通过**。

用 pytest 去收集一个只有 `check(candidate)` 的脚本式测试，可能"一个测试都没跑"——程序正常退出 ≠ 真正检查过。

`Result` 的关键定义（真实源码）：

```python
@property
def total(self) -> int:
    # A skipped/xfail test is not evidence that the candidate passed it.
    return (self.passed + self.failed + self.errors + self.skipped
            + self.xfailed + self.xpassed)

@property
def pass_ratio(self) -> float:
    return self.passed / self.total if self.total else 0.0

@property
def all_passed(self) -> bool:
    return self.status == "ok" and self.total > 0 and self.passed == self.total
```

一个测试都没跑起来时 `status = "no_tests"`（语法错误导致连 collection 都没跑也落这里）。

### 5.5 训练奖励（`reward_code.py`）

$$
R = \frac{\text{passed}}{\text{passed} + \text{failed} + \text{errors} + \text{skipped} + \text{xfailed} + \text{xpassed}}
$$

设计决定和理由（都写在源码注释里）：
- **outcome-only**：只看最后一次提交的代码，不给过程奖励。"本项目要测的正是纯 outcome reward 下增益落在哪一轮，给过程奖励等于先把答案掺进假设里。"
- **部分得分而不是 0/1**：1.5B 在难题上整组全 0，GRPO 的 advantage 归零、没有梯度。
- **取最后一次提交**（`extract_submitted_code`）：先找 `<tool_call>` 里的 `code`，没有就退回最后一个 markdown 代码块；轨迹是"写→测→改"，第一个是初稿。
- **三条堵死的 hacking 路径**：没代码 → 0；没测试字符串 → 0（"返回正分会制造出免费满分的题，GRPO 会迅速学会往那些题上靠"）；沙箱异常 → 0。
- **历史 verifier 的两个缝隙**（P3 之前已修）：summary 正则未锚定、取第一个 `(\d+) passed` 匹配（模型 `print("5 passed")` 可以伪造）；skipped/xfail 不进分母。

**教学例子**：10 个测试过 7 个，训练奖励 0.7，严格评测记 0。平均奖励上涨，可能只是更多题从 2/10 变成 5/10，全通过的题数未必涨。

**"给错题负奖励"不是准确描述**：原始奖励在 0–1 之间；组内比较之后 advantage 才可能为负。

**sandbox 不是安全沙箱**：资源限制、超时、临时目录主要控制故障影响，不构成对任意不可信程序的完整隔离。不要把生产机密放进实验执行环境。`code_tool_core.suspicious()` 只**记录**可疑模式（`sys.exit`、`pytest.skip`、改 builtins、动态 import、碰测试文件、`signal.alarm`），不拦截——拦了就测不到倾向了。

### 5.6 任务级指标

| 指标 | 定义 | 注意 |
|---|---|---|
| `turn1_pass` | 第一份程序通过 | — |
| `final_pass` | 最终程序通过 | — |
| `rescued` | 第一轮没过、最终过了 | — |
| regression | 第一轮过了、最终没过 | 路径 C 通过即停，不会发生 |

一般情况下：

$$
\text{final} - \text{turn1} = \text{rescues} - \text{regressions}
$$

只有"通过即停、无回退"时，差值才等于 rescue。换一个会继续操作的环境就不能直接沿用。

**两种 rescue 分母**：

```text
rescues / 全部任务数        多轮对总体通过率的贡献
rescues / 首轮失败任务数    在失败机会中救回的比例
```

后者跨模型比较时，失败集合会变——更强的模型只剩更难的题，条件修复率低不代表修复能力差。这正是要建 repair 通道（固定 buggy 初稿）的原因。

**native FC 的 rescue 定义又不一样**：要求"第一次**实际工具测试**失败、之后某次工具测试通过"，不能和路径 C 的 rescue 合并。

### 5.7 `pass@1` 不一定只生成过一次

代码生成文献里的 pass@k 是多个独立候选中至少一个通过。本项目部分历史字段把"一次完整多轮交互的最终成功"也叫 pass@1。必须说明是"无反馈单份初稿"还是"最多四轮、有测试反馈的一次 agent 尝试"——后者不能和公开的无工具单轮分数直接比。

### 5.8 四种调用率

| 指标 | 分子 | 分母 |
|---|---|---|
| first parsed | 第一次请求被解析的题数 | 题数 |
| any parsed | 整条轨迹至少解析一次的题数 | 题数 |
| parsed calls | 调用对象个数 | 要说明：请求、generation 还是题 |
| executed tasks | 至少真正执行一次工具的题数 | 题数 |

例：τ-bench repaired 臂 636 次调用发生在 115 个任务内，**不是"636% 的任务调用了工具"**；103 个任务至少执行过一次，那才是任务级覆盖。

### 5.9 漏斗的记录粒度

要能把漏斗逐条连起来，每条记录至少要有：

```text
run_id / arm / checkpoint / task_id / trial_id / rollout_id / turn_id / call_id
```

历史产物并不都有这些字段。P3 的经验：派发、执行、观测可以按 request ID 唯一关联；但**抽取记录缺每轮请求标识**，所以"accepted → dispatch"只能在**轨迹计数粒度**匹配。缺关联时就只报告能成立的聚合计数，不要声称逐条闭合了整条链。

---

## 6. 实验逐个读：问题 → 设计 → 结果 → 边界

按**科学问题**排列，不按开机顺序。实验编号有历史混用（`p3/` 既放早期 rollout 探针也放正式训练；`p6/PLAN_P3.md` 是后来的训练方案），不要只凭 P 编号判断实验性质。每个实验的完整数字表见 [PAPER_STUDY_GUIDE.md §5](PAPER_STUDY_GUIDE.md)，这里讲**为什么做、怎么读**。

### 6.1 E0 起点：历史 1.5B 训练（08-26 → 08-28）

**问题**：结果奖励 RL 能不能让代码模型更会利用测试反馈？

**设计**：Qwen2.5-Coder-1.5B-Instruct，verl 0.9.0，GRPO×2 seed + RLOO×1 seed，16×8 = 128 条轨迹/步，150 步，KodCode 7,669 题，单卡 A800。评测走路径 C。

**结果**：

| run | multi final | turn-1 | repair |
|---|---|---|---|
| GRPO seed42 | 64.8% → 67.6%（*p*=0.064） | +2.6%（*p*=0.093） | +4.6%（*p*=0.030） |
| GRPO seed1 | 64.3% → 67.0%（*p*=0.078） | +3.0%（*p*=0.053） | +3.7%（*p*=0.097） |
| RLOO seed42 | 64.6% → 67.2%（*p*=0.077） | +2.6%（*p*=0.082） | +0.9%（*p*=0.786） |

- 新通过的题里 **91–94% 是第一轮就过的**；多轮 rescue 在 540 题里一直是 5–10 个，横盘。
- repair 的 +4.6%（*p*=0.030）在三个 run 里是 +4.6 / +3.7 / +0.9——**跨 run 摆动超过效应本身**，降级为不可报告。单个 run 的 *p*<0.05 撑不起一个发现。

**读法**：RL 提升了初稿质量，多轮调试能力没变。自然的解释是"1.5B 太小学不会"——**这个解释是错的**，要证明它错，用了后面所有实验。

**边界**：三个 *p* 都不显著，方向一致也不能自动合成显著；历史训练没保存完整 rollout 原文，事后无法补回它当时到底输出了什么。所以论文只把它当**动机**，不当证据。

### 6.2 E1 第一版跨家族探针，以及一次彻底翻车（08-29 → 08-30）

**问题**：如果 1.5B 学不会，大一点的模型会不会？FC 和 ReAct 有没有区别？

**设计**：不训练只测。现成模型在同一批 `clean[:100]` 上分别用 ReAct 和 FC 跑，看发起率、解析率、通过率。

**第一版结果**：Llama-3.1-8B ReAct 81% vs FC 50%，McNemar 38/7，*p*=3.1×10⁻⁶。当时的结论是"FC 协议压制模型发起动作"。

**审计找出的 8 个洞**（`fixes/AUDIT.md`）——这一节是全项目最值得反复读的工程教训：

| # | 洞 | 后果 | 教训 |
|---|---|---|---|
| 1 | **`except Exception: code = ""`** 把"JSON 解析失败 / key 名不对 / 模型真给空"压成同一个空串，且没存原始 arguments | 26 条空代码撑起全部显著性；剔掉后 74 题上 81% vs 68%，17/7，*p*=0.064 | 不要用一个 except 抹平不同的失败原因；出错时至少记原始输入 |
| 2 | `gen()` 和 `gen_fc()` 都写 `max_tokens=1024` | FC 把代码塞进 JSON 字符串更费 token，更容易截断成残缺 JSON；而 `finish_reason` 被丢了，无法证伪 | 公平 ≠ 相同参数值 |
| 3 | 比率分子按轮累加、分母按题 | 多轮模型被系统性抬高（Mistral react 脚本报 67%，逐题实际 54%）；DS-6.7B 恰好单轮，bug 差点永远藏着 | 分子分母必须同单位 |
| 4 | 日志写着"请求错误: 3 ⚠️ 本组结果不可用"，**返回码仍是 0** | 不可用数据被填进对比表 | 警告没有强制力；要非零退出码或异常 |
| 5 | `parser_adapter()` 对 Qwen 用 `legacy`、对别的家族用 `cross_family` | 同一行为在不同家族判成不同结果 | 判据跨臂必须一致 |
| 6 | 轨迹只存 `protocol / strength` | 消融臂除了文件名无法区分 | 每条数据带 provenance 落盘，记**实际生效**的值 |
| 7 | `idxs = clean[:a.n]` 取前 100 | 配对成立，但不代表题库 | 有些坑要认下来写进 limitation（后来用随机 300 题复制） |
| 8 | 关机守护"满 100 行就关机"，同步循环 45 秒一轮 | DS-1.3B 本地只有 93/100 行 | 自动关机要先 sync 再留缓冲 |

8 个洞里**只有洞 4 是脚本主动报出来的**，其余 7 个都是人工翻代码找到的。

**更致命的是后来发现的**：洞 2 的"修复"根本没生效，而且方向判反了。修复脚本只替换了 `gen_fc()` 里的字面量 `1024`，ReAct 侧的 `gen()` 用的是**函数默认参数** `max_tokens=1024`，没被匹配到；而 provenance 统一写的是全局 `MAX_TOKENS=2048`。结果：**ReAct 实跑 1024、FC 实跑 2048，两者都记成 2048**——拿到更长预算的其实是 FC。发现后用真正的 2048 重跑三个主 ReAct 臂：80→80、74→74、32→33，实际影响很小。7 个受影响的臂全部列进 `runs/final/ERRATA.md`（论文附录 A.1）。

**另一个推翻原结论的发现**：早先的跨家族 FC 实验启动 vLLM 时漏了 `--enable-auto-tool-choice` 和家族对应的 `--tool-call-parser`。补齐后（Llama→`llama3_json`、Mistral→`mistral`、Qwen→`hermes`），Llama 在 `auto` 下发起率 97%——根本没被压制；Mistral 只有 2%。**效应是家族依赖的，不是协议普适的**。

**读法**：这一节本身就是论文论点的例证——**我们自己的测量也被接口和管线细节扭曲过**。

### 6.3 E2 重跑：四个家族、四个失败层 + Llama 的四个对照

**四个家族在四个不同的层失败**：

| 家族 | 层 | 症状 | 能事先查出来吗 |
|---|---|---|---|
| DeepSeek-Coder | template | chat template 从不注入 tools | ✓ 看模板就知道 |
| Qwen2.5-Coder | parser（envelope） | 输出 ```json / 裸 JSON，不是 `<tool_call>` | × 模板检查给出**假阳性**（模板里写着支持） |
| Llama-3.1-8B | schema / 角色 | 23% 把题目函数当工具调 | × 要检查参数才能发现 |
| Mistral-7B-v0.3 | token | 重复 `[TOOL_CALLS]` → HTTP 400 | ~ 有报错，但只是有时 |

只有 Mistral 的失败会"自报家门"，其余都是静默的。

**一个假阳性检查的教训**：模板里有 tools ≠ 模型会用 tools。用"检查模板是否支持 tools"来做 preflight，在 Qwen2.5-Coder 上会通过，然后在线上静默清零。

**Llama 的四个单变量对照**——每个对照排除一种替代解释：

| 替代解释 | 对照 | 错调用 /100 | 结论 |
|---|---|---|---|
| schema 缺参数描述 | rich vs terse | 23 → 22（6/5，*p*=1） | 排除 |
| ReAct 只是多了推理步 | FC + Thought 脚手架 | 23 → **59**（更糟） | 排除 |
| 服务器没按文档配 | + 官方模板 | 23 → 22（1/0，*p*=1） | 排除 |
| 没开 schema 约束 | 官方模板 + `strict: true` | 22 → **0**（22/0，*p*=4.77×10⁻⁷） | **接受** |

strict 之后：有效执行 73→97，turn-1 33→46，final 44→61（+17，过了 Bonferroni）。

**读两个基线要小心**：`44→61` 是"官方模板 FC → +strict"的单变量差；`49→61` 是"terse FC → 官方+strict"的 +12。不能把不同起点混成"一个开关同时提高了 12 和 17 分"。

**教训**：排除了 N 个替代解释 ≠ 找到了唯一真原因。单变量对照只能否定，strict 有效也不能证明"模型对角色的理解被治好了"——它只是让错误动作无法生成。

### 6.4 E3 Qwen2.5-Coder 规模阶梯 + 离线重解析 + 失败层分解

**问题**：服务端恒报 0 调用时，我们漏掉了什么？

**设计**：同一批 100 题、同一 optional 提示、同一 Hermes 配置，五个 Coder checkpoint（1.5B/3B/7B/14B/32B）。另用 tight 分类器（`analysis/intent.py`）判断原文里有没有调用候选。

| 尺寸 | server parsed | tight 候选 | final pass |
|---|---|---|---|
| 1.5B | 0 | 0 | 31 |
| 3B | 0 | 4 | 37 |
| 7B | 0 | 21 | 53 |
| 14B | 0 | 36 | 53 |
| 32B | 0 | 80 | 68 |

**离线重解析**（`analysis/reparse_matrix.py`）——回答"是不是你的抽取器只是比 hermes 好"：对**同一批存储字节**用四种规则重解析，不重新调用模型。hermes 重放每个尺寸都是 0（模型从不产生那个格式）；bare JSON 规则 0/5/22/42/100；tight 0/4/21/36/80。

**失败层分解**（`analysis/failure_layer.py`，逐行照抄 vLLM 0.27.1 hermes parser 的正则）：把每个未解析输出归到第一个失败点——no envelope / payload malformed / strictness only / **parser loss**。全档案去重后 **4,254** 个首轮，parser loss = **0**。

- Coder：100% no envelope（输出合法 JSON，就是没包装层）。
- Instruct：标签齐全，但 payload 坏（把 Python 塞进 JSON 字符串时 docstring 的 `"""`、裸换行没转义）。

**这一步改变了论文的措辞**：从"parser 静默丢弃合法调用"改为"**尝试从未以 parser 定义接受的形式到达 parser**"。同一个后果，不同的机制——审稿人一定会追问是哪一个。

**这一步也有一个"让自己有利的错误"**：第一版全档案首轮数没去重，数字更大更唬人；去重后才是 4,254。这类错误不会有人来纠正你。

**随机 300 题复制**（P0）：tight 计数 0、13、90、119、242（/300），即 0、4、30、40、81 每百；server 仍全 0。7B 从 21/100 变成 30/100——单个 100 题子集对中间档不稳定，但方向和单调性不变。注意 1500 个"题×尺寸"观测来自 300 个共享题，**不是 1500 个独立任务**。

**边界**（详见 PAPER_STUDY_GUIDE §3.6）：
- tight 是启发式正则，不做 `json.loads`；人工金标准下 tight 层精度 36/40 = 0.900，80×0.9 ≈ 72 只是**敏感性估计**。
- 4,000 字符存储截断造成假阴性，精度 < 1 造成假阳性 → **净偏差无法识别**，80 不是下界。
- no envelope 是**首失败**分类：Coder 在这一层就返回了，payload 层根本没被检验。"parser loss = 0"只对带 envelope 的输出才是真正的检验。
- 一个家族、一个错配接口、一个任务内的观察，不是 scaling law。

### 6.5 E4 Qwen2.5-Instruct 同谱系对照（P2）

**问题**：是不是整个 Qwen2.5 谱系在 Hermes 下都不行？

**设计**：换成 Instruct 的五个尺寸，其余全同。

| 尺寸 | parsed | turn-1 | final | rescue |
|---|---|---|---|---|
| 1.5B | 1 | 13 | 13 | 0 |
| 3B | 11 | 14 | 17 | 3 |
| 7B | 63 | 34 | 40 | 6 |
| 14B | 77 | 38 | 52 | 14 |
| 32B | 88 | 42 | 65 | 23 |

**一次环境事故**：初次运行时主机**没装 pytest**，执行层全部无效，但请求错误为零、脚本正常退出。补装后在 `runs/p2_pytest/` 完整重跑，解析层逐值复现。教训：`rc=0` 和"无请求错误"都不代表执行环境正常。

**读法**：同谱系的 Instruct 不是 Coder 那种"全 0"，所以不能归因于共享的家族/模板名本身；更可能是 checkpoint 特定的后训练差异。

**边界**：Coder 和 Instruct 的训练混合、专门化、对齐都可能不同，这不是"只改变了工具格式训练"的单变量实验；parsed 和 rescue 一起随尺寸上升，也不能单独证明前者造成了后者。

### 6.6 E5 Qwen3 预注册阶梯 + Granite rescue（P4）

**问题**：如果信封匹配，同样跨度的规模上还会有大量静默调用吗？

**设计**：Qwen3（其要求的信封与 Hermes 匹配）0.6B/1.7B/4B/8B。预测"每个尺寸 parsed > 0"在推理前提交（`p4/PREREGISTRATION.md`）。

**结果**：parsed 86/44/70/97（预测成立，但不单调）；静默数 1/1/2/0。

**两个协议偏离**（都记录了）：默认 thinking 模式在调用前就耗尽 token 预算，所以关了 thinking（`enable_thinking=false`）；parsed 不单调。

**Granite-3.1-8B rescue 臂**：在 Hermes 和 Granite 原生 parser 下**都没有调用**。所以它是**无信息的**（uninformative），不能凑成"又一个复制成功的家族"。**教训：阴性臂需要救援臂才有意义**——一个 0 可能是"被删失了"，也可能是"它根本没试过"。没有这条救援臂，Granite 的 0 会被当成又一个案例写进论文。

**边界**：家族和信封一起变了，不能隔离信封的因果作用；"有 git 提交"不自动等于完整的正式预注册。

### 6.7 E6 评测时修复闭环（Qwen2.5-Coder-7B）

只换服务适配器（hanXen parser + 改过 few-shot 的模板），**不训练**：

```text
first parsed      0 → 84
≥2 实际测试步骤   0 → 37
rescued           0 → 9
turn-1 pass      53 → 53
final pass       53 → 62     （16/7，p=0.093；随机 300 题 160→178，p=0.0385，都没过 Bonferroni）
ReAct 参照        95 / 40 / 57 / 74 / 17
```

**读法**：动作和多轮反馈通道明显恢复，最终成功数数值上涨，但不足以给出显著性结论。turn-1 两臂都是 53——总数相同不代表通过的是同一批题，也不能证明模板没影响首轮（adapter 改了模板），所以论文把它降为"一致性检查"。

**adapter 不是万能开关**：同一个 adapter 在 7B 恢复 84/100，在 **3B 只有 1/100**——3B 完全无视 `<tools>` few-shot（0 个标签，99 个直接写代码）。

**第三方组件的处理**：hanXen 模板的 few-shot 示例调用 `get_weather`（我们 schema 里没有的工具），不改会诱导调用不存在的工具，同时虚增发起数和空参数统计。改成 `run_tests` 示例；原版、改版、parser 的哈希都记录（`runs/final/plugin/`）。

### 6.8 E7 BFCL v4：在别人的尺子上重测（P1，09-01 → 09-03）

**为什么要做**：如果问题只在自研小任务上出现，可能是我们的 harness 写得怪。用 BFCL 自己的数据、执行器、官方 scorer，检验它是否超出自研任务。

**设计**：`simple_python` + `multi_turn_base` 各 100，Gorilla commit `6ea5797`；不改 `bfcl_eval` 自身文件（在 import 时内存注册 OpenAI handler）；中间加 recording proxy 记录每一个 HTTP 请求；两个 case 的端到端 smoke 先过；result 数、score 文件 `total_count`、proxy 里不同 case id 数**必须都等于 200** 且 0 个 HTTP 错误，否则拒收。

**结果**：

| | documented | dedicated |
|---|---|---|
| 200 题首请求解析 | 0 | 196 |
| `simple_python` 官方 accuracy | 0.00 | 0.96（96/100） |
| `multi_turn_base` 官方 accuracy | 0.00 | 0.19（19/100） |
| multi-turn 执行过工具 | 0 | 98 |

**2×2**（`runs/p1_2x2/`）：只换 template 0、只换 parser 0、都换 196。**在基线处两个简单效应为 0，主效应各 +98（平均后），交互 196**——不要说"主效应为零"（项目早期在 arXiv v1 里写错过，见 PAPER_STUDY_GUIDE §2.10(g)）。

**固定输出重放**（审稿后补）：冻结 documented 臂的 100 条 `simple_python` 原始响应，**不重新生成**，用一个新的 answer-blind 离线 JSON 抽取器重解析：97 个恢复出结构合规的调用，**92 个通过原官方 AST checker**；repaired 臂阳性对照复现 96。这是 masking 最直接的证据——正确调用**就在原始字节里**。三个限定：这是新离线规则不是部署 parser；92 vs 96 的差不能全归给模板；只适用于单轮（多轮改动作会改变后续环境反馈）。

**两次运行的稳定性**：hermes 臂两次逐列相同；repaired 臂差 1 个 case（197 vs 196 first parsed），success 都是 19。原因：greedy 解码在不同 batch 组合下 vLLM 不保证逐位确定，这个臂 8 线程跑多轮执行器 → 中间列报 ±1。

**边界**：0.96 是**一个类别**的分数，不是 BFCL 排行榜总分；我们走的是 OpenAI 兼容 handler 这条服务路径。

### 6.9 E8 τ-bench retail：非代码、有状态、有模拟用户（P5）

**设计**：retail 全部 115 个任务，documented vs repaired 适配器。用户模拟器是本地 Llama-3.1-8B（不是原 harness 的 gpt-4o）。

**结果**：解析并执行的调用 0 → 636；进入工具循环的任务 0 → 103；solved 7 → 10（7 个两臂都成功、3 个只有 repaired 成功、0 个只有 documented 成功，3/0，*p*=0.25）。documented 臂 789 个回合有 call-shaped payload 但没解析，全部归到 missing-envelope。

原始结果 `p5/results_full/`，分析入口 `p5/analyze_tau.py`。

**读法**：envelope 层机制扩展到了非代码有状态环境。**不能再说"我们完全没有非代码实验"**；但它也不是非代码的规模阶梯，不是 RL 结果。

**边界**：
- 绝对分不能和排行榜比（用户模拟器不同）；两臂对话会分叉，模拟器随机性也是归因边界。
- 789 个全部在 envelope 层失败，**没有一个到达 payload 验证阶段**——不能说"0 个 payload 是坏的"（我们早期犯过这个错，被 Codex 抓出来了）。

### 6.10 E9 历史训练诊断：为什么 1.5B 的 0 不能当因果证据

**训练侧计数**（同模型、数据、超参、seed，prompt 强度对齐为 mandatory）：

| 臂 | 步数 | `num_turns/mean` | 工具耗时 | 每 rollout 调用 |
|---|---|---|---|---|
| FC（tool_agent + hermes） | 10 | **2.000** 恒定 | **0.00 s** | **0.000** |
| ReAct（react_agent） | 3 | 5.883 | 12.21 s | **2.052** |

三条独立证据一致：`num_turns` 钉在最小值 2（1 个 user + 1 个 assistant，没有工具轮）、工具耗时正好 0、专用计数器 FC 0 行 / ReAct 788 行（标签全是 `react`）。注意步数不匹配（10 vs 3）。

**但"零执行"不等于"零尝试"**：FC 训练臂没保存原文，三种情况分不开——直接给代码、畸形调用、被 parser 丢掉的合法调用。用相同 mandatory prompt 在 1.5B 上复刻探针（n=100）：

| 类别 | n |
|---|---|
| 错名字，arguments 带完整代码 | **0** |
| 错名字，arguments 是题目函数的参数 | **36** |
| 错名字，可执行代码在别处 | 16 |
| **点名 `run_tests`** | **0** |

我们写的 verl 侧 parser 能把可恢复调用从 0 提到 52/100，但**没有一条是那个工具**——名字归一化救不了，因为没有调用携带代码可归一。加 role-disambiguation few-shot（把错误调用和正确调用并排给模型看）之后更糟：可恢复调用 52→64，final 15→13，点名 `run_tests` 仍是 0。

另一个"加压适得其反"的例子：1.5B 把 optional 工具指令换成 mandatory，parser 接受数仍是 0，final pass 31→15，无法解析的输出 ~0→46。**当 agent 看起来不愿用工具时，加强指令是最便宜、也最可能误导的尝试**——它能让任务表现变差而调用数保持 0，看起来像一个"既不愿意又没能力"的模型。

**读法**：在 1.5B 上，"本来就没有格式完好的调用可供删失"是最可能的解读。所以历史 1.5B 的零是**过度决定的（over-determined）**，不能证明 RL 部分的主张。这直接决定了 P3 必须换 7B。

**措辞降级**：早期写"It renders a policy branch unreachable by gradient"——数学上过度声称（策略对其他 token 的更新会改变整体分布；观测到零样本 ≠ 概率支撑为零）。改成"在观测到的 rollout 分布下没有直接的 on-policy 梯度信号"。

### 6.11 E10 7B rollout 探针：把机制测进训练循环内部（09-01 → 09-03）

**设计**：把探针（`p3/rollout_probe.py`）装进 verl 自己的 AgentLoop，7B、文档推荐配置（hermes），跑一个 rollout batch。

| | 计数 |
|---|---|
| 裸 JSON `run_tests` 调用，合法且完整 | 45 |
| 点名 `run_tests` 但 payload 坏 | 12 |
| 没尝试调用 | 58 |
| **被 verl 的 hermes parser 接受** | **0** |
| 工具执行 / 观测 | 0 / 0 |

**115 次生成里 45 次带着"只要包上包装层就能执行"的调用，没有一次到达环境。**

**工程细节**：探针脚本是**自证式**的——任何一个钩子触发次数为 0 就写 `P3_INVALID`。这条救过一次：早期版本钩子没装上，但脚本"成功"退出了。

**诚实记一笔**：这个 run 在 rollout 之后的 actor 更新阶段 **CUDA OOM**，没完成任何梯度步。rollout 观测产生于 OOM 之前不受影响，但这句话必须写进论文。

**纠正一个下错了很久的诊断**：复盘里原本写着"P3 卡在 verl 不暴露 guided decoding"。这和论文自己的论点**矛盾**——论文主张的修法是"让 parser 接受模型已经在发的东西"，不需要约束解码。真正挡路的是**规模**：1.5B 没东西可接受，会发合规调用的是 7B，而 7B 全参单卡 OOM。所以 P3 是一个"**7B + LoRA**"的实验，不是"换 rollout backend"的工程项目。**自相矛盾就是诊断出错的信号**。

### 6.12 E11 P3：正式的 7B 单变量 RL 对照（09-15 → 09-19）

设计细节见 §8。结果：

| | broken | repaired |
|---|---|---|
| generation 事件 | 19,203 | 33,367 |
| tight 事件 | 8,689 | 17,608 |
| parser 接受的调用 | 10 | 16,912 |
| 接受了调用的 generation | 10 | 16,844 |
| 执行 = 观测 | 10 | 16,844 |
| held-out final（step 0 → 150） | 430 → 430 | 430 → 427 |
| rescue（step 0 → 150） | 12 → 12 | 12 → 12 |

- repaired vs broken @150：0/3，*p*=0.25，−0.55pp，CI [−1.79, 0.72]。repaired 自己 step 0→150：1/4，*p*=0.375。
- 多出的 68 个 accepted 调用来自 40 个多调用 generation；`max_parallel_calls=1` 只派发第一个，其余不派发、不合并——**不是漏执行事故**。
- broken 是**几乎关闭**（nearly closed），不是字面关闭：8,689 个 tight 事件只对应 10 次执行。8,689 和 10 是分别统计的量，没有逐事件关联时不要写成"这 8,689 个有效调用里执行了 10 个"。
- tight 和 accepted 是两种检验：16,323 个事件两者都满足，1,285 个 tight 但没被接受，521 个被接受但不 tight。
- **策略确实更新了**：150 步全部有非零梯度范数；LoRA 张量审计显示 56% 的值改变了。所以零结果不是"根本没训练"。

**三个验收问题的答案**：
1. 通道恢复了吗？**是**（10 → 16,844）。
2. 固定评测上的行为改善了吗？**没测到**（turn-1 418 → 415）。
3. 多轮结果提高了吗？**没测到**（final 430 → 427，rescue 12 → 12）。

**这不是"实验失败"**，也不是"证明了接口对学习没用"。直接测到的是训练期间工具经验大幅增加，而这些经验在共同程序化反馈协议下没有表现为可检测的收益。可能原因（数据分不开）：训练是 tool-role 反馈、评测是 user-message 反馈的格式偏移；LoRA 容量与 150 步预算；多轮信用分配。

### 6.13 E12 共同原生 FC 评测（09-23 → 09-26）

**为什么加**：路径 C 不测自愿发起。审稿人的问题合理。

**设计**：base、broken step 150、repaired step 150 三个权重，同样 542 题，路径 D。审稿后设计，在训练预注册之外（post-hoc）。

**结果**：final 239 / 248 / 248；repaired vs broken 16/16，*p*=1，CI [−3.47, 3.47]；发起任务 46/50/47；tool-test rescue 2/1/1。

**准备阶段的门控**（值得学的工程做法）：
- 先在 CPU 上：确认两个 step-150 checkpoint 还在、LoRA 张量摘要和先前审计一致（392 张量、80,740,352 参数）；导出约 155 MiB 的 LoRA adapter 并逐张量回读；用真实 tokenizer + 强制工具 fixture 验证 tool-role 回路；小 CPU 模型做 PEFT 往返检查。
- 再 GPU 冒烟（6 题、24 题 pilot），验证加载的 LoRA 内存张量、缩放、身份，再测吞吐，然后全量。
- runner 检查 manifest、parser、配置、权重身份；**已有输出目录拒绝覆盖**。
- **"导出正确、GPU 实际加载正确、完整交互正确、预算足够"是四道独立门控**，过了第一道不能直接跑全量。
- 一个资源陷阱：CPU 主机上 `free` 显示的可能是宿主机资源；那次容器实际只有 2 GiB 和 0.5 CPU，要读 cgroup 限额。

**边界**：不是等价性结论，不是第二个训练种子；程序化反馈下 430/427 vs 原生 FC 下 248/248 的跨协议差**不能**归因为"自动反馈的收益"（后端、历史渲染、提交处理、停止规则全变了），只解读协议**内**的权重对比。

### 6.14 E13 人工验证：用被截断的仪器测量截断

简述（完整流程见 PAPER_STUDY_GUIDE §4.6）：
- 98 个按分类器判决分层的样本（40 tight / 28 strong-only / 30 neither）。
- **第一轮 κ = 0.713**：12 个假阳性里 11 个是阅读顺序（调用在一个 ```python 代码块后面，标注者读完代码块就停了），1 个是**仪器失败**——标注包把每个输出截在 2,400 字符，而那个 32B 输出的调用从第 2,585 字符开始，证据根本不在包里。一个研究"截断如何隐藏调用"的项目，自己的标注工具也截断了。
- **第一轮裁决是单边的**：被复查的 13 题恰好就是标注者和分类器不一致的那些，10 个推翻全朝分类器方向。这种选择规则下一致性只能上升，0.713 → 0.936 不是可靠性证据。
- 修正：第二个盲标注者 A2、完整输出、**事先固定**"任何三方不一致都复查"的规则。同字节 62 题上 A1–A2 κ = 0.967；第三方裁决者 15 次里 6 次偏离分类器；分类器 vs 金标准 κ = 0.871；tight 精度 0.950 → 0.975 → **0.900**（报整个轨迹，用最保守的终点）。

**教训**：裁决/复核的选择规则必须先定，且不能引用你要验证的那个对象。

### 6.15 E14 方差与"总数不变、题目在变"

- Llama，temperature 0.6，3 个 seed：ReAct final 72/72/72，但解集两两 Jaccard 0.71–0.80（15–24 题不同）——**总数巧合，不是同一批题**。
- ReAct turn-1 波动大（43–56）而 final 收敛；FC turn-1 几乎不变（44–45）而 final 分散——机制是 rescue 的补偿（19/16/29）。三个 seed 里"turn 1 过了但 final 失败"都是 0。
- FC seed 3 有一个请求错误（模型发了并行调用，vLLM 返回"只支持单个工具调用"）：端到端记 57/100；clean-arm 分析排除这个 seed。
- 训练 step 15→30 之间，542 个首轮完成里 85 个逐字改变、6 个题翻转（3 得 3 失），turn-1 和 final 数值完全相同（340/346）。"**策略在动，总数不动**。"

**读法**：报告题目级变化，不只报总数。

---

## 7. 从梯度下降理解 agentic RL

### 7.1 SFT 和 RL 优化的信号不同

**【原理】** SFT：给定输入和目标输出，对目标 token 做交叉熵。RL：让当前策略自己生成，再用结果的奖励调整"生成这些输出"的倾向。

pytest 不可微没关系——RL 不需要把梯度穿过环境，它用"生成动作的 log 概率 × 回报"构造梯度估计。

把轨迹记为 τ，目标是：

$$
J(\theta) = \mathbb{E}_{\tau \sim p_{\theta, I}}[R(\tau)]
$$

**这里特意把接口 I 写进轨迹分布**。同一个模型输出，在不同接口下可以进入不同的环境分支、收到不同的反馈，从而改变之后的输入和采样结果。这一行公式就是"接口影响 RL 经验"的数学表述。

### 7.2 策略梯度的直觉

省略基线、KL、裁剪、token mask 的教学形式：

$$
\widehat{\nabla J} = \frac{1}{N} \sum_i A_i \sum_{t \in \text{模型生成的 token}} \nabla_\theta \log \pi_\theta(y_{i,t} \mid h_{i,t})
$$

- A_i > 0：提高这类已采样输出的概率；A_i < 0：降低。
- 求和只在**模型生成的 token** 上——环境注入的 observation 虽然进了上下文，但 mask = 0，不被优化（`react_agent_loop.py` 的 `response_mask += [0] * len(obs_ids)`）。mask 错了，模型会去学"预测工具输出"。
- 整条轨迹共享一个奖励，不等于定位了哪一轮有功——这就是**信用分配（credit assignment）**问题：功劳该算给选工具、初稿质量、读报错，还是一次幸运采样？

**核心推论**：如果"调用工具 → 看到报错 → 修好代码"这类轨迹**从未出现在采样里**，梯度就无从强化它。parser 吃掉调用，等于把这类轨迹从经验分布里删掉了。

### 7.3 几个角色别混

| 角色 | 作用 |
|---|---|
| actor / policy | 正在优化的生成模型 |
| rollout policy | 产生当前训练样本的策略快照（推理副本） |
| reference policy | 约束偏离程度的参考模型（是否启用看配置） |
| critic / value model | 估计价值的模型——GRPO 不需要单独训练 |
| reward function | 给样本打分；本项目就是 pytest，不是神经网络 |

日志里叫 `critic/rewards/mean` 不代表真的训练了一个 critic，要看算法配置。

### 7.4 GRPO：同一道题的多个输出相互比较

**【原理】**（DeepSeekMath）对同一 prompt 采样一组 G 个输出（我们 G = `rollout.n` = 8），用组内奖励比较估计 advantage：

$$
A_i = \frac{r_i - \bar{r}}{s_r + \epsilon}
$$

教学例：四个样本奖励 [0, 0.5, 0.5, 1]，均值 0.5，减均值后 [−0.5, 0, 0, 0.5]，再除以标准差（`learning_lab.py` 算出 [−1.41, 0, 0, 1.41]）。

**为什么用部分得分**：如果一组全 0，组内没有相对优劣，advantage 全为 0，这道题不贡献奖励信号。若各样本通过 2、4、5、7 个测试，就有了可区分的信号。

三个限定：
- 部分得分不保证产生有用的工具使用策略，可能主要改善初稿（历史 1.5B 正是如此）。
- 组内 advantage 为 0 ≠ 整个 loss 为 0（KL 等项仍可能有贡献）。
- GRPO 有标准差归一化、长度归一化、裁剪等实现变体；具体公式回到所用的 verl 版本看，不要凭名字猜。

### 7.5 RLOO

**【原理】** 基线用同 prompt 下**除自己以外**其他样本的均值：

$$
A_i = r_i - \frac{1}{G-1} \sum_{j \neq i} r_j
$$

[0, 0.5, 0.5, 1] → [−0.67, 0, 0, 0.67]。历史 1.5B 跑了一个 RLOO 种子，结论（91.2% 增益来自首轮）和 GRPO 一致——"主结论不限于 GRPO"。

### 7.6 PPO 裁剪和 KL

同一批 rollout 可能做多个 minibatch 更新，新策略逐渐偏离产生数据的旧策略，于是用概率比和裁剪限制过大的更新：

$$
\min\big(\rho_t A_t,\ \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t\big), \quad \rho_t = \frac{\pi_\theta(y_t \mid h_t)}{\pi_{\text{old}}(y_t \mid h_t)}
$$

KL 正则约束相对参考策略的偏离。它们解决**优化稳定性**，不会自动修复执行路径，也不保证获得工具能力。

### 7.7 为什么工具不执行，奖励还能涨

`reward_code.compute_score` 会提取最终代码直接跑测试。所以完全可能：

```text
模型没成功调用工具 → 直接给代码 → 奖励函数跑测试 → 正奖励 → 强化"直接给代码"
```

历史 FC 诊断臂里 `critic/rewards/mean` 从 0.233 涨到 0.281——**确实在涨**。这条路径没有任何"读反馈后修复"的经验，但有代码正确性信号。所以**奖励曲线健康不能代替工具路径监测**——这就是为什么要有 §4.2 的计数器和 `guard_tool_calls.sh`。

### 7.8 冷启动风险

工具路径的回报从来没被采样到，直接作答路径持续拿正回报 → 优化持续偏向直接回答，很难自己发现工具路径。强制性 prompt + GRPO 不足以跨过这个冷启动——每一次尝试都在协议层被丢弃，从未产生过一条带正回报的工具轨迹供强化。

但别走到另一个极端说"永远学不会"：参数共享、其他样本、探索都可能让未来行为变化；**有限样本的零 ≠ 策略支撑集里概率为零**。P3 broken 臂的 10 次接受就是必须保留的观察。

**把冷启动从"机制解释"变成因果结论，需要对照训练 + 固定评测**——这就是 P3。

---

## 8. P3 是怎么设计出来的

### 8.1 三个互相独立的验收问题

1. **通道恢复了吗？** 正确的调用被接受、执行，observation 回到下一轮输入。
2. **模型行为变了吗？** 同一评测条件下，训练后更常正确调用工具或更会利用反馈。
3. **任务结果提高了吗？** 固定任务集上的最终成功 / rescue 有可支持的变化。

第一个成立，不会自动带出后两个。"机制恢复"专指第一个。

### 8.2 最小的干净对照

同一个初始 7B checkpoint，两个 LoRA 训练臂，固定数据、prompt、schema、reward、步数、样本预算、seed：

```text
broken  : actor_rollout_ref.rollout.multi_turn.format=hermes
repaired: actor_rollout_ref.rollout.multi_turn.format=qwen2_5_coder
```

`p3/launch_arm.py` 的 `plan()` 里就是这一行三元表达式——两臂的命令行只在这里不同。

### 8.3 为什么评测接口也要统一

如果 broken 臂只用 broken parser 评测、repaired 臂只用 repaired parser 评测，看到差异时你无法知道是**训练学到了不同策略**，还是**评测时一边仍然不能调用**。

```text
θ_broken  ─┐
            ├→ 同一个可工作的评测接口 → 比较学习结果
θ_repaired ─┘
```

P3 的预注册评测用路径 C（绕开两个训练 parser）；审稿后再加路径 D（同一个修好的 parser）。两者回答的问题不同：C 测"被动反馈下的修复"，D 测"同一可用接口下的自主调用与修复"。

### 8.4 为什么一定要 step 0

如果训练前 repaired 配置就已经能做很多工具动作，训练后的高分可能主要是接口带来的即时改善，而不是 RL 增益。要比的是两臂**变化之差**：

$$
\Delta_{\text{learn}} = (R_{\text{rep},150} - R_{\text{rep},0}) - (R_{\text{broken},150} - R_{\text{broken},0})
$$

P3 两臂 step 0 用同一个 base checkpoint 和同一个评测器，所以 step 0 都是 418/430/12——这是**观察**，不是用来推断评测设计的证据。

**不能先看 repaired 涨不涨，再决定要不要补 broken**——那样实验设计就受了已观察结果的影响。

### 8.5 几种不正确的判断

| 容易做出的判断 | 为什么不够 |
|---|---|
| reward 涨了，所以学会了多轮 | reward 可以来自直接答案 |
| num_turns 涨了，所以利用了反馈 | 多轮可能只是重复、循环、错误 |
| repaired 比自己 step 0 好，所以 parser 造成了学习 | 缺匹配的 broken 趋势和统一评测 |
| *p* > 0.05，所以修复没用 | 可能样本太少，不能证明等价 |
| 7B repaired 学会了，所以 1.5B 的历史观察被推翻 | 不同模型、协议、预算 |

### 8.6 开跑前必须回答的技术问题（P3 实际都做了）

| 问题 | P3 的做法 |
|---|---|
| 评测用的是当前 LoRA 还是 base？ | `NativeRolloutPolicy` 每个请求核验 `global_steps == expected_step`，不等就抛错；`CODE_EVAL_REQUIRE_NATIVE=1` |
| 评测真的触发了吗？ | 每个评测步写 `step_XXXXX.jsonl` + `.complete.json`（含 sha256 和指标），缺了就不算 |
| 不同 arm、重试、step 的输出会不会混？ | `launch_arm.py` 按 `RUN_ID/arm` 建**独占目录**（`root.mkdir()` 已存在就失败，"exclusive launch lock; retained on failure, never removed"） |
| 题目身份唯一吗？ | `unique_ids()` 检查；repair probe 重复 task_id 直接报错 |
| 原文、parser 选择、接受对象、observation 有没有保留？ | `rollout_probe` 把每次 extract 的原文、哈希、parser 名、接受数写进 events |
| 跑的代码是不是预期那版？ | `launch.json` 记录 15 个关键源文件的 sha256 |
| checkpoint 能恢复吗？ | `verify_formal_resume()`：见 8.7 |
| 磁盘够吗？ | 见 §9.4 |

`save_freq > 0` 只是保存设置，不保证能完整恢复。历史 runner 默认复用目录、包含清理操作，还有 `resume_mode=disable`——不能当成可直接重跑的手册。

### 8.7 中断与恢复：`verify_formal_resume()` 做了什么

repaired 臂在 step 116 后中断。恢复前，`launch_arm.py` 的 `verify_formal_resume()` 强制检查：
1. 源 checkpoint 含模型、optimizer、RNG/scheduler（`extra_state`）、dataloader（`data.pt`）四类状态；
2. 源 checkpoint 属于同一个 run 和 arm，且原始 launch 是 formal 模式；
3. **新旧命令行的"科学配置"完全一致**——只允许 `resume_mode`、`resume_from_path`、`default_local_dir`、`experiment_name` 这几个运维字段不同，其他任何漂移直接拒绝；
4. 恢复点那一步的评测完整存在、哈希匹配、且核验过 native weight step；
5. 除启动器本身外，所有运行时源文件哈希和原始 launch 一致。

通过后从**验证过的 step 90** 恢复（而不是中断点 116），被放弃的 26 次更新（91–116）排除在正式谱系外；恢复后重复的 step-90 评测差了 2 题（428 vs 426），作为测试–复测变异报告，不替换曲线。

**教训**：恢复不是"接着跑"，而是一个需要证明"除了簿记什么都没变"的契约。

---

## 9. 工程：显存、LoRA、checkpoint、磁盘、Ray

### 9.1 为什么 7B 推理能跑，训练会 OOM

bf16 每参数 2 字节，7B 纯权重约 14 GB。但训练还要梯度、优化器状态（AdamW 每参数两个 fp32 动量）、激活、临时张量、可能的参考模型；rollout 侧还要推理副本和 KV cache。所以不能用"80 GB 减 14 GB 还剩很多"估计 RL 能不能跑，也不能用 rollout 阶段的低显存保证后面的 backward 安全——7B rollout 探针就是在 rollout 之后的 actor 更新阶段 OOM 的。

### 9.2 LoRA

$$
W' = W + \frac{\alpha}{r} BA
$$

W 冻结，只训 A、B。P3：r = α = 32 → scaling 1，作用于所有线性层，392 个张量、80,740,352 个参数，导出的 adapter 约 155 MiB。

LoRA 省的是可训练权重、梯度和优化器状态；base 模型、激活、生成缓存不会消失。两臂用同一个 LoRA 配置有利于臂间比较，但**不能**和历史 1.5B 全参训练比学习效率。

### 9.3 降低显存的办法

| 方法 | 省什么 | 代价 |
|---|---|---|
| LoRA | 可训练权重、梯度、优化器状态 | 不等于全参 |
| activation checkpointing | 中间激活 | backward 重算 |
| CPU offload | 部分状态放 CPU | PCIe 传输、CPU 内存压力 |
| 小 microbatch | 单次前后向的激活峰值 | 变慢，注意有效 batch |

`train_batch_size`、PPO minibatch、microbatch、`rollout.n` 是不同的参数。16 道 prompt × 8 条轨迹 = 128 条；每条轨迹可能多次 generation，所以 generation 日志不必正好 128 行。

### 9.4 checkpoint 与磁盘

三种文件别混：
- **推理权重**：用于生成。
- **LoRA 增量**：要配对 base 模型才有意义。
- **训练态 checkpoint**：权重 + optimizer + scheduler + RNG + dataloader 位置。adapter 的大小不能用来推断完整 checkpoint 的大小。

**本项目和同服务器栈上踩过的磁盘坑**：
- 历史 1.5B 训练的 step-75 评测**丢了**：step 75 的 checkpoint 写盘要 4.86 GB，磁盘只剩 4.6 GB，训练跑完后进程 abort。论文里写明了这次丢失，没有留一个无解释的缺口。
- verl 0.9 默认保存的 checkpoint **包含优化器状态**，体积是纯权重的好几倍（同栈上 4B 全参一份约 47 GB，去掉 optimizer 约 17 GB）。
- `trainer.max_actor_ckpt_to_keep` 在同栈 verl 0.9 的一条路径上**观察到不生效**——旧 checkpoint 一份都没删，堆满磁盘后崩在 `_save_checkpoint`。
- 写新 checkpoint 时新旧并存，所以 `max_to_keep=2` 可能短时占三份。

**纪律**：开跑前算 `空闲 ≥ 2 × 单份产物 + 日志/数据/临时文件 + 余量`；数据盘和系统盘分开检查（系统盘只有 30 GB，`HF_HOME`、vLLM 编译缓存必须显式指到数据盘）；**把这个检查写进启动脚本**，不足就中止。

### 9.5 Ray、钩子与"看起来装上了"

- verl 训练主循环跑在 `TaskRunner` 这个 **Ray actor（独立进程）**里。只在 driver 里 patch `RayPPOTrainer._validate` 对它无效——日志里有"已挂载"，但**一条评测数据都不产出，训练看起来一切正常**。用 `sitecustomize.py`（§4.2）。
- **`launch_ppo.py` 不能用 `python -c`**：hydra 的 `@hydra.main` 依赖 `sys.argv[0]`，`-c` 模式会把参数解析弄乱；用真正的脚本文件。
- **flash-attn shim**：verl 的 log-prob 计算硬依赖 `flash_attn.bert_padding`，而 flash-attn 预编译轮子只到 torch 2.9，和本机 torch 2.13 ABI 不符——`launch_ppo.py` 第一件事就是把 `flash_attn_shim` 放到 `sys.path` 最前面。
- **Ray 的 AF_UNIX socket 路径上限 107 字节**：`launch_arm.py` 把 Ray 临时目录压成 `rp3/<12 位哈希>`，超过 42 字符就拒绝启动。
- verl 默认往 wandb 打日志，没配 key 会崩在启动阶段：传 `trainer.logger='[console]'`。

### 9.6 远程运维的坑（AutoDL）

| 坑 | 症状 | 对策 |
|---|---|---|
| `pgrep -f xxx` 匹配到发起检查的命令自己 | 看门狗恒判"训练还活着"，永不触发 | 用 `ps -eo args \| grep "[x]xx"` 或按 PID |
| `ssh -n host 'cat > f' < local` | 传过去的是 **0 字节文件**，不报错 | `-n` 只在启动后台任务时用；传文件用 `tar czf - … \| ssh host 'tar xzf - -C dir'` |
| `setsid nohup cmd &` | 进程若已是组长，`setsid()` 失败直接 exec，ssh 一断整棵进程树没了 | 用 `setsid --fork` |
| SSH 中途断 | 长命令执行一半（旧 vLLM 苟活、端口冲突） | 命令拆短；长任务写成服务器上的脚本用 setsid 启动，脚本自己做校验 |
| OpenBLAS 按上百核开线程池 | 受限子进程里 `import numpy` 报内存分配失败 | `OPENBLAS/OMP/MKL/NUMEXPR_NUM_THREADS=1` |
| hf-mirror 下载 401 | — | `HF_HUB_DISABLE_XET=1` |
| 无卡模式只有约 2 GB 内存、0.5 CPU | 任何重计算都 OOM；`free` 显示的可能是宿主机资源 | 读 cgroup 限额；无卡模式只写代码、装包、准备数据（这也是 `code_tool_core.py` 和 `code_tool.py` 拆两层的原因：verl 装不进无卡模式，纯逻辑要能本地测） |
| 自动关机守护 | 数据没同步完就断电 | 先 `sync` 再留缓冲 |

### 9.7 估时间

总耗时 = 下载与启动 + rollout + 奖励执行 + update + 评测 + checkpoint IO。用当前环境**已完成**的完整 step 和评测实测，分项统计；不要拿纯生成吞吐估整个训练。repaired 臂调用和上下文更长，wall time 和 broken 不一样——"两臂跑同样小时数"和"同样步数/rollout 预算"不是同一个控制条件，公平性要事先定义（P3 定义为同样步数和 rollout 预算）。

---

## 10. 全项目的元教训（每条都对应一次真实事故）

按重要性排：

1. **声称修好而未验证生效，比不修更危险。** `max_tokens` 修复只替换了字面量，没碰到函数默认参数；provenance 却写成已修。它同时污染了数据和对数据的信任。此后所有修复都用外部验证核对，不采信"补丁已应用"。（§6.2）
2. **配置日志 ≠ 生效配置，看产物。** hydra 启动时打印的配置树是合并前的默认值快照；`resume_mode=auto` 会从 checkpoint 恢复旧 scheduler、覆盖配置里的学习率。判断配置是否生效，一律看产物大小或运行时读数，并把关键配置的运行时校验写进启动脚本。
3. **不要用 `except: pass` 抹平不同的失败原因。** 26 条空代码因此无法归因。（§6.2 洞 1）
4. **警告没有强制力。** 要让错误必须被处理，就用非零退出码或异常。（§6.2 洞 4；`probe_react_full.py` 的 exit 2）
5. **守护进程必须自证在工作。** 关机守护启动即失败、`grep -c` 多行输出把比较搞崩、`pgrep` 匹配到自己——三次"看起来装上了，其实没跑"。启动后要确认进程在、日志文件存在且有内容。P3 探针"任何钩子触发 0 次就写 `P3_INVALID`"是同一原则。
6. **每条数据带 provenance 落盘，而且记的是实际生效的值。**
7. **公平 ≠ 相同参数值。** 两个协议的 token 效率不同，同样的上限就不公平。
8. **观测到 0 ≠ 概率为 0。** 从有限样本推全称命题要加"under the observed …"。
9. ***p* > 0.05 ≠ 没有效应。** 同时报效应量、置信区间、配对数。
10. **排除了 N 个替代解释 ≠ 找到了真原因。** 单变量对照只能否定。
11. **阴性臂需要救援臂才有意义。** Granite 在自己的 parser 下也是 0，才敢判它"无信息"而不是"确证"。
12. **裁决/复核的选择规则必须先定，且不能引用要验证的那个对象。**（§6.14）
13. **让自己有利的错误最危险。** 没去重的首轮数更唬人；这类错误没人会来纠正你。
14. **能拆成 2×2 就别只跑单臂，也要正确命名效应**（简单效应 vs 主效应）。
15. **诊断也会错，而且会错很久。** "P3 卡在 verl 不暴露 guided decoding"躺在复盘里很多天——它和论文自己的论点矛盾。**自相矛盾就是诊断出错的信号**。
16. **长任务开跑前先算磁盘。**（§9.4）
17. **有些坑要认下来写进 limitation，而不是立刻修**——修了会切断历史可比性（`clean[:100]`）。
18. **不同精度的测量不能直接相减。** 低精度探针的波动区间内的差，不是新证据。比较前先确认两个数来自同一套测量条件。
19. **源码注释也会过时。** `react_agent_loop.py` 的 docstring 还写着被推翻的"0/1000 从不发起"。用控制流和产物验证注释。
20. **空的输出也是输出，要追。** 一次自评里有两行"删除："内容为空，看到了没查——那正是被一次查找替换损坏、内容重复四遍的结论文件，而且编译不报错。
21. **对方指出的问题和对方给的修法要分开评估。** 别人指出的风险可能是真的，但建议的修法不一定对。
22. **能实测就别估算。** 页数估了很多轮，装上 TeX 实测才发现低估了 11%。
23. **诚实的零结果比勉强的正结果更能撑住论文。** P3 的零结果连同它的边界一起报告，比把 −0.55pp 包装成什么都强。

---

## 11. 证据边界：哪些能讲，哪些不能讲

### 11.1 按证据类型理解项目

| 证据类型 | 材料 | 合理用途 |
|---|---|---|
| 服务/执行侧直接计数 | BFCL 解析与官方 scorer、τ-bench 执行与任务记录 | 描述指定配置下的系统行为 |
| 有人工校准的启发式计数 | Coder tight 阶梯、意图分类 | 描述调用候选，同时报误差 |
| 固定字节重放 | BFCL 97/92、reparse matrix、failure layer | 隔离解析规则对已有字节的作用 |
| rollout 内直接事件 | P3 两臂的 extract/accept/execute/observe 和原文 | 训练采样中哪些调用到达了环境 |
| 单变量训练对照 | P3 broken vs repaired | 机制：强阳性；学习收益：设定限定的零结果 |
| 动机性历史观察 | 1.5B 训练曲线、1.5B 零执行 | 只作动机，不作因果证据 |

### 11.2 需要牢牢记住的十处边界

1. **tight 不等于 JSON 合法。** `analysis/intent.py` 用正则找字段和代码关键词，不做 `json.loads`。准确叫法是 tight-classified call candidate。
2. **no_envelope 不排除 bad payload。** 首失败分类在信封层就返回了。
3. **parser loss = 0 只在被测试的范围内成立。** 不能升级为"所有服务栈都没有 parser bug"；不同 parser、流式/非流式、版本要分别验证。
4. **2×2 的零是基线处的简单效应。** 主效应各 +98。
5. **53 vs 53 不识别失败层。** 只是一致性检查。
6. **P3 旧离线 verl 重放辅助函数曾有实现错误**：`VERL_RE` 只有一个捕获组，`findall()` 返回字符串，旧 `_accepts` 却取 `m[0]`，把首字符当完整 JSON。后续诊断实现已修正；实际 parser 的 accepted 日志是独立测量，不受影响。用新的诊断产物，不沿用旧派生标签。
7. **broken 是 nearly closed，不是 literally closed**（10 次执行）。
8. **P3 的零结果是设定限定的**：一个种子、7B、LoRA、一个代码任务、150 步。不是等价性结论。
9. **程序化反馈评测不测自主发起**；原生 FC 评测补上了，但仍是 post-hoc、非逐字节训练重放。
10. **旧 step-0 的 20/1108 已作废**，只读 542 题的正式重跑。

### 11.3 几个看起来谨慎、实际上仍然过强的句式

- "新模型换对 parser 成功，所以旧 parser 实现一定没问题。"——成功对照不能免除实现审计。
- "ReAct 的 token 预算以前更小，修正后只会更好。"——更长预算不保证单调改善，要看重跑。
- "有 98 行 preflight，所有静默失败都能发现。"——它只做 auto/required 请求、工具名和参数检查，没做真实 executor → observation 的端到端测试。
- "一共有 50 个臂，所以有 50 个独立验证。"——有不合格臂，也共享题目和条件；数量不能替代独立性。
- "reward 曲线平滑，所以没有 reward hacking。"——没有异常不是排除利用的证据（论文的写法是"审计了 5 个保留生成代码的日志，0 次出现签名"，并说明审计的覆盖边界）。
- "同一个 seed 控制了所有随机性。"——异步调度、数值实现、batch 组合、模拟用户都可能产生差异（BFCL repaired 臂两次差 1 个 case）。

### 11.4 怎样稳妥地描述自己的工作

> 搭建并审计了一个从模型输出到工具反馈的评测流程，在自研代码任务、BFCL 两个类别和 τ-bench retail 上比较接口配置，观察到解析和执行层的巨大差异；把同一机制测进了训练 rollout，并做了一个只换训练 parser 的 7B RL 对照。对规模趋势和 RL 学习效果，分别报告了证据范围，没有把工具执行的恢复当成学习成功。

更完整的"误读 vs 正确说法"对照表见 [PAPER_STUDY_GUIDE.md §7](PAPER_STUDY_GUIDE.md)。

---

## 12. 实验设计与统计（精简版）

> 完整的统计讲解（McNemar 手算、Clopper–Pearson 区间构造、κ、析因设计、后处理条件化）见 [PAPER_STUDY_GUIDE.md §2.10](PAPER_STUDY_GUIDE.md)。这里只讲**跑实验时**要记住的东西。

### 12.1 "只改一行配置"不等于"只改一个机制"

换模板的一行配置，同时改变了工具说明、few-shot 和序列化格式；`strict` 一个开关改变了生成分布。好的实验描述会列出**实际改变的内容**，并把它整体当成干预。BFCL 的 adapter 级干预本身就有价值，不需要伪装成 parser-only——2×2 就是为了把它拆开。

### 12.2 为什么必须同题配对

两种配置在同 100 题上分别过 49 和 61。只看总数，不知道变化来自哪几题：

| | 新配置过 | 新配置不过 |
|---|---|---|
| 旧配置过 | 46 | 3 |
| 旧配置不过 | 15 | 36 |

净增 12，但反映方向的是 15 vs 3。同样净增 12，如果是 30 vs 18，不确定性完全不同。**日志必须保留 task_id**。精确 McNemar：15/3 → *p* ≈ 0.00754；3/0 → *p* = 0.25（τ-bench 的 7→10"只赢不输"，但有方向信息的只有 3 题）。

注意仓库里不同脚本的检验不一样：`analyze_step3.py` 里有连续性校正的近似形式。引用前说明是精确还是近似。

### 12.3 *p* 值、效应量、多重比较

- *p* 值不是效应大小，也不是结论为真的概率。*p* > 0.05 可能是效应小，也可能是样本少。
- 论文的 27 个检验用 Bonferroni 阈值 0.05/27 ≈ 0.00185，**回顾性、非预注册**。判断时用没四舍五入的值：0.00188 没过，0.00183 过了。
- 稀疏配对（只有 3 个不一致对）不能用正态近似，要用精确二项界构造的保守区间。

### 12.4 什么时候"零"很有力

- 对**已存字节**，给定确定的 parser 重放为零，是确定性结果。
- 但字节本身是采样的。推广到"新题也几乎总是零"仍有抽样不确定性。同题多轮、多次生成彼此相关，不能当独立样本扩大 n；训练结果还要考虑 seed，而不是把同一 seed 的上万 token 当独立证据。

### 12.5 后处理条件化

只比较"两臂都解析成功"的题，筛选条件本身已经受接口影响——新 parser 额外救回的难题被排除了。可以作描述性分析，但要写明筛选规则，不能当作"去掉接口影响后的纯协议效应"。同理，"只在各模型自然失败的题上测修复率"也难以公平比较——所以要有固定 buggy 初稿的 repair 通道。

### 12.6 人工标注

precision 回答"判为正的里有多少真为正"；κ 回答"扣除偶然一致后的一致度"。κ 高不代表判断一定正确——两个人可能受同一个显示截断影响。校准样本分层抽样，不能假定各尺寸误报率相同。

---

## 13. 按源码学习：五次阅读路线

每次只追**一条数据流**，在纸上写下一个变量经过的每个位置。第一次不需要读懂整个文件。

### 13.1 第一次：从服务请求追到执行（路径 A）

读 [runs/final/probe_react_full.py](https://github.com/nebula-1999/Interface-Induced-Trajectory-Censoring/blob/main/runs/final/probe_react_full.py)：
1. 找 `gen_fc()`：请求体里的 `tools`、`tool_choice: "auto"`、schema 怎么选（terse / rich / strict）。
2. 看返回的 `tool_calls` 怎么处理；找到 wrong-tool 分支，理解为什么它不执行。
3. 找调用 `run_tests` 的地方，追 code 和 test 从哪来。
4. 找 observation 拼回 messages 的地方，确认下一轮模型真的能看到。
5. 找轨迹落盘字段：parser 失败时保存了哪些原文？`n_err` 怎么变成退出码 2？

### 13.2 第二次：从训练输入追到奖励（路径 B + E）

| 文件 | 要回答的问题 |
|---|---|
| [prepare_code_data.py](https://github.com/nebula-1999/Interface-Induced-Trajectory-Censoring/blob/main/prepare_code_data.py) | prompt、测试、`agent_name`、`tools_kwargs` 怎样进入数据行？ |
| [launch_ppo.py](https://github.com/nebula-1999/Interface-Induced-Trajectory-Censoring/blob/main/launch_ppo.py) | 为什么先插 flash_attn_shim、再 import code_patch？ |
| [sitecustomize.py](https://github.com/nebula-1999/Interface-Induced-Trajectory-Censoring/blob/main/sitecustomize.py)、[p3/sitecustomize.py](https://github.com/nebula-1999/Interface-Induced-Trajectory-Censoring/blob/main/p3/sitecustomize.py) | 钩子怎样进入每个 Ray actor？为什么要等目标模块"本身"导入完成？为什么要接力？ |
| [p3/rollout_probe.py](https://github.com/nebula-1999/Interface-Induced-Trajectory-Censoring/blob/main/p3/rollout_probe.py) | `install()` 怎样注册 `qwen2_5_coder` 并包住每个 parser 的 `extract_tool_calls`？写进 events 的字段有哪些？ |
| [qwen_tools_parser.py](https://github.com/nebula-1999/Interface-Induced-Trajectory-Censoring/blob/main/qwen_tools_parser.py) | 三种格式的优先顺序；为什么要按 tools 白名单校验名字、要求 code 非空？ |
| [code_tool.py](https://github.com/nebula-1999/Interface-Induced-Trajectory-Censoring/blob/main/code_tool.py)、[code_tool_core.py](https://github.com/nebula-1999/Interface-Induced-Trajectory-Censoring/blob/main/code_tool_core.py) | 工具拿到哪些测试？计数器和凭据文件写在哪？observation 怎么压缩（`format_observation`）？ |
| [react_agent_loop.py](https://github.com/nebula-1999/Interface-Induced-Trajectory-Censoring/blob/main/react_agent_loop.py) | stop 串、mask、`num_turns` 怎么算？ |
| [reward_code.py](https://github.com/nebula-1999/Interface-Induced-Trajectory-Censoring/blob/main/reward_code.py) | 最终代码怎样提取？哪三种情况返回 0？ |
| [sandbox.py](https://github.com/nebula-1999/Interface-Induced-Trajectory-Censoring/blob/main/sandbox.py) | pytest vs script 两种模式；`total`、`pass_ratio`、`all_passed`、`no_tests` 的定义 |

**特别注意**：函数存在，不代表框架会调用它（`calc_reward`）；注释说会调用，也不代表调用路径真实存在。确认入口、调用者和运行事件。

### 13.3 第三次：评测与落盘（路径 C）

读 [code_patch.py](https://github.com/nebula-1999/Interface-Induced-Trajectory-Censoring/blob/main/code_patch.py) → [code_eval_hook.py](https://github.com/nebula-1999/Interface-Induced-Trajectory-Censoring/blob/main/code_eval_hook.py) → [chat_policy.py](https://github.com/nebula-1999/Interface-Induced-Trajectory-Censoring/blob/main/chat_policy.py) → [eval_decompose.py](https://github.com/nebula-1999/Interface-Induced-Trajectory-Censoring/blob/main/eval_decompose.py)，对着代码回答：
- 哪个条件触发 `_validate` 里的分解评测？
- 请求里有没有 `tools`？（没有）
- 谁运行测试，谁决定继续下一轮？
- `multi` 和 `repair` 分别选了哪些 task_id？
- `NativeRolloutPolicy` 怎样核验权重步？为什么普通 HTTP 请求可能评测到 base model？
- 历史 `dump()` 是**追加**模式（`open(path, "a")`）——多个 seed 写同一目录会混进不同 run 的记录（`code_patch.py` 注释专门提醒了这一点）。P3 改用 `StepArtifact` 按步写文件 + 完成标记。
- HTTP 失败如何进入最终指标？

### 13.4 第四次：原生 FC 评测（路径 D）

读 [p3/native_fc_eval.py](https://github.com/nebula-1999/Interface-Induced-Trajectory-Censoring/blob/main/p3/native_fc_eval.py) 的 `accepted_calls`、`final_code`、`evaluate_task`，和 [p3/test_native_fc_eval.py](https://github.com/nebula-1999/Interface-Induced-Trajectory-Censoring/blob/main/p3/test_native_fc_eval.py)：
- 什么情况才触发真实工具执行？
- 第二个被接受的调用会怎样？
- rescue 怎么定义？和路径 C 的有什么不同？
- 终端评分为什么不返回反馈？

### 13.5 第五次：从结果表反查原始数据

挑一张表，不要一次扫整个仓库。例如 τ-bench：

```text
论文表
  ← p5/analyze_tau.py 的指标定义
  ← p5/results_full/ 两臂 task records
  ← task_id / trial / traj / reward
```

随机挑一题，手数 assistant 消息、tool_calls、tool 消息、最后的 reward，再和脚本结果比较。不相等时先查单位和终止条件，不急着换公式。

同样可以做：BFCL 2×2 ← `runs/p1_2x2/`；失败层 ← `analysis/failure_layer_table.py`；27 个检验 ← `analysis/submission_statistics.py`（CPU 上可复算）。

### 13.6 目录导航

| 目录 | 内容 | 读的时候注意 |
|---|---|---|
| `runs/final/` | 50 个受控探针臂、ERRATA、hanXen 插件 | 有 superseded / 不合格臂（ERRATA 里列着） |
| `runs/p1_clean/`、`runs/p1_2x2/`、`p1/` | BFCL clean run 与 2×2 | 恢复批和干净批不要混表 |
| `runs/p2_pytest/` | Instruct 执行层重跑 | 与缺 pytest 的原批、离线 replay 区分 |
| `runs/p4/`、`p4/` | Qwen3 / Granite 结果与预注册 | 结合预测修订记录读 |
| `p5/` | τ-bench 全量 | 不要拿 smoke 当全量 |
| `p3/` | rollout 探针、正式 P3、原生 FC 评测的脚本、结果和验收记录 | 不同 run、channel 不能按行直接合并 |
| `analysis/` | 所有离线分析脚本 | 部分脚本需要完整档案（见 `analysis/VERIFICATION_SCOPE.md`） |
| `submission_packages/native_fc/` | 原生 FC 配置与 1,626 条题目记录 | 不含生成代码和完整对话 |
| `validation/` | 人工标注、裁决、评分材料 | 注意展示字节一致性和抽样分层 |
| `fixes/` | 第一版探针的审计 | 8 个洞的原始记录 |
| `internal/handoffs/` | 不同会话/代理之间的交接记录 | 是历史状态，不是当前事实 |
| `paper/` | LaTeX 源码 | 文稿结论要反查代码和原始结果 |

---

## 14. 再开机之前：验收清单

### 14.1 写一张 run card

一页就够，能避免很多事故：

```text
科学问题：
主指标（以及它测的是漏斗的哪一层）：
两臂的实际差异（列出全部，不只是"改了一行"）：
模型 / tokenizer / 模板 / parser 的 revision 与哈希：
数据 manifest 与题数：
seed / 解码参数 / 上下文与输出预算：
训练步数 / rollout 数 / 评测步：
reward 与 executor 版本：
输出目录与 run_id（是否独占）：
checkpoint：保存内容、单份大小、磁盘余量、恢复验收方式：
哪些错误会让哪一层指标失去解释资格：
停止条件与守护（守护怎样自证在工作）：
```

"机器开了、模型能下载、服务能启动"只完成了基础设施的一部分，不等于实验准备好了。

### 14.2 preflight 至少三层

1. **接口层**：请求带了 tools；模板确实注入了；已知合规的文本能被正确的 parser 识别；工具名和参数校验生效；`tool_choice: required` 作为阳性对照。（[preflight_toolcall.py](https://github.com/nebula-1999/Interface-Induced-Trajectory-Censoring/blob/main/preflight_toolcall.py) 覆盖这一层的一部分。）
2. **执行层**：一个已知会通过的程序、一个已知会失败的程序，都得到预期结果；确认测试真的跑了（不是 `no_tests`）；错误确实回到下一轮输入。
3. **训练层**：小 rollout 能收集；完整 update 能完成（7B 探针就是死在这一步）；评测读到的是更新后的权重；checkpoint 写完并能恢复。只跑完 generation 不能验收训练。

preflight 一题 `auto` 不调用，也可能是模型选择直接回答——要结合原文判断，不能自动贴"错配"标签。

### 14.3 退出码与"零请求错误"

`rc=0` 只表示脚本按它自己的逻辑正常退出。如果脚本没检查 pytest 是否存在、没检查产物数量、没检查钩子是否装上，它仍可能"正常结束"一个无法回答问题的实验（P2 初次运行就是这样）。

分层记录：HTTP errors、parser errors、wrong tools、executor unavailable、test failures、observation failures。不要把环境故障归为模型失败，也不要把"parser 不接受"这个研究对象本身当成无效实验。

### 14.4 "零执行"时怎样证明计数器没坏

不能要求阴性臂的执行钩子必须触发——否则真正零执行的臂永远过不了验收。合理做法是三种证据分开：
1. **安装记录**：钩子装上了（`rollout_probe` 写 `register_custom_parser` 事件）；
2. **阳性功能检查**：在独立的阳性条件下它确实会触发；
3. **阴性臂结果**：在阴性臂上观察运行事件。

历史诊断用了三个独立信号互相印证：`num_turns` 恒为 2、工具耗时 0.00 s、专用计数器 0 行（同时 ReAct 臂 788 行证明计数器本身能工作）。

### 14.5 数据完整性不只是"存了全文 + 哈希"

哈希能证明文件后来没被改，不能证明：没漏 generation、没把两个 run 混在一起、task_id 没错配、模型看到的输入和记录一致、计数来自正确的工具入口。要同时保存最终配置、事件身份、代码版本和完成标记。**哈希不能替代把文件给人检查**——这也是补充包后来加入原始内容样本的原因。

---

## 15. 练习（附答案）

### 15.1 运行配套脚本

```bash
python3 -B learning_lab.py
```

只用标准库，演示：同一合法裸调用为什么被一条规则识别、被另一条忽略；JSON 合法 / 工具合法 / 语法合法 / 任务正确为什么不是一回事；2×2 的简单效应、平均主效应、交互；组内相对奖励；救回与回退的分解；精确 McNemar。把每一行输出解释一遍。

### 15.2 十个基础问题

**Q1. 模型在 content 里写了 `run_tests`，算调用了吗？**
<details><summary>答案</summary>不算。可能在讨论工具、复述 schema，也可能是没被识别的候选。要继续查文本、parser、参数、执行和反馈。</details>

**Q2. `required` 成功、`auto` 为零，能宣布 auto 的 parser 坏了吗？**
<details><summary>答案</summary>不能。两者可能走不同的生成约束；模型在 auto 下也可能选择直接回答。查原文和实际解析路径。</details>

**Q3. 工具执行次数增长，说明学会多轮了吗？**
<details><summary>答案</summary>不说明。可能反复提交同样错误的程序。需要固定评测下的调用质量、反馈利用和 rescue。P3 就是执行次数涨了三个数量级、rescue 不变。</details>

**Q4. 为什么奖励函数跑了很多次测试，而 agent 工具调用为零？**
<details><summary>答案</summary>训练器要给最终代码打分（路径 E），这些测试结果不进入模型的上下文。</details>

**Q5. 同权重、同题、不同模板，输出会一样吗？**
<details><summary>答案</summary>不保证。模板是输入的一部分，输入 token 序列已经变了。</details>

**Q6. 框架里的 `num_turns = 2` 是调用了两次工具吗？**
<details><summary>答案</summary>不是。在 verl 的这个实现里 2 是最小值（初始 user + 一次 assistant），意味着没有任何工具轮。要看实现定义和工具专用事件。</details>

**Q7. 更大的模型 tight 数更多，说明它更容易触发 parser bug？**
<details><summary>答案</summary>不说明。parser 规则始终相同且按设计运行（parser loss = 0）；变化的是模型输出。</details>

**Q8. vLLM 服务端装了 hanXen parser，训练里的调用就能被识别了吗？**
<details><summary>答案</summary>不能。训练走 verl 自己的 ToolParser 注册表（multi_turn.format 选择），必须在 verl 里注册，并且要在每个 Ray actor 进程里注册。</details>

**Q9. P3 的 broken 训练臂几乎没采到工具经验，为什么评测里也有 12 个 rescue？**
<details><summary>答案</summary>评测走路径 C：评测器无条件执行代码、以 user 消息回反馈，不需要模型发起调用。</details>

**Q10. 用普通 OpenAI HTTP 请求评测训练中的 LoRA 有什么风险？**
<details><summary>答案</summary>HTTP 端点只知道 base model 的注册表；verl 加载的 LoRA 是通过 native generate 以 LoRARequest 注入的。普通 HTTP 请求可能静默地评测了 base model。所以 P3 用 native rollout RPC 并核验每个请求的 global weight step。</details>

### 15.3 五道手算题

**A.** 100 题首轮过 53；9 题后续救回；4 题首轮过但最终回退。最终过多少？
<details><summary>答案</summary>53 + 9 − 4 = 58；final − turn1 = 5，不等于 rescue = 9。（路径 C 通过即停，所以那里不会出现回退。）</details>

**B.** 合法 JSON `{"name":"add","arguments":{"a":1,"b":2}}`，只声明了 `run_tests`。在哪一层失败？
<details><summary>答案</summary>工具名白名单校验，不是 JSON 语法。如果 parser 本身不查名字，可能先解析成功再被宿主拒绝——这就是 `qwen_tools_parser.py` 要在 parser 内按 tools 白名单过滤的原因。</details>

**C.** 两臂都成功 7 题，新臂另外成功 3 题，旧臂没有独有成功。该报什么？
<details><summary>答案</summary>旧 7、新 10，净增 3，不一致对 3/0，精确双侧 *p* = 0.25。报方向和效应大小，不宣称显著。（这就是 τ-bench。）</details>

**D.** 一组 8 个 rollout 奖励全是 0.5，GRPO 的组内信号怎样？
<details><summary>答案</summary>减均值后全为 0，这组没有基于奖励差异的优势；不要推出"全部参数更新都为零"（KL 等项仍可能有贡献）。</details>

**E.** 2×2 四格 0/0/0/196（/200）。平均主效应、基线简单效应、交互各是多少？
<details><summary>答案</summary>平均主效应各 +98（比例 0.49）；基线处单换的简单效应各 0；交互 196（比例 0.98）。说明用的是哪种定义。</details>

### 15.4 四个应该亲手完成的小任务

1. **画路径**：不看文档，画出 §4 的四条路径 + 路径 E，标出 parser 在哪、谁跑测试、谁看到反馈、反馈角色。
2. **读一条轨迹**：从 `runs/final/` 或 `p5/results_full/` 挑一题，只读不改。写下题目身份、调用候选、工具名、参数、执行、observation、最终结果；缺字段就写"无法从此记录确认"。
3. **审核一句话**：挑论文里一句强结论，写出它的证据文件、计数单位、替代解释和最小可支持的措辞。
4. **写一张 run card**：假设要给 P3 加第二个种子，按 §14.1 写完整（不启动实验）。重点想清楚：磁盘、恢复、评测权重核验、输出目录独占。

---

## 16. 术语表与原始资料

### 16.1 术语表

| 术语 | 在本项目中的含义 |
|---|---|
| Agent | 模型策略 + 环境交互循环组成的系统 |
| Tool calling / FC | 模型提出工具名和参数，由外部系统执行 |
| ReAct | 推理与动作交替；本项目用文本协议实现，由宿主正则解析 |
| Chat template | 把 messages 和工具描述渲染成模型输入的 Jinja 规则 |
| Envelope | 标识调用边界的外层标签（`<tool_call>`、`<tools>`） |
| Parser | 把输出文本转成结构化调用的程序（vLLM 和 verl 各有一套） |
| Schema | 参数形状的约束 |
| Constrained decoding / `strict` | 生成时限制可输出的形式 |
| Serving adapter | parser + template（+ few-shot）这套服务配置 |
| LoRA adapter | 训练得到的低秩权重增量 |
| Harness | 组织请求、执行、记录、评分的程序 |
| Executor | 实际执行工具动作的组件（`sandbox.py`） |
| Observation | 返回给 agent 的环境反馈 |
| Verifier / reward | 给结果打分的程序；奖励不等于最终成功率 |
| Programmatic-feedback evaluation | 路径 C：评测器无条件执行、user 消息反馈 |
| Native FC evaluation | 路径 D：模型自主调用、tool 消息反馈 |
| Rollout | 当前策略采样得到的一条交互经历 |
| Generation event | 一次模型生成调用（P3 事件计数的单位） |
| Advantage | 样本回报相对基线的好坏 |
| GRPO / RLOO / PPO / KL | 见 §7 |
| Credit assignment | 结果归功于哪些动作 |
| Cold start | 有价值的经验从未被采样到，优化难以发现它 |
| Rescue / regression | 首轮失败最终成功 / 首轮成功最终失败 |
| Repair channel | 固定 buggy 初稿的修复评测 |
| Tight / strong / weak | KodCode 调用候选分类器的三个级别 |
| Failure layer / first-failure | 按第一个失败点归类 |
| Parser loss | 按 parser 自己规则应接受却没接受（全档案 0 例） |
| Masking / constructive suppression | 已有动作被接口抹掉 / 生成时被约束阻止 |
| Positive control / rescue control | 预期应产生明确响应的检查 / 用预期能修复的条件排除替代解释 |
| Confound | 和干预一起变化、可能解释结果的因素 |
| Post-treatment conditioning | 按受干预影响的变量筛选样本 |
| Provenance | 数据来自什么版本、配置、过程的记录 |
| Errata | 对已报告内容的明确勘误 |
| Run card | 一页纸的实验说明（§14.1） |
| Formal resume contract | `verify_formal_resume()` 检查的恢复条件（§8.7） |

### 16.2 原始资料（按"读懂这个项目"的顺序）

1. [Hugging Face Chat Templates](https://huggingface.co/docs/transformers/chat_templating)：messages 怎样变成模型输入。
2. [vLLM Tool Calling](https://docs.vllm.ai/en/stable/features/tool_calling/)：auto、required、parser、strict 的区别（文档会更新，解释历史实验以我们固定的 0.27.1 版本为准）。
3. [verl Agent Loop](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)：把框架对象映射回 §4.2。
4. [ReAct](https://arxiv.org/abs/2210.03629)：Thought / Action / Observation。
5. [EvalPlus](https://arxiv.org/abs/2305.01210)：增强测试为什么改变代码能力评测。
6. [BFCL v4](https://gorilla.cs.berkeley.edu/blogs/15_bfcl_v4_web_search.html) 与 [τ-bench](https://github.com/sierra-research/tau-bench)：基准自己怎样定义任务和评分。
7. [DeepSeekMath](https://arxiv.org/abs/2402.03300)：GRPO 的动机和组内优势。
8. [RLOO](https://arxiv.org/abs/2402.14740) 与 [PPO](https://arxiv.org/abs/1707.06347)：基线和更新稳定化。
9. [LoRA](https://arxiv.org/abs/2106.09685)。

这些是学习通用机制用的公开资料；复现实验必须用本项目固定的版本和产物。我们自己的论文、源码注释、AI 助手的解释，都需要接受同样的证据检查。

---

*本文依据仓库源码、`runs/` 与 `p3/` 下的实验产物、`writeup/PROJECT_RETROSPECTIVE.md` 和论文 arXiv v2 编写。论文的逐表逐数字精读见 [PAPER_STUDY_GUIDE.md](PAPER_STUDY_GUIDE.md)。*
