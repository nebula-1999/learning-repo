# 论文精读学习文档：Interface-Induced Trajectory Censoring

> **对照版本**：arXiv v2（内容与 ICLR 2027 投稿版逐字一致，只是去匿名、页眉改成 Preprint）。
> 源文件：`paper/iclr2027/arxiv_v2/source/main.tex` + `sections/*.tex`。
> 本文档里所有"§x / 表 x"都按这份源码的结构叫法；引用的数字都从源码核对过。
>
> **和 `writeup/PROJECT_LEARNING_GUIDE.md` 的分工**：那份是"按项目和概念"组织的（怎么做出来的、代码怎么跑）；
> 这份是"**按论文本身**"组织的——一句一句、一张表一张表地读，讲清每个数字的分母、它测的是哪一层、
> 能推出什么、**不能**推出什么。两份互补，先读这份更容易把论文讲清楚。

---

<a id="toc"></a>

## 目录

- [0. 怎么用这份文档](#ch0)
  - [0.1 建议学习路线（约 4 个半天）](#ch0-1)
  - [0.2 读每个数字时问自己的四个问题](#ch0-2)
  - [0.3 本文档的标记约定](#ch0-3)
- [1. 一页纸全景](#ch1)
  - [1.1 一句话主张](#ch1-1)
  - [1.2 漏斗（全文的骨架）](#ch1-2)
  - [1.3 四条证据线（对应 Introduction 的四个 bullet）](#ch1-3)
  - [1.4 论文不说的话（背下来，被追问时最容易踩）](#ch1-4)
- [2. 前置知识](#ch2)
  - [2.1 Tool calling 的完整链路](#ch2-1)
  - [2.2 Chat template](#ch2-2)
  - [2.3 vLLM 的 tool parser](#ch2-3)
  - [2.4 Qwen2.5-Coder 到底输出了什么，适配器做了什么](#ch2-4)
  - [2.5 `strict: true` 与约束解码](#ch2-5)
  - [2.6 ReAct vs Function Calling (FC)](#ch2-6)
  - [2.7 多轮代码修复循环里的指标](#ch2-7)
  - [2.8 基准](#ch2-8)
  - [2.9 RL：GRPO、LoRA、verl](#ch2-9)
  - [2.10 统计](#ch2-10)
- [3. 正文逐节精读](#ch3)
  - [3.0 标题与摘要](#ch3-1)
  - [3.1 §1 Introduction](#ch3-2)
  - [3.2 §2 Related Work](#ch3-3)
  - [3.3 §3 Framework and Testable Signatures（理论核心，逐句理解）](#ch3-4)
  - [3.4 §4 Measurement Setup（每个设计选择都有理由）](#ch3-5)
  - [3.5 §5.1 External validity: BFCL and τ-bench](#ch3-6)
  - [3.6 §5.2 Scale under mismatched and matched interfaces](#ch3-7)
  - [3.7 §5.3 Llama: a constraint suppresses the wrong action](#ch3-8)
  - [3.8 §5.4 Repair changes evaluation trajectories](#ch3-9)
  - [3.9 §5.5 Parser repair restores RL experience; no held-out improvement detected（最重要的一节）](#ch3-10)
  - [3.10 §6 Implications and Limitations](#ch3-11)
  - [3.11 §7 Conclusion](#ch3-12)
- [4. 附录逐个精读](#ch4)
  - [4.1 附录 A：Data Errata（"披露而非吸收"）](#ch4-1)
  - [4.2 附录 B：Reproducibility](#ch4-2)
  - [4.3 附录 C：Prompts and tool schemas（英文渲染）](#ch4-3)
  - [4.4 附录 D：Serving and training configuration](#ch4-4)
  - [4.5 附录 E：Arm inventory（50 个臂）](#ch4-5)
  - [4.6 附录 F：Human validation of the intent criterion（审稿人最爱追问的部分）](#ch4-6)
  - [4.7 附录 G：Supplementary results（最长的附录，逐小节）](#ch4-7)
  - [4.8 附录 H：Limitations, in full（17 条，"按怀疑型读者应该应用的顺序"）](#ch4-8)
  - [4.9 附录 I：Statistical estimands and retrospective test family](#ch4-9)
- [5. 数字速查表](#ch5)
  - [5.1 BFCL v4（Qwen2.5-Coder-7B-Instruct，Gorilla `6ea5797`）](#ch5-1)
  - [5.2 τ-bench retail（115 任务，本地 Llama-3.1-8B 用户模拟器）](#ch5-2)
  - [5.3 规模](#ch5-3)
  - [5.4 失败层（no envelope / malformed / strictness / parser loss）](#ch5-4)
  - [5.5 Llama-3.1-8B](#ch5-5)
  - [5.6 Qwen2.5-Coder-7B 评测修复](#ch5-6)
  - [5.7 正式 7B RL（P3）](#ch5-7)
  - [5.8 Native FC 评测（post-hoc）](#ch5-8)
  - [5.9 历史 1.5B](#ch5-9)
  - [5.10 人工验证](#ch5-10)
  - [5.11 杂项](#ch5-11)
- [6. 代码与数据地图](#ch6)
  - [6.1 服务与探针](#ch6-1)
  - [6.2 分析脚本（`analysis/`）](#ch6-2)
  - [6.3 训练](#ch6-3)
  - [6.4 图](#ch6-4)
  - [6.5 论文源码](#ch6-5)
- [7. 常见误读与正确说法](#ch7)
- [8. 自测题（附答案）](#ch8)
  - [⭐ 基础](#ch8-1)
  - [⭐⭐ 理解](#ch8-2)
  - [⭐⭐⭐ 审稿人级追问](#ch8-3)
- [9. 术语表](#ch9)
- [10. 延伸阅读](#ch10)

---

<a id="ch0"></a>

## 0. 怎么用这份文档

<sub>[↑ 回到目录](#toc) ｜ [下一章：1. 一页纸全景 →](#ch1)</sub>

<a id="ch0-1"></a>

### 0.1 建议学习路线（约 4 个半天）

| 阶段 | 读什么 | 目标 |
|---|---|---|
| 第 1 段 | §1 全景 + §2 前置知识 | 能用 3 分钟讲清"什么是 interface-induced trajectory censoring"，能手算一个 McNemar *p* |
| 第 2 段 | §3 正文逐节（对照 PDF 看） | 能对每张正文表说出：分母、测哪一层、支持什么结论、不支持什么 |
| 第 3 段 | §4 附录 + §5 数字表 | 能回答审稿人式的追问（"你的分类器准吗？""为什么不显著还要写？""LoRA 和全参能比吗？"） |
| 第 4 段 | §7 误读表 + §8 自测题 | 合上文档能把自测题答出 80% 以上 |

<a id="ch0-2"></a>

### 0.2 读每个数字时问自己的四个问题

这篇论文最大的特点是**每个数字都有严格的口径**。养成习惯，每看到一个数先问：

1. **分母是什么？** 100 个 KodCode 题？200 个 BFCL case？115 个 τ-bench 任务？542 个 EvalPlus 题？还是"生成事件（generation events）"？
2. **它测的是漏斗的哪一层？** emission（模型吐了什么字节）/ parse（解析器接不接受）/ execution（工具有没有真跑）/ observation（结果有没有回到对话里）/ outcome（任务最终成没成）。
3. **它是机制计数（mechanism count）还是任务结果（task outcome）？** 论文反复强调：机制计数大幅变化 ≠ 任务结果显著变化。
4. **它能推出什么、不能推出什么？** 论文几乎每个结果后面都跟一句"This does not establish …"。这些否定句是论文的灵魂，学的时候要和正面结论一样重视。

<a id="ch0-3"></a>

### 0.3 本文档的标记约定

- **【论文原意】**：论文明确说了的。
- **【不能推出】**：论文明确拒绝的推论，最容易被说错。
- **【我的解释】**：为了帮助理解加的背景，不是论文原文。
- `等宽字体`：代码、配置项、文件名。

---

<a id="ch1"></a>

## 1. 一页纸全景

<sub>[↑ 回到目录](#toc) ｜ [← 上一章：0. 怎么用这份文档](#ch0) ｜ [下一章：2. 前置知识 →](#ch2)</sub>

<a id="ch1-1"></a>

### 1.1 一句话主张

> 工具调用率（tool-call rate）测的不是"模型"，而是 **模型 × 序列化格式（template）× 解析器（parser）× 执行环境** 的组合。
> 当模型吐出的字节和接口约定的格式对不上时，服务端会返回 **HTTP 200 + 空的 `tool_calls`**——
> 这和"模型主动不调用工具"在日志上**一模一样**。于是评测分数、协议对比、RL 训练经验都被悄悄扭曲了。
> 论文把这种现象叫 **interface-induced trajectory censoring（接口诱导的轨迹删失）**。

<a id="ch1-2"></a>

### 1.2 漏斗（全文的骨架）

```
 模型权重 + prompt + 解码参数
        │
        ▼
 ① emission       模型产出的原始字节 Y              ← C(Y)：这些字节里"有没有工具意图"（独立分类器判断）
        │
        ▼
 ② serialization  chat template 规定的包装格式       ← 比如 Hermes 要 <tool_call>{...}</tool_call>
        │
        ▼
 ③ parse          服务端 parser 抽出结构化调用 P(Y)   ← 失败时：HTTP 200，tool_calls = []
        │
        ▼
 ④ execution      环境/沙箱真的执行这个调用 E(P(Y))
        │
        ▼
 ⑤ observation    执行结果作为 tool 消息回到对话
        │
        ▼
 ⑥ score / reward 基准打分 / RL 奖励
```

**masked（被遮蔽）的定义**：C(Y)=1（字节里确实有调用意图）但 P(Y)=∅（解析器什么也没抽出来）。

**关键区分**：
- **masking（遮蔽）**：模型已经吐出了动作，接口把它抹掉了。→ 改 parser 就能"找回"，不需要重新生成。
- **constructive suppression（建设性抑制）**：比如 `strict: true` 约束解码，在**生成时**就不允许非法动作出现。→ 这是对生成的干预，改变的是 Y 本身，**不是**"找回被删掉的动作"的证据。

<a id="ch1-3"></a>

### 1.3 四条证据线（对应 Introduction 的四个 bullet）

| # | 证据线 | 核心数字 | 正文位置 |
|---|---|---|---|
| 1 | **外部基准**：BFCL v4 + τ-bench | BFCL 官方分 0.00→0.96（simple_python）、0.00→0.19（multi_turn_base）；template×parser 2×2 = 0/0/0/196；τ-bench 执行 0→636，覆盖 0→103/115 个任务 | §5.1，表 1、表 2 |
| 2 | **规模阶梯**：同一错配接口下，越大的 Coder 吐出越多"看起来像调用"的字节，但服务端始终 0 | Coder tight：0, 4, 21, 36, 80；Qwen3 matched 静默数：1, 1, 2, 0 | §5.2，表 3、表 4 |
| 3 | **Llama 反例**：接口不只会"吃掉"动作，也能"挡住"错误动作 | `strict:true` 让错调用 22→0，final pass 44→61 | §5.3，表 5 |
| 4 | **RL 控制实验**：只换 verl 的 parser 注册 | 执行次数 10→16,844；但 held-out 成功率无检测到的提升（427 vs 430，*p*=0.25；native FC 248 vs 248，*p*=1） | §5.4–5.5，表 6、表 7 |

<a id="ch1-4"></a>

### 1.4 论文**不说**的话（背下来，被追问时最容易踩）

- ❌ "parser 有 bug" → ✅ 全档案 4,254 个去重首轮里，"服务端该接受却没接受"的情况是 **0**。parser 按自己的语法正确地拒绝了；问题在于**契约不匹配**。
- ❌ "两个主效应为零" → ✅ "从文档基线单换任一组件恢复为 0；两个都换恢复 196/200"。主效应各是 +98，**简单效应**（在基线处）才是 0。
- ❌ "修复接口提升了模型性能 / RL 效果" → ✅ 修复恢复了**机制**（解析、执行、观测），但**没有检测到**结果提升。
- ❌ "修复对 RL 没用 / 证明了接口修复无法提升学习" → ✅ "单种子、150 步、7B LoRA 这个设定下未检测到提升"，不是等价性结论。
- ❌ "censoring 随模型规模增长是普遍规律" → ✅ 一个家族（Qwen2.5-Coder）、一个错配接口、一个任务内的观察，不是 scaling law。
- ❌ "80 是下界 / 真实调用数至少 80" → ✅ 4,000 字符截断造成假阴性，分类器精度 0.900 造成假阳性，**净偏差无法识别**。
- ❌ "τ-bench 上修复让成功率提升" → ✅ 7→10，3 个不一致对全朝修复方向，*p*=0.25，**不做结果声明**。

---

<a id="ch2"></a>

## 2. 前置知识

<sub>[↑ 回到目录](#toc) ｜ [← 上一章：1. 一页纸全景](#ch1) ｜ [下一章：3. 正文逐节精读 →](#ch3)</sub>

<a id="ch2-1"></a>

### 2.1 Tool calling 的完整链路

**【我的解释】** 以 OpenAI 兼容 API（vLLM 提供的就是这个）为例：

1. 客户端发请求：
   ```json
   {
     "model": "...",
     "messages": [{"role":"system","content":"..."}, {"role":"user","content":"实现函数 xxx"}],
     "tools": [{"type":"function","function":{"name":"run_tests","parameters":{...}}}],
     "tool_choice": "auto"
   }
   ```
2. 服务端用 **chat template** 把 messages + tools 渲染成一段纯文本 prompt（模型只认 token，不认 JSON 结构）。
3. 模型生成一段文本（这就是 **Y**，原始字节）。
4. 服务端的 **tool parser** 扫描这段文本，尝试抽出工具调用：
   - 抽到了 → 放进 `choices[0].message.tool_calls`，`content` 通常变空或只剩剩余文字；
   - 没抽到 → 整段文本留在 `content`，`tool_calls` 为空。
5. **无论哪种情况都是 HTTP 200**。客户端（基准 harness、RL agent loop）只看 `tool_calls`。

`tool_choice` 的三种常见取值：
- `auto`：模型自己决定调不调（论文主实验用这个）；
- `required`：强制必须调一个工具（论文把它当 **positive control**——先确认这台服务器在"必须调"时能产出正确命名、可解析的调用，再去解释 `auto` 下的 0）；
- `none`：不许调。

**核心洞察**：第 4 步失败时的可观测现象 = "模型选择不调用工具"。两者在 `tool_calls` 字段上不可区分。这就是论文第一段说的 "the same observable as a model declining to act: HTTP 200 and an empty `tool_calls` array"。

<a id="ch2-2"></a>

### 2.2 Chat template

- **是什么**：一段 Jinja2 模板，通常放在模型仓库的 `tokenizer_config.json` 里（字段 `chat_template`），也可以用 vLLM 的 `--chat-template` 覆盖。
- **做什么**：
  1. 把 `tools` 列表渲染进 system prompt（告诉模型有哪些工具、参数是什么）；
  2. 规定模型**应该用什么格式**输出调用（比如 Qwen2.5 系列 template 写的是"用 `<tool_call></tool_call>` 包起来"）；
  3. 把历史中的 tool 调用和 tool 返回渲染回去（多轮时）。
- **陷阱 1：模板继承（论文的 "template check false-positives"）**。Qwen2.5-Coder 的 template 继承自 Qwen2.5 通用版，里面**写着**支持 tools、写着用 `<tool_call>` 包装。所以"检查模板支不支持 tools"这一步会给出"支持"的结论——这是**假阳性**：模板说支持，但 Coder 这个 checkpoint 实际上并不按这个格式输出。见附录 G 的 taxonomy 表："× template check **false-positives**"。
- **陷阱 2：仓库修订依赖**（局限 16）。template 随 `tokenizer_config.json` 发布，发布者可以在不改模型权重的情况下改模板。所以论文强调要 pin 模型 revision 和模板哈希。
- **DeepSeek-Coder 的情况**：它的 template **根本不注入 tools**，模型连有什么工具都不知道。这是 template 层的失败，而且"可以事先检查出来"（taxonomy 表里唯一的 ✓）。

<a id="ch2-3"></a>

### 2.3 vLLM 的 tool parser

- 启动参数：`--enable-auto-tool-choice`（不加的话带 tools 的请求直接 HTTP 400）+ `--tool-call-parser <name>`。
- **hermes parser 的逻辑**（vLLM 0.27.1，论文附录 G 用 `analysis/failure_layer.py` 逐行重放了它）：
  1. 用正则 `<tool_call>(.*?)</tool_call>|<tool_call>(.*)` 找包装层（envelope）；
  2. 对捕获到的内容做 `json.loads`；
  3. 要求 JSON 里有 `name` 和 `arguments`。
- 所以一个输出可能在三个地方失败（论文的 failure-layer 分类，按**第一个失败点**归类）：

| 类别 | 含义 | 典型原因 |
|---|---|---|
| **no envelope** | 连 `<tool_call>` 标签都没有 | Coder 输出裸 JSON 或 ```json 代码块 |
| **payload malformed** | 有标签，但里面的 JSON 坏了 | Instruct 把 Python 代码塞进 `code` 字符串时，docstring 的引号或换行没转义 |
| **strictness only** | JSON 本身没问题，只是捕获段里有尾随文字，`json.loads` 不容忍 | 标签内 JSON 后面还跟了别的字 |
| **parser loss** | 按 parser 自己的规则**应该**接受，服务端却没接受 | ——（全档案 0 例） |

  **重要细节**：`classify()` 是"首个失败点"分类，遇到 no envelope 就直接返回，不再检查 payload。所以"parser loss = 0"这个结论只对**带了 envelope 的输出**才是真正的检验。对 Coder 来说（100% no envelope），payload 层从没被测试过。——这就是为什么 τ-bench 上"789 个未解析回合全部是 missing-envelope、没有一个到达 payload 验证阶段"，**不能**说成"0 个 payload 是坏的"（我们早期犯过这个错，被 Codex 抓出来了）。

- 其他 parser：
  - `llama3_json`：Llama 3.1/3.2 用，配 `tool_chat_template_llama3.1_json.jinja`（"official template"）。
  - `mistral`：要求 `tool_call_id` 恰好 9 个字符；模型重复输出 `[TOOL_CALLS]` token 会导致 HTTP 400。官方推荐的 parallel template 反而把错误率从 2% 抬到 42%。
- **verl 有自己的 parser**（RL 部分的关键）：verl 的 AgentLoop 在 `verl/experimental/agent_loop/tool_parser.py` 里有自己的 `ToolParser` 注册表（hermes 那个是从 vLLM v0.9.1 移植的），由配置 `multi_turn.format` 选择。**给 vLLM 服务端装 parser 插件对训练链路无效**。我们必须在 verl 内部注册 `qwen2_5_coder` parser（代码：仓库根目录 `qwen_tools_parser.py`，接受 `<tools>` 标签、```json 代码块、裸 JSON 三种格式）。

<a id="ch2-4"></a>

### 2.4 Qwen2.5-Coder 到底输出了什么，适配器做了什么

- **事实**：Qwen2.5-Coder 在 Hermes 配置下输出的是**裸 JSON** 或 **```json 代码块**，不带 `<tool_call>` 包装。
  例（示意）：
  ````
  ```json
  {"name": "run_tests", "arguments": {"code": "def foo(x):\n    return x+1"}}
  ```
  ````
  hermes parser 找不到 `<tool_call>` → no envelope → `tool_calls=[]`。
- **"documented" 配置**：vLLM 0.27.1 文档"Qwen Models"一节写 `Qwen/Qwen2.5-*` 用 `hermes`。我们把这个**家族级**指引套到 Coder 上。论文明确说："documented" 指的是**配置选择的来源**，不是"官方保证 Coder 兼容"，也不是"部署中普遍这样用"的证据（附录 D）。
- **这个不兼容不是我们发现的**：vLLM issue #32926 早就报告过。论文的贡献是**量化**它作为评测效度问题、**分解**原因、并跟进到交互评测和 RL。
- **dedicated adapter（专用适配器）**：社区项目 `hanXen/vllm-qwen2.5-coder-tool-parser`（commit `1b92150`，Apache 2.0），包含两部分：
  1. **parser**：接受 `<tools>…</tools>` 标签；
  2. **chat template**：带 few-shot 示例，诱导模型输出 `<tools>` 标签。
  我们**改了 template 里的 few-shot**：原版示例调用 `get_weather`（我们 schema 里没有的工具），不改会诱导模型调用不存在的工具，同时虚增"发起率"和"空参数"统计。原版和修改版的哈希都记录了。
- **为什么必须两个一起换**（2×2 的直觉）：
  - 只换 parser（接受 `<tools>`），但模板没教模型输出 `<tools>` → 模型还是吐 ```json → 0；
  - 只换 template（教模型输出 `<tools>`），但 parser 还是 hermes 只认 `<tool_call>` → 0；
  - 两个都换 → 196/200。
  这叫**强互补（strong complementarity）**：每个组件都按自己的契约正常工作，但两个契约不一致。

<a id="ch2-5"></a>

### 2.5 `strict: true` 与约束解码

- **【我的解释】** 在 tool schema 里加 `"strict": true`（以及 `additionalProperties: false`），并开启服务端的 strict tool calling，服务端会用**结构化输出 / 引导解码（guided decoding）**：生成每个 token 时屏蔽掉会违反 schema 的 token。于是模型**不可能**生成一个名字不在工具列表里的调用、或者参数名不对的调用。
- **在论文里的作用**：Llama-3.1-8B 有 23% 的题会把"题目要求实现的函数"（比如 `can_form_word`）当成工具来调用，参数是那个函数的参数（`{"tiles": ..., "word": ...}`），而不是调用 `run_tests(code=...)`。`strict:true` 让这种错调用从 22 → 0。
- **为什么这是"constructive suppression"而不是"masking 修复"**：它改变的是**生成**（turn-1 输出变了：turn-1 pass 33→46），不是对已有字节重新解析。论文 §3 的第一个诊断签名专门讲了这个区别。
- **verl 0.9.0 的 rollout 路径不暴露约束解码**（它唯一的 `strict` key 属于 profiler），所以 RL 实验没法用这招（附录 D (iv)）。
- **相关文献**：Tam et al. 2024 "Speak Freely" 发现格式约束会伤害推理；我们的 Llama 是一个互补的案例——约束移除了角色混淆、改善了执行。

<a id="ch2-6"></a>

### 2.6 ReAct vs Function Calling (FC)

- **FC**：用 API 的结构化 tools 通道，依赖 template + parser。
- **ReAct**：模型在纯文本里写 `Thought: ... Action: run_tests ... Action Input: ...`，由 **harness 自己的正则**来抽取代码，不经过服务端 parser。所以 ReAct 绕开了 FC 的接口问题——在论文里它是**参照臂（reference arm）**。
- 两种 prompt 强度：
  - `FC_OPTIONAL`："You **may** call run_tests…"
  - `FC_MANDATORY`："You **must** first call run_tests… do not answer directly."
- **协议差距（protocol gap）**：ReAct 和 FC 的 final pass 差。论文在 Llama 上把 terse-FC→ReAct 的 +31 分解成 interface repair +12（49→61）和 protocol +19（61→80）。注意：这两个分解都是**描述性的**，都没过 Bonferroni。

<a id="ch2-7"></a>

### 2.7 多轮代码修复循环里的指标

这些指标在全文反复出现，必须分清：

| 指标 | 定义 |
|---|---|
| **turn-1 pass** | 第一轮提交的代码就通过 held-out 测试 |
| **final pass** | 对话结束时最终代码通过 |
| **rescue（救回）** | final pass 但 turn-1 没过——即"靠后续轮次的反馈修好了"。`rescue = final − turn-1`（因为论文观察到修复循环是单调的：turn-1 过了的 final 也过，见附录 G 方差一节"passed at turn 1 but failed finally = 0"） |
| **tested step** | 一次"抽出了代码并送进沙箱测试"的生成。由 `n_turns` 计数器记录，**只有抽到代码时才 +1** |
| **≥2 tested steps** | 至少走到了第二个有代码、被测试的步骤。比"收到过反馈"更严格：如果收到反馈后的回复抽不出代码，计数器不动 |
| **initiation（发起）** | 至少一次工具调用。论文后期定义更严格：**被接受并执行**的调用，而不是所有"像调用的输出" |

**为什么 FC+Hermes 的 turn-1 = final = 53、rescue = 0**：工具通道断了，模型永远拿不到测试反馈，只能一次性交卷。
**为什么 Hermes 臂的平均 tested steps 是 0.96 而不是 0**：KodCode 探针的评测器在 FC 没解析出调用时，仍会从 `content` 里抽代码做最终评分；100 个首轮输出中 4 个抽不出代码，所以 96/100。

<a id="ch2-8"></a>

### 2.8 基准

#### KodCode（训练数据 + 受控探针）
- 合成的 Python 编程题，每题自带 pytest 测试。
- 论文的**受控探针**：100 题，`clean[:100]`（去污染后列表的前 100 个，**不是随机抽样**——局限 2）。所有 50 个完整臂用的是同一个 100 题集合（"unique item-set count = 1"），所以臂间比较是**严格配对**的。
- 随机复制：随机抽 300 题（seed 20260901），用来检验 `clean[:100]` 是否有偏。7B 那一档从 21 变成了 30，说明单个 100 题子集对中间档不是稳定估计，但方向和单调性不变。
- **去污染（decontamination）**：对 EvalPlus 做 n-gram containment。题面 n-gram 8/5，解答 n-gram 10/6，最小 5-gram；containment = 交集 / 较小集合；阈值 0.10。去污染后池子 7,669 题。**不排除语义重叠或预训练污染**。
- 一个工具：`run_tests(code: str)`，沙箱里跑 pytest。

#### EvalPlus（RL 的 held-out 评测）
- HumanEval+ 164 + MBPP+ 378 = **542**。
- 两个已知有缺陷的题：`HumanEval/32`、`Mbpp/599`。历史 1.5B 曲线排除它们用 n=540；正式 7B 预注册评测保留全部 542（两个题在两臂都失败）；native FC 评测中 `MBPP/599` 对两个训练后权重**通过**，所以附录给了 540 题的敏感性分析。

#### BFCL v4（Berkeley Function Calling Leaderboard）
- 固定在 Gorilla commit `6ea5797`。
- `simple_python`：单轮，给一个函数定义，要求生成一个调用。**AST checker** 打分：把调用解析成语法树，比对函数名、参数名、参数类型/取值是否在答案允许范围内。
- `multi_turn_base`：多轮，文件系统 / 车辆控制 / 交易等 API，按**执行后的状态**和响应打分（state-based）。
- **分数 = accuracy = correct_count / total_count**。所以 0.96 = 96/100，0.19 = 19/100（我们从 BFCL 分数文件核对过）。对外引用时要写清楚这是"官方 scorer 下的 accuracy"。
- 我们做了：不改 `bfcl_eval` 自身文件（在 import 时在内存里注册 OpenAI handler），中间加 recording proxy 记录每一个 HTTP 请求；`max_model_len=16384`、`temperature=0`、推理 seed 0、manifest seed 20260901。

#### τ-bench retail
- 客服场景：模型扮演零售客服，通过工具（查订单、改地址、退货……）操作数据库状态；一个**用户模拟器**扮演顾客。
- 打分看**最终环境状态**是否和目标一致。
- 我们用全部 115 个 retail 任务；**用户模拟器是本地 Llama-3.1-8B，不是原 harness 的 gpt-4o**。所以绝对分数**不能和排行榜比**，只声明配对臂差异。
- 意义：把 envelope 层机制扩展到**非代码、有状态**的环境。但规模和 RL 结论仍是代码任务专属。

<a id="ch2-9"></a>

### 2.9 RL：GRPO、LoRA、verl

#### Policy gradient 的直觉
**【我的解释】** RL 只能从**采样到的经验**里学习。策略梯度 ≈ Σ advantage × ∇log π(动作)。如果某种动作（"调用工具→看到测试结果→修代码"）**从未出现在采样轨迹里**，梯度就无从强化它。parser 把调用吃掉，等于把这类轨迹从经验分布里删掉了——这就是论文第三个诊断签名："in RL the parser changes the sampled experience distribution **before** a policy-gradient estimator sees it"。

但要注意论文的克制：恢复经验 = 恢复了"**接触**（access）"工具经验的机会；**不等于**测出了梯度贡献，也不等于最终学会了。

#### GRPO（Group Relative Policy Optimization，DeepSeekMath 提出）
- 对每个 prompt 采样一组（group）G 条回答（我们 `rollout.n = 8`）。
- 每条的 advantage = (r_i − mean(r)) / std(r)，**组内相对**。不需要 value network（critic）。
- 推论：如果一组 8 条奖励完全一样（全对或全错），advantage 全为 0，这个 prompt 不贡献梯度。
- 我们的规模：`train_batch_size 16 × rollout.n 8 = 128` 条轨迹 / step，150 步。
- **RLOO**（REINFORCE Leave-One-Out）：baseline 用组内其他样本的均值。历史 1.5B 跑了 GRPO×2 种子 + RLOO×1 种子。

#### LoRA
- 冻结原权重 W，学低秩增量 ΔW = B·A（A: r×d, B: d×r），实际用 W + (α/r)·BA。
- 我们：rank r = 32，alpha α = 32 → scaling α/r = **1**；作用于**所有线性层**（all-linear）。
- 规模：392 个 LoRA 张量，80,740,352 个参数。
- **为什么用 LoRA**：7B 全参 GRPO 在单张 80GB 卡上 OOM。两臂用同一个 LoRA 配置 → **保留了臂间对比**，但**不能**和历史 1.5B **全参**曲线严格比较。

#### verl（HybridFlow）
- 字节开源的 RL 框架。我们用 verl 0.9.0，rollout 后端是 vLLM 0.27.1。
- **AgentLoop**：多轮 agent rollout 的实现。`tool_agent`（FC）和 `react_agent`（ReAct）两种。
- **ToolParser 注册表**：`multi_turn.format=hermes` 选 verl 自带的 hermes parser。正式实验唯一的臂间差异：`multi_turn.format=hermes`（broken）vs 注册的 `qwen2_5_coder`（repaired）。
- **instrumentation 陷阱**：
  - verl 的 step metrics **没有**工具调用计数。名叫 `timing_s/agent_loop/tool_calls` 的指标是**秒数**，不是次数。
  - 计数器要放在 `CodeTool.execute`（FC 路径，`code_tool.py`）和 `ReActAgentLoop._run_tests`（ReAct 路径，`react_agent_loop.py`），**不能**放在 `sandbox.run_tests`——因为 reward 函数评估最终程序也走这个函数，会把"奖励评估"和"agent 调用工具"混在一起。我们踩过这个坑，后来把沙箱自己的计数器改名为 `sandbox_exec_count`。

#### Reward（`reward_code.py`）
- outcome-only：取最后提交的程序，跑该题自己的 pytest，返回**通过比例**。
- 三条 reward hacking 路径被构造性地堵上：没有任何测试执行 → 0 分（而不是 1）；超时 → 0；题目没有测试字符串 → 0。
- **历史 verifier 的两个缝隙**：(i) 在合并的 stdout/stderr 上用未锚定的 `(\d+) passed` 取第一个匹配 → 模型可以 `print("5 passed")` 伪造；(ii) `skipped`/`xfail` 不计入分母。审计：5 个保留了生成代码的训练日志里 0 次出现这类签名，`critic/rewards/mean` 在 0.46–0.73 间振荡、没有"发现漏洞"时那种阶跃。**正式 7B 实验前修好了**：正则锚定到 pytest 的 summary 行，分母包含 skipped/xfailed/xpassed，每条 rollout 全文保留 + SHA-256。两臂用同一个修好的 verifier → 它不能解释两臂差异。
- held-out 通过率**不用** reward verifier，用 EvalPlus `mode="script"`，要求 exit code 0 且 `status=="ok"`。

<a id="ch2-10"></a>

### 2.10 统计

#### (a) 配对设计 & 精确 McNemar 检验
- 同一批题目在两个臂上各跑一次，每题结果是 (A 过?, B 过?)。四种情况：都过、都不过、只有 B 过（记 b，"gain"）、只有 A 过（记 c，"loss"）。
- **只有不一致对（discordant pairs）携带差异信息**。在 H0（两臂没差别）下，每个不一致对等概率落在 b 或 c：b ~ Binomial(b+c, 0.5)。
- **精确双侧 p** = min(1, 2 × P(X ≤ min(b,c)))，X ~ Bin(b+c, 0.5)。
- **手算练习**（都能和论文对上）：

| 对比 | b/c | 计算 | *p* |
|---|---|---|---|
| Llama official→strict 错调用 | 0/22（或写作 22/0） | 2 × 0.5²² = 2⁻²¹ | **4.77×10⁻⁷** ✓ |
| P3 broken→repaired step150 final | 0/3 | 2 × 0.5³ | **0.25** ✓ |
| τ-bench documented→repaired | 3/0 | 2 × 0.5³ | **0.25** ✓ |
| P3 repaired step0→150 | 1/4 | 2 × (1+5)/32 | **0.375** ✓ |
| Llama terse→strict final（T12） | 15/3 | 2 × (1+18+153+816)/2¹⁸ = 2×988/262144 | **0.00754** ✓ |
| Llama official→strict final（T15） | 18/1 | 2 × (1+19)/2¹⁹ | **7.63×10⁻⁵** ✓ |
| native FC repaired vs broken | 16/16 | min=16=n/2 | **1** ✓ |

- **为什么不能只比总数**：430 vs 427 看起来"差 3 个"，但关键是这 3 个是 0/3 的不一致对。如果是 50/53 的不一致对，总数差也是 3，但 *p* 完全不同。所以论文**总是同时报两个不一致计数**。
- **"Equal totals do not imply equal passing sets"**：16/16 的总数相同，但有 32 个题换了归属。同理附录 G 的方差实验：ReAct final 72/72/72，但解集 Jaccard 只有 0.71–0.80。

#### (b) 多重比较 & Bonferroni
- 做 m 个检验，每个用 α=0.05，至少一个假阳性的概率远大于 0.05。Bonferroni：每个检验用 α/m。
- 论文的**回顾性报告族**：27 个检验（5 个 Llama 对照 × 3 个指标 = 15，加 12 个其他对比），阈值 **0.05/27 = 0.00185185**。调整后 *p* = min(1, 27p)。
- **关键措辞**：这个族是**回顾性的报告审计（retrospective reporting audit）**，**不是预注册的多重性策略**。论文明说 "Neither policy was preregistered"。
- 擦边案例：Llama strict→ReAct（T16）*p* = 0.00188，**刚好**没过 0.00185。
- native FC 后加的 9 个对比（3 个权重对比 × final/initiation/rescue）是**另一个**探索性族，单独报告；即使 36 个合并修正，新结果也没有一个显著（而且它们连名义显著都没有）。

#### (c) 保守的配对风险差区间（RL 终点）
- 目标量：Δ = p₊ − p₋，其中 p₊ = "只在 repaired 过"的概率，p₋ = "只在 broken 过"的概率。
- 做法：对 p₊ 和 p₋ 各自算一个 **97.5% 双侧 Clopper–Pearson 区间** [L₊,U₊]、[L₋,U₋]；用 **union bound（并集界）**：两个区间同时覆盖的概率 ≥ 1 − 2.5% − 2.5% = 95%，于是 [L₊−U₋, U₊−L₋] 至少有 95% 覆盖率。
- **为什么不用正态近似**：只有 3 个不一致对，正态近似完全不可靠。
- 数值：(n₊,n₋,n) = (0,3,542) → 点估计 −0.55pp，区间 **[−1.79, 0.72]** pp。
  **【我的解释】** 0/542 的 97.5% CP 上界 = 1 − 0.0125^(1/542) ≈ 0.805%；由论文数字反推，3/542 的 97.5% CP 区间约 [0.085%, 1.79%]。于是下界 0 − 1.79 = −1.79，上界 0.805 − 0.085 ≈ 0.72。✓
- repaired 自己 step0 vs 150：(1,4,542) → [−2.06, 1.01]。
- native FC repaired vs broken：16/16 → [−3.47, 3.47]。
- **这个区间覆盖什么、不覆盖什么**：只覆盖**题目配对的抽样不确定性**（假设题目独立）；**不**覆盖训练种子、推理复测、基准选择、任务簇的不确定性。而且是**逐点**区间，不是对整个检验族的同时区间。

#### (d) Clopper–Pearson 区间
- 二项比例的"精确"区间，用 Beta 分布分位数反解：下界 = Beta(α/2; x, n−x+1)，上界 = Beta(1−α/2; x+1, n−x)。保证覆盖率 ≥ 名义水平（偏保守）。
- **"三分法则"（rule of three）**：观察到 0/n 时，95% 上界 ≈ 3/n。这和论文局限 10 "near-zero observed frequency is not zero probability" 是同一个道理：broken 臂 10 次执行不是 0，0 次也不等于概率为 0。

#### (e) Cohen's κ
- κ = (p_o − p_e) / (1 − p_e)：观测一致率扣除"随机一致"的期望后的一致度。
- 经验解读：0.61–0.80 substantial，0.81–1.00 almost perfect。
- 论文：A1 vs A2 全部 97 题 κ=0.736；**同字节**的 62 题 κ=0.967；分类器 vs 金标准 κ=0.871。
- 置信区间用 **bootstrap**：5,000 次重抽样，seed 7，按题目 ID 排序（之前区间不一致是因为 bootstrap 输入顺序不一致，不是标注者不同）。

#### (f) 精度、召回 & 分层抽样
- precision = TP/(TP+FP)，recall = TP/(TP+FN)。
- 论文的 98 题样本是**按分类器判决分层**的（40 tight、28 strong-but-not-tight、30 neither），故意过采决策边界。所以**未加权的样本指标不是总体精度/召回**。
- tight 层精度 36/40 = 0.900。

#### (g) 析因设计（2×2）：主效应 vs 简单效应 vs 交互（**必须彻底搞懂**）

BFCL 2×2 表（单元格 = 首请求解析数 / 200）：

|  | documented template | dedicated template | 行均值 |
|---|---|---|---|
| **Hermes parser** | 0 | 0 | 0 |
| **Dedicated parser** | 0 | **196** | 98 |
| 列均值 | 0 | 98 | |

- **parser 的主效应**（marginal / main effect）= 在 template 两个水平上**平均**的效应 = 98 − 0 = **+98**。
- **template 的主效应** = 98 − 0 = **+98**。
- **parser 的简单效应（simple effect）**，在 documented template 处 = 0 − 0 = **0**；在 dedicated template 处 = 196 − 0 = 196。
- **template 的简单效应**，在 Hermes parser 处 = 0 − 0 = **0**；在 dedicated parser 处 = 196。
- **交互效应**（difference of differences）= (196 − 0) − (0 − 0) = **196**。
- 所以正确说法：**"在文档基线处，两个单因素简单效应都是 0；联合替换显示强互补"**。说"主效应为零"是**统计上错误**的（我们早期在 arXiv v1 里犯过）。
- 另一个口径细节：这个 2×2 是**解析计数**。官方 BFCL 分数只对两个**角上的端到端配置**（documented/documented 和 dedicated/dedicated）算了，不是四个格子都算了。

#### (h) 后处理条件化（post-treatment conditioning）
- **【我的解释】** "是否解析成功"是**被处理（接口）影响的变量**。在"两臂都解析成功"的子集上比较，等于按处理后的结果选样本——选出来的不是随机子样本，比较结果不是干净的因果估计（类似 collider bias / 选择偏差）。
- 论文：Qwen 两臂都解析的 83 题上 ReAct 63 vs FC 56，残差 +8.4pp，95% CI [−0.5, 17.4]，*p*=0.118 → "**我们没检测到残差，这不等于证明没有残差**"。Llama 96 题上 78 vs 60，*p*=0.0021（也没过 0.00185）。两者都只是**描述性**的。
- 同理附录 G 的"given feedback 的修复成功率"（ReAct 42.5% vs adapter 24.3%）是处理后选样，**不能**定位或因果分解协议差距。

#### (i) 预注册（preregistration）
- 实验前写下假设、设计、分析方法，防止事后挑结果。
- 论文里**预注册**的：Qwen3 matched ladder（预测 "parsed > 0 at every size" 在推理前提交）；正式 7B RL 控制（训练预注册）。
- **偏离也要记录**：Qwen3 默认 thinking 模式在调用前就耗尽了 token 预算，所以关掉了 thinking；Qwen3 parsed 不单调。两者都被记为 protocol deviation。
- **非预注册的**：27-test Bonferroni 族（回顾性）；native FC 评测（审稿后设计，在训练预注册之外，"post-hoc"）。

#### (j) 机制计数 vs 任务结果
论文 §4 原话的精神："a large deterministic movement in the former does not license a performance claim when the paired outcome test is null"。
- 机制计数：解析数、执行数、观测数——接口一换，这些会**确定性地**大幅变化（0→196，10→16,844）。
- 任务结果：pass/fail、成功率——需要配对检验。
- 两者是不同的问题。**恢复机制 ≠ 提升结果**。

---

<a id="ch3"></a>

## 3. 正文逐节精读

<sub>[↑ 回到目录](#toc) ｜ [← 上一章：2. 前置知识](#ch2) ｜ [下一章：4. 附录逐个精读 →](#ch4)</sub>

<a id="ch3-1"></a>

### 3.0 标题与摘要

**标题**：*Interface-Induced Trajectory Censoring: When Tool-Call Interfaces Distort Evaluation and RL Experience*
- "Interface-Induced"：原因在接口，不在模型。
- "Trajectory"：被删的不只是一个调用，而是整条轨迹的后续（执行、观测、后续轮次、修复）。
- "Censoring"：**【我的解释】** 借了统计里"删失"的概念——数据存在，但观测机制让你看不到。论文后来明确：censoring 在本文的意思是"**the attempt never reaches the parser in a form it is defined to accept**"，**不是** "the parser discards valid calls"。
- "Evaluation and RL Experience"：两个受影响的下游——评测分数和训练经验。注意是 "RL **experience**"，不是 "RL **performance**"。

**摘要逐句拆解**（摘要 1,743 字符、纯 ASCII，符合 arXiv ≤1920 的限制）：

1. *Tool-use evaluations typically measure calls through a serving stack, which can report zero calls even when model outputs contain call-shaped payloads.* → 问题陈述。注意用词 "call-shaped payloads"（看起来像调用的内容），不说"有效调用"。
2. *We study interface-induced trajectory censoring: a mismatch between model emissions and the interface that converts them into actions.* → 定义。重点：mismatch 在**模型输出**和**接口**之间。
3. *On selected BFCL v4 subsets, changing only the serving adapter while holding model weights and other evaluation settings fixed moves scores under the official scorer from 0.00 to 0.96 on simple_python and from 0.00 to 0.19 on multi_turn_base.* → 头条数字。"selected subsets" 很重要：只跑了两个子集，不是 BFCL 全榜。
4. *A template–parser factorial experiment shows strong complementarity: neither component replacement alone restores parsing from the baseline, whereas their joint replacement recovers 196/200 first-request parses.* → 2×2。措辞是 "from the baseline"——即简单效应。
5. *Across five Qwen2.5-Coder checkpoints, classifier-positive emissions increase with model size while the mismatched interface reports no calls; a separate matched-interface Qwen3 ladder exhibits few silent emissions through 8B.* → 规模。"classifier-positive" 而不是 "true calls"。
6. *Experiments on 115 τ-bench retail tasks extend the mechanism to a stateful, non-code environment.* → 扩展的是**机制**，不是结果。
7. *In a preregistered, single-seed, 150-step 7B LoRA experiment, changing only the parser registered in verl increases tool executions from 10 to 16,844.* → RL 机制恢复。限定词一个不少：preregistered、single-seed、150-step、7B、LoRA。
8. *However, neither a shared programmatic-feedback evaluation nor a post-hoc common native function-calling evaluation detects a held-out success gain from repair; rescue counts do not increase.* → 关键的零结果。"detects" 是刻意的用词：未检测到 ≠ 不存在。
9. *Interface repair thus restores tool-mediated experience without a detected performance gain in this setting.* → 总结，"in this setting" 限定范围。
10. *These findings show that tool-use measurements depend on the model–interface configuration and motivate layer-wise observability and conformance checks before attributing evaluation or training outcomes to the model alone.* → 实践建议。

<a id="ch3-2"></a>

### 3.1 §1 Introduction

**第 1 段**：
- 链条：emission → serialization → parse → execution → observation → score/reward。
- 关键句："A failure at several of these boundaries has the same observable as a model declining to act: HTTP 200 and an empty `tool_calls` array."
- 定义了 masking vs constructive suppression（见 1.2）。

**图 1（fig0_funnel）**：每臂 n=100 的工具使用里程碑。
- 第四根柱子是"至少两个有代码、被测试的步骤"的任务数（repaired 37，ReAct 40），**不是**"返回的观测数"。
- N/A 的含义：服务端一旦成功解析，调用就被挪进 `tool_calls`、`content` 变空——此时没法再从 content 读原始字节做 emission 分类，所以填 N/A 而不是 0。

**第 2 段：历史 1.5B 的角色**
- "Historical 1.5B runs motivated the investigation, but their ambiguous zero-execution result does not identify parser masking."
- 【我的解释】这是论文的诚实之处：最早发现问题的那批 1.5B 实验，1.5B 模型压根没产出正确命名的调用（见附录 G 的 probe 表：52 个可恢复调用里 0 个叫 `run_tests`），所以"0 次执行"是**过度决定的（over-determined）**——既可能是 parser 问题，也可能是模型本身就没调对。所以它只作动机，不作因果证据。

**第 3 段：与 vLLM issue #32926 的关系**
- 不兼容早已被报告。贡献 = 量化为评测效度问题 + 分解原因 + 跟进到交互评测和 RL。

**四个贡献 bullet**（见 1.3）。注意第 2 条的措辞："a **preregistered** matched-interface ladder keeps the silent count at 0–2 per 100"；第 3 条："the **opposite** interface effect"（接口不是只会坏事，也能挡错）。

**最后一段：主张比"the stack matters"更窄**
- "A serving interface is part of both the benchmark's **measurement function** and an agent's **effective action space**."
- 这句话是全文最值得背的理论表述：接口同时是(1)测量仪器的一部分、(2)agent 实际动作空间的一部分。
- 所以："tool-use results require layer-wise observability and a conformance check before model or protocol attribution."

<a id="ch3-3"></a>

### 3.2 §2 Related Work

| 引用 | 它是什么 | 论文怎么定位自己 |
|---|---|---|
| Sclar 2024 (format spread)、Mizrahi 2024、Dehghani 2021 (benchmark lottery) | 评测结果随 prompt 格式、基准选择等"讨厌因素"变化 | 他们改**输入**；我们研究**测量仪器内部**的一个组件 |
| Bhat 2026、Raj 2026 (model or harness)、Soni 2026 (ToolFailBench)、Albayaydh 2026 | 工具使用审计：评测器缺陷、失败归因模糊 | 他们大多在**服务边界之上**看 trace |
| **Garware 2026 (broken ruler)** | 安全日志研究：对**同一批字节**重新解析，准确率 0% → 76% | **最接近的类比**。我们把这个结构扩展到标准 agent 基准、下游执行和 RL 经验 |
| Yao 2026 (Harness-Bench) | 主张报告 model–harness 配置，基于跨 harness 工作流测量 | 我们用**组件级干预**隔离一个契约，并跟进到训练 |
| Chen 2026 (Agent-Reactive Bugs) | 从 issue 报告里发现 model–harness 边界的静默失败 | 我们是**受控干预**，不是 bug 报告分类学 |
| BFCL、τ-bench、ToolLLM、API-Bank | 工具基准 | 评测背景 |
| verl (HybridFlow)、GRPO (DeepSeekMath)、LoRA | 训练栈 | — |
| Tam 2024 (Speak Freely) | 约束生成会伤害推理 | 我们的 Llama 是**互补**案例：约束移除角色混淆、改善执行 |

<a id="ch3-4"></a>

### 3.3 §3 Framework and Testable Signatures（**理论核心，逐句理解**）

**记号**：
- **Y**：固定权重、prompt、解码参数下产生的 assistant 字节。
- **C(Y)**：一个**独立的**指示函数，判断这些字节是否包含"工具意图的动作"。（实现上就是 tight 分类器 / BFCL 的 JSON-call 指示器 / τ-bench 的 call-shaped 指示器。）
- **P(Y)**：配置的 parser 返回的结构化动作。
- **E(P(Y))**：环境产生的执行和观测。
- 基准通常记录的是 P(Y) 或 E(P(Y)) 的函数，**而不是 Y**。

**masked**：C(Y)=1 但 P(Y)=∅。

**这个定义的两个好处**：
1. **避开了形而上学的"真实能力"**。我们不需要说"模型其实会调用工具"，只需要说一个可直接观察的事实："一个行为存在于原始生成中，却不在结构化轨迹中"。
2. **不暗示 parser 有 bug**。parser 可能**正确地**拒绝了不符合它语法的字节；无效测量的来源是：把 checkpoint、template、parser 当成一个兼容的整体，却没测试它们之间的契约。

**三个诊断签名**（要能解释每一个为什么成立）：

**签名 1：parser-only 干预不能改变已生成的字节，但能改变执行、观测、后续轮次和 rescue。**
- 推论：如果一个干预**只**换 parser，那么 turn-1 的输出字节应该不变（turn-1 pass 不变），但后续指标会变。
- 论文的诚实之处：我们的评测时 Qwen 修复同时换了 parser + template + few-shot，所以它的 turn-1 53 vs 53 相等**只是一致性检查（consistency check），不是层识别（layer identification）**——因为 template 也变了，理论上 turn-1 字节可以变。
- 真正定位失败层用的是：**fixed-output replay**（固定字节重放）和 **event tracing**（事件追踪）。
- 对比：Llama 的 `strict:true` 作用在**更早**的生成阶段，所以**能**改变 turn-1 输出（33→46）。

**签名 2：C(Y) − 𝟏[P(Y)≠∅] 随规模的趋势，需要"变化的 emission"和"固定的拒绝边界"同时存在。**
- 即：未解析但有意图的数量随规模增长，前提是 (a) 模型越大越会吐调用形状的字节，(b) parser 的拒绝规则不变。
- Qwen3 matched ladder（envelope 匹配）在可比的规模跨度上**没有**同样的增长。
- 但论文立刻限定：这个跨家族观察**不能把家族和 envelope 分开**（Qwen3 家族和匹配 envelope 是一起变的），**不是严格的规模反事实**。

**签名 3：在 RL 中，parser 在策略梯度估计器看到经验**之前**就改变了采样经验分布。**
- 恢复动作和观测 = 建立了对工具介导经验的**接触（access）**，不是测出了**梯度贡献**或**最终学习**。
- 正式 7B 控制**分开检验两个主张**：事件计数识别 experience access；held-out checkpoints 检验这种 access 是否改变行为。

<a id="ch3-5"></a>

### 3.4 §4 Measurement Setup（每个设计选择都有理由）

**受控探针**：
- 100 个 KodCode 题，对 EvalPlus 去污染（normalized text/code n-gram containment），臂间配对，一个 `run_tests(code)` 工具。
- FC：`tool_choice: auto`，temperature 0，2,048 输出 token。
- BFCL v4 @ Gorilla `6ea5797`；τ-bench 全部 115 个 retail 任务（本地 Llama-3.1-8B 用户模拟器 → 不可与排行榜比，只声明配对差异）。

**服务**：vLLM 0.27.1。"Server-parsed calls are insufficient for the question under study"——因为我们要问的恰恰是"服务端没解析出来的那些"。

**tight 分类器**（C(Y) 在 KodCode 上的实现）：
```python
TIGHT = re.compile(
    r'"name"\s*:\s*"run_tests".{0,200}?"arguments"\s*:\s*\{'
    r'.{0,80}?"code"\s*:\s*"(.{0,4000}?)"\s*\}', re.S)
REAL = re.compile(r'\\n|def |class |return |import |lambda ')

def is_tight_call(raw):
    m = TIGHT.search(raw or "")
    return bool(m and REAL.search(m.group(1)))
```
- 逐部分理解：
  - `"name"\s*:\s*"run_tests"`：必须点名 `run_tests`；
  - `.{0,200}?"arguments"\s*:\s*\{`：200 字符内出现 `"arguments": {`；
  - `.{0,80}?"code"\s*:\s*"(.{0,4000}?)"\s*\}`：80 字符内出现 `"code": "…"`，捕获最多 4000 字符的代码，后面跟闭合 `}`；
  - `REAL`：捕获的代码里必须有 Python 词法标记（`\n`、`def `、`class `、`return `、`import `、`lambda `）——用来排除 schema 回显和占位符（比如 `"code": "string"`）。
- **它不做的事**：不验证 JSON 语法、不验证 Python 可执行性、不判断语义意图。要求 `name` 在 `arguments` 之前，所以合法的其他键顺序可能漏掉。
- **验证结果**：对第三方裁决金标准 κ=0.871；tight 层精度 36/40=0.900；两位标注者在 62 个同字节题上 κ=0.967。

**4,000 字符截断**：
- 存储的评测输出最多 4,000 字符 → 截断后才出现的调用看不到 → **假阴性**；
- 精度 0.900 → **假阳性**；
- 两个方向相反 → **净偏差无法识别**。所以 80 不是下界。
- 分层验证样本里各规模的截断命中：0/6, 0/13, 0/15, 0/24, **10/40**（1.5B→32B）→ 截断不确定性集中在头条那一档（32B）。
- 全部 100 个存储首轮输出里的截断命中：0, 0, 1, 0, **18**。截断命中不会告诉你被截掉的后续是什么。
- 正式 RL rollout 全文保留 + 哈希（没有这个问题）。

**positive control vs admissibility gate**（两个不同的东西，别混）：
- **positive control**：解释一个 0 之前，先在同一台在线服务器上验证 `tool_choice: required` 能产生正确命名、可解析的调用（在这个控制可用的地方）。论文提醒：不假设 required 模式和 auto 模式的解析行为相同。
- **admissibility gate（零错误准入门）**：一个臂要进入 clean-arm pass-rate 分析，必须 100 题齐全、exit code 0、**零请求错误**、provenance 匹配。**在代码里强制执行**（`validate_arms.py` 一类的脚本）。
- 注意：auto 模式下 0 解析**不是**排除标准（否则就把要研究的现象排除掉了）。
- 端到端分析单独报告，用完整任务分母，包括请求失败。
- 这个规则逼我们撤回了一个早期无效的 *p*=0.648（附录 A）。

**统计**：精确双侧 McNemar + 报告两个不一致计数；RL 终点用保守有限样本 95% 配对风险区间；27-test 回顾族 + 9 个 native FC 探索族；**两个策略都不是预注册的**。

**打分来源**：BFCL 官方 scorer；τ-bench 最终环境状态；代码任务执行生成程序跑 held-out 测试。训练事件排除评测请求。

**两个缺陷题的处理**（见 2.8 EvalPlus）。

<a id="ch3-6"></a>

### 3.5 §5.1 External validity: BFCL and τ-bench

**设置**：Qwen2.5-Coder-7B-Instruct，两个外部基准都用它；BFCL 上固定权重、case、解码、seed、执行器和 scorer。

**结果链**：
1. documented Hermes 配置：官方分 `simple_python` **0.00**、`multi_turn_base` **0.00**。
2. dedicated adapter：**0.96** 和 **0.19**。
3. 因为 adapter 同时换了 template 和 parser → 跑了完整 2×2（**表 1**）：

   |  | documented template | dedicated template |
   |---|---|---|
   | Hermes parser | 0 | 0 |
   | Dedicated parser | 0 | **196** |

   n=200（simple_python 100 + multi_turn_base 100 的首请求）。
4. "Each component behaves according to its own contract, but the contracts disagree."
5. documented 臂：**0 个 HTTP 错误**，**166/200** 个响应满足 BFCL JSON-call 判据，却 0 解析。→ 这就是"静默"：没有任何报错。
6. `multi_turn_base`：修复后执行 0→98/100，成功 0→19/100。
7. "Documented" = 应用 Qwen2.5 家族级指引，不是 Coder 专属保证。
8. **fixed-output replay**：一个新的、answer-blind 的离线 JSON 抽取器作用于**原来那 100 个** `simple_python` 响应（不重新生成），恢复 **97** 个结构上符合 schema 的调用；**92** 个通过固定 commit 的官方 AST checker。这**区别于**部署的 dedicated parser。
   - 意义：证明正确的单轮调用**就在原始字节里**——这是 masking 最直接的证据（签名 1 的应用）。
   - 不能推出：剩下的差异（96 vs 92）全是模板变化造成的；这是一个安全的通用 fallback parser；能恢复反事实的多轮成功率（多轮里改变一个动作会改变后续环境反馈和生成）。

**τ-bench（表 2 下半）**：
- 同样的 adapter 交换：解析并执行的调用 0 → **636**；至少执行过一次的任务 0/115 → **103/115**。
- documented 臂 **1,389** 个请求全部带 tools；**789** 个 assistant 回合含 call-shaped payload 但未解析。
- 重放把 789 个全部归到 **missing-envelope**；**没有一个到达 payload 验证阶段**；配置的抽取器一个都不接受。
- 结果 7 → 10 solved，**3 个不一致对全朝修复方向，*p*=0.25**；**不做结果声明**。
- 扩展了 envelope 层机制到非代码有状态环境；规模和 RL 结论仍限于代码。

**表 2（external_main）**每一行：

| 测量 | documented | repaired | 读法 |
|---|---|---|---|
| BFCL simple_python score | 0.00 | 0.96 | 官方 accuracy，96/100 |
| BFCL multi_turn_base score | 0.00 | 0.19 | 官方 accuracy，19/100 |
| BFCL multi-turn executed/succeeded | 0/0 | 98/19 | 执行是机制，成功是结果 |
| τ-bench parsed/executed calls | 0/0 | 636/636 | 调用次数（跨 115 个任务），机制 |
| τ-bench tasks reaching a tool | 0 | 103 | 任务数，机制 |
| τ-bench tasks solved | 7 | 10 | 结果，*p*=0.25，不声明 |

【我的解释】为什么 documented 臂 τ-bench 0 次执行也能解出 7 个？可能的原因是：有些零售任务的目标状态就是"数据库不变"（比如该拒绝的请求），模型只靠对话也能"成功"。论文没有分析这 7 个任务，这里只是帮助理解的猜测。

<a id="ch3-7"></a>

### 3.6 §5.2 Scale under mismatched and matched interfaces

**Coder ladder（表 3 上半）**：documented Hermes 接口下，服务端在 5 个规模上都报告 **0** 调用；tight 分类的 emission 单调上升 **0 → 4 → 21 → 36 → 80**（/100）。

- **敏感性估计 ≈72**：如果 tight 层精度 0.900 能迁移到 32B，80 × 0.9 = 72。**不**恢复假阴性，**不**估计全部真实调用。
- **随机 300 题复制**：0, 4, 30, 40, 81。7B 从 21 → 30，但方向和单调性保持。
- "This is a within-family observation under one mismatch, **not a scaling law**."
- "the server column is constant, so the slope comes entirely from increasingly frequent or regular emissions"——服务端列是常数 0，所以斜率**完全**来自 emission 越来越多/越来越规整。
- 净偏差无法识别（见 3.4）。

**Qwen3 matched ladder（表 3 下半，预注册）**：
- Qwen3 checkpoints ≤8B，其要求的 envelope 和 Hermes **匹配**。
- 预测 "parsed > 0 at every size" 在推理前提交 → **成立**（86, 44, 70, 97）。
- **静默数**（tight 且未解析）：1, 1, 2, 0。对比：错配 ladder 在 32B 达到 80。
- 解读："consistent with contract mismatch, rather than size alone, driving the undercount"；**不能**隔离 envelope（家族和 envelope 一起变了）。
- 协议偏离：parsed 不单调；关闭了 thinking（默认在调用前就耗尽 token 预算）。
- Granite rescue 尝试：在 Hermes 和其原生 parser 下都**没有产生调用**→ **无信息**，不是确认。（【我的解释】一个从不尝试调用的 checkpoint 产生的是"无信息的 0"，不是"被删失的 0"——局限部分也用这个理由说明非代码 ladder 需要在小规模加 rescue 臂。）

**表 3 的 caption 要读懂**："Tight counts concern unparsed outputs only (including Qwen3); they are not total emitted-call counts."——tight 列只统计**未解析**的输出，不是总调用数。对 Qwen3 来说，已经解析成功的调用不在 tight 列里。

**Qwen2.5-Instruct 同谱系控制（表 4）**：
- 同样的 template、parser、题目、解码，**只换 checkpoint**（Instruct 而不是 Coder）。
- parsed：**1, 11, 63, 77, 88**（Coder 全是 0）。
- 用能工作的执行器重跑后，多轮 rescue：**0, 3, 6, 14, 23**，跟着 parsed 列走。
- 完整表：

  | size | parsed | turn-1 | final | rescues |
  |---|---|---|---|---|
  | 1.5B | 1 | 13 | 13 | 0 |
  | 3B | 11 | 14 | 17 | 3 |
  | 7B | 63 | 34 | 40 | 6 |
  | 14B | 77 | 38 | 52 | 14 |
  | 32B | 88 | 42 | 65 | 23 |

- 解读："consistent with **checkpoint-specific post-training differences**, not parameter count or the shared family/template alone"；**不能**认定工具格式训练是唯一原因。
- **4,254 个去重首轮**（全档案所有臂）：重放发现 vLLM 配置的 parser **应该接受却没接受**的情况 **0 例** → "this is contract mismatch, not a parser implementation bug"。

【我的解释】Instruct 控制的价值：它排除了"是不是 Qwen2.5 这个 template/parser 组合本来就坏了"——同一套接口下，Instruct 能被解析（越大越多），Coder 一个都不行。问题在 Coder checkpoint 的输出习惯。

<a id="ch3-8"></a>

### 3.7 §5.3 Llama: a constraint suppresses the wrong action

**现象**：Llama-3.1-8B 在 23/100 题上调用**题目要求实现的函数**而不是 `run_tests`（例子：`can_form_word`、`check_password_strength`、`floyd_warshall`）。这是**角色混淆（role confusion）**——把"要写的函数"当成"可调的工具"。

**四个单变量控制（表 5）**：

| arm | wrong-tool calls | exact paired *p* (baseline) | 检验的替代解释 | 结论 |
|---|---|---|---|---|
| terse FC | 23 | — | — | 基线 |
| rich schema | 22 | 1.000 (terse)，不一致 6/5 | "schema 缺参数描述" | 拒绝 |
| thought scaffold | **59** | 2.84×10⁻¹⁰ (terse) | "ReAct 只是多了一步推理" | 拒绝（**更糟**） |
| official template | 22 | 1.000 (terse)，不一致 1/0 | "服务器没按文档配置" | 拒绝 |
| official + strict | **0** | **4.77×10⁻⁷ (official)**，22/0 | "没开 schema 约束" | **接受** |

- 注意 rich schema 加的话："This tool is not the function you are being asked to implement. Do not pass the task function's parameters into it…" + 示例 `def add(a,b): return a+b`。**即使这样明确警告，还有 22/100 题传了任务函数参数。**
- thought scaffold 前缀："First write your reasoning in a Thought: … Once you have thought it through, call the run_tests tool." → 错调用 23→59。
- 澄清：之前引用的 *p*=0.125 属于 final-pass 比较（T09），不是错调用标签。

**strict 的完整效果**（official template 基线 → + strict）：
- 错调用 22 → **0**；有效执行 73 → **97**；turn-1 33 → **46**；final 44 → **61**（**+17 分单变量变化**，T15 *p*=7.63×10⁻⁵，**过了** Bonferroni）。

**ReAct 分解**（用 terse-FC 作起点）：
- terse-FC → official+strict：49 → 61，**+12**（interface bundle，*p*=0.0075，没过）；
- official+strict → ReAct：61 → 80，**+19**（protocol，*p*=0.0019，没过——刚好比 0.00185 大一点）；
- 这两个是描述性分差。"The official-template strict final-pass comparison does survive that correction."

**机制解读**："the model exhibits role confusion under unconstrained function calling, while the **interface determines whether that confusion reaches the environment**."

**initiation 修正**：附录 A 把 Llama 的 initiation rate 从 97 修正为 **74**（97 个"调用"里 23 个调的是任务函数）。

**DeepSeek 和 Mistral**：只用来说明还有其他失败层（template 层 / token 层）。DeepSeek 从不注入 tools；Mistral 的 4 个 FC 变体**全部**有请求错误。所以两者都**不支持**可比的性能声明。

【我的解释】这一节为什么重要：它防止读者把论文理解成"接口总是坏事、总是吃掉好动作"。Llama 展示了相反的方向——接口（strict 约束）**阻止**了错误动作到达环境。所以论文的主张是"接口是动作空间的一部分"，而不是"接口有害"。

<a id="ch3-9"></a>

### 3.8 §5.4 Repair changes evaluation trajectories

**表 6（repair_main）**：Qwen2.5-Coder-7B，只换服务适配器。

| arm | parsed | ≥2 tested steps | turn-1/final | rescues |
|---|---|---|---|---|
| FC + Hermes | 0 | 0 | 53/53 | 0 |
| FC + dedicated adapter | **84** | **37** | 53/62 | **9** |
| ReAct reference | 95 | 40 | 57/74 | 17 |

- 机制全部恢复：parsed 0→84，≥2 tested steps 0→37，rescues 0→9。
- turn-1 两臂都是 53：**一致性检查**，不是层定位（因为 adapter 也改了 template 和 few-shot）。对比 Llama 的 schema 干预让 turn-1 从 33 变 46。
- **final 53 → 62 不显著**：n=100 时 16 gains / 7 losses，*p*=0.093；随机 n=300 复制 160→178，名义 *p*=0.0385，没过 0.00185。
- "Repair restores the mechanism; we do not claim an evaluation outcome effect."

【我的解释】这一节和 RL 那一节是同一个故事的两次讲述：**机制恢复是确定性的、巨大的；结果变化是小的、不显著的**。

<a id="ch3-10"></a>

### 3.9 §5.5 Parser repair restores RL experience; no held-out improvement detected（**最重要的一节**）

**设计**（"a preregistered single-variable 150-step 7B GRPO comparison"）：
- 两臂相同：LoRA rank 32、数据、mandatory prompt、seed、解码、verifier、评测、训练预算。
- **唯一干预：verl 注册的 parser**（`multi_turn.format=hermes` vs `qwen2_5_coder`）。
- 全部 emission、parser verdict、执行、观测、哈希都保留。
- 训练事件计数排除 held-out 评测请求；每个 checkpoint 用同一个配对 542 题评测集。
- 两个 step-0 评测用同一个 base checkpoint 和评测器，它们相等（430/430）是**观察**，不是用来推断评测设计的证据。

**表 7（p3formal）**：

| arm | tight | accepted | exec. | final / rescues, step 0→150 |
|---|---|---|---|---|
| broken FC | 8,689 | 10 | 10 | 430→430 / 12→12 |
| repaired FC | 17,608 | 16,912 | 16,844 | 430→427 / 12→12 |

- tight：**生成事件**中被 tight 分类的数量；accepted：parser 接受的调用数；exec.：工具调用次数（全部都返回了观测）。final/rescues：评测**题目**数。
- **16,912 vs 16,844 的 68 差**：40 个生成里有多个调用，贡献了 68 个"第一个之外"的调用。配置的单调用上限（`max_parallel_calls=1`）选 parser 返回的**第一个**调用，其余**不派发**（既不合并也不排队）。全部 16,844 次派发都按 request ID 和 assistant turn 关联到执行和观测。抽取到派发的计数是**按轨迹**匹配的，不是按每个调用 ID（抽取记录缺这些标识）。
- **broken 通道是"几乎关闭"，不是"字面关闭"**：8,689 个 tight 生成事件只对应 10 次执行。
- 修复让采样的工具经验变化**超过三个数量级**，每次执行都返回了观测。

**held-out 结果（程序化反馈评测）**：
- repaired 427/542 vs broken 430/542：配对差 **−0.55pp**，保守 95% CI **[−1.79, 0.72]**，**3 个不一致对全朝 broken**，精确 McNemar *p*=**0.25**。
- rescues 在初始化和终点都是 12；中间 checkpoint 有波动。
- repaired 相对自己的 step 0：1 gain / 4 losses，*p*=0.375。
- "**We do not interpret these non-significant contrasts as harm.**"
- "At this scale, budget, and single seed, parser repair restores tool-mediated experience, but we detect no improvement in held-out success or rescue counts."

**held-out 评测协议（必须懂，否则会误读 broken 臂的 12 个 rescue）**：
- 两臂相同，**故意绕开两个训练 parser**：当前 LoRA 通过 verl 的 native rollout RPC 查询，temperature 0，最多 4 轮、每轮 1,024 token；生成的代码**被自动执行**在同样的 542 个 EvalPlus 题上，测试输出作为 **user 消息**追加。
- 所以这是**程序化反馈**评测，**不是**测"自愿发起工具调用"——这就解释了为什么 broken 训练臂也能有 12 个评测 rescue：评测时反馈是自动给的，不需要模型发起调用。
- 训练用的是结构化 **tool-role** 反馈；评测是 **user-message** 反馈 → 这个格式偏移可能限制迁移，此外还有适配容量、训练预算、信用分配等可能原因。
- 每个评测请求都核验加载的 global weight step。

**post-hoc 共同 native FC 评测**（回应"你的评测不测自愿发起"）：
- 三个权重（base、broken step150、repaired step150）在同样 542 题上。
- final：base **239**，broken **248**，repaired **248**；repaired vs broken：16/16 不一致，差 0pp，保守 95% CI **[−3.47, 3.47]**，*p*=**1**。
- accepted-call initiation（至少一次被接受并执行的调用）：**46 / 50 / 47** 个任务。
- tool-test rescue：**2 / 1 / 1**。
- 只有被接受的调用才收到 tool-role 反馈；终端评分独立。
- "This is **not an equivalence result** or an additional training seed."

**resume 细节**：
- repaired 任务在 step 116 后中断，从**验证过的** step-90 checkpoint 恢复；检查了模型、optimizer、scheduler、RNG、dataloader、源码、配置、weight-step 的哈希。
- 重复的 step-90 评测差了 2 个 final-pass 题（415/428/13 → 415/426/11）→ 作为**测试–复测变异**报告，不替换进正式曲线。
- 正式事件总数用最终模型的谱系（lineage）：原始 step 1–90 + 恢复后 step 91–150；**排除**被放弃的 26 次更新的 step 91–116 分支。

**两个防误读细节**：
1. **两臂都用 LoRA** 是因为 7B 全参 GRPO 单卡 80GB 放不下。保留了臂间对比，但**不能**和历史 1.5B 全参曲线严格比较。
2. **事件总数不同**（broken 19,203 vs repaired 33,367 个生成事件）是因为修复改变了轨迹：调用执行后对话轮次更多，就有更多机会再发调用。所以每臂内的比率**只用来描述该臂的漏斗**，因果归因用的是**单一注册 parser 干预**，不是原始分母相等。
- 历史 ReAct 臂（23,676 次执行，rescue 平坦）方向一致，但只是**支持性证据**（它同时改了协议和接口）。

<a id="ch3-11"></a>

### 3.10 §6 Implications and Limitations

**Measure the funnel（实践建议）**：报告应包括：
- 在明确判据下的 emitted-call rate；
- parser acceptance；
- dispatch/execution；
- observation return；
- request errors；
- adapter/template/parser 的版本和哈希；
- task outcome。

**98 行 preflight**（`preflight_toolcall.py`）：发一个规范的带 tools 请求，验证返回的 name 和 arguments，再在 `tool_choice: required` 下重复作为 positive control。这是**筛查**检查：能标出没返回预期工具调用的配置，**本身不能识别底层失败机制**。

**generation constraints** 是另一种潜在补救：改变 emission 而不是重新解析已有字节；本身不保证任务成功。

**measurement card（测量卡）**：应成为机器可读的 conformance report，至少持久化：模型和 tokenizer revision、template 和 parser 哈希、解码参数、请求错误、emitted-intent 判据、acceptance、dispatch、execution、observation 计数、outcome delta。"The standard changes the unit of reporting from 'model call rate' to a versioned **model–interface measurement card**, making silent zeros auditable and adapter changes comparable."

**Interpretation limits**（正文版的局限）：
- 规模和 RL 结果：一个代码任务、一个工具；只有 envelope 机制在非代码基准上复制了。
- 规模趋势：一个家族、一个错配、单样本头条臂、32B 上限。
- RL：一个 7B LoRA 种子和预算；接近零的样本不等于零策略概率。
- 27-test 族阈值 0.00185，只有最强的干预过了；不把任何结论建立在名义显著或零结果对比上。
- 历史 verifier 的缝隙不能解释正式两臂的对比（两臂用同一个修好的 verifier）。
- 版本依赖：vLLM 0.27.1、verl 0.9.0；native FC 用 Transformers 5.10.4、PEFT 0.20.0。
- 附录 H 列了全部 **17 条**局限。

**直接影响头条的几条**：
- 原始 100 题是 `clean[:100]`，不是随机抽样；随机 300 题复制把 7B 档从 21 移到 30。
- 两臂都解析子集上的对比是后处理选择：Qwen 残差 +8.4pp（CI [−0.5, 17.4]，*p*=0.118）是**未解决的**，不是等价证据。
- taxonomy 是**故意不对称**的：Qwen 和 Llama 有受控修复；Mistral 没有可准入的 100 题 FC 运行；DeepSeek 只有模板诊断。
- 完整的"尝试 vs 丢弃"归因只在新的 7B 运行里有；历史 1.5B 数据是过度决定的，不用于因果训练主张。

**What would change the conclusions（可证伪性）**：
- 复制的或更长的 repaired-FC 运行若出现一致的 rescue 增益 → 缩小当前 150 步单种子零结果的范围。
- 第二个家族的欠计数若不随规模增长 → 规模观察只限于 Qwen+Hermes；若同样上升 → 加强规模解释。
- 非代码的规模 ladder → 分离 envelope 机制和代码形状 payload。
- 第二个服务栈（如 SGLang）→ 识别哪些发现是 vLLM 特有的。

<a id="ch3-12"></a>

### 3.11 §7 Conclusion

三句话：
1. "A tool-call rate measures a model–serialization–parser–execution **composition**."
2. BFCL：单组件替换都不能从基线恢复解析，联合替换让官方分 0.00→0.96；RL：只换 parser 让观测到的执行 10→16,844，两种共同评测协议下都没检测到增益。
3. "verify the contract and report each funnel layer before attributing a terminal zero, a protocol gap, or a training curve to the model."
   - 注意这里点名的三种误归因：**terminal zero**（终端的 0 调用）、**protocol gap**（ReAct vs FC 的差距）、**training curve**（训练曲线）——正好对应论文的三大块证据。

---

<a id="ch4"></a>

## 4. 附录逐个精读

<sub>[↑ 回到目录](#toc) ｜ [← 上一章：3. 正文逐节精读](#ch3) ｜ [下一章：5. 数字速查表 →](#ch5)</sub>

附录顺序（`main.tex` 里 `\input` 的顺序）：A Errata → B Reproducibility → C Prompts（英文渲染）→ D Config → E Arm inventory → F Human validation → G Supplementary results → H Limitations in full → I Statistics。

<a id="ch4-1"></a>

### 4.1 附录 A：Data Errata（"披露而非吸收"）

开篇一句话定调："Every number in this paper traces to a trajectory file under `runs/final/`." 内容复制自仓库 `runs/final/ERRATA.md`。

**A.1 七个 ReAct 臂记录了错误的 `max_tokens`**
- 记录 2048，实际 **1024**。原因：探针里 `gen()` 默认 `max_tokens=1024`，而 `gen_fc()` 用全局 `MAX_TOKENS=2048`，但写到磁盘的 provenance 两者都记了 `MAX_TOKENS`。2026-08-31 修复。
- 受影响：`traj_v3_{Llama8B,Qwen7B,Mistral7B}_react`、`traj_v11_{DS1.3b,DS6.7b,Llama32_1B,Llama32_3B}_react`。
- **偏差方向**：同期 FC 臂真的是 2048 → 在每个受影响的 ReAct-vs-FC 比较里，**FC 一方的生成预算更大**。论文谨慎地说"更大预算不一定提升性能，所以这个不对称本身不能确定偏差方向"。
- 用真正 2048 重跑三个主 ReAct 臂：final 最多变 1 题（80→80，74→74，32→33）。
- 替换数据：v11 → `traj_v13_*_react.jsonl`；v3 → `traj_v14_*_react.jsonl`。所有表格引用 v13/v14。

**A.2 Llama-3.1-8B initiation rate 手动修正**
- `traj_v3_Llama8B_fc.jsonl` 记录 initiation 97/100，但其中 23 个调用的是**任务函数**。当时探针不验证工具名，`has_action` 区分不了。
- **修正后的 `run_tests` initiation = 74/100**。
- 针对这 23 题的复跑（`traj_v8_Llama8B_recheck.jsonl`，seed 固定）：23/23 复现为错调用。
- 同批次：Qwen7B fc_nosuffix initiation 0 不用修；Mistral7B fc 是 2，那 2 个没逐一复查工具名。

**A.3 不可准入的臂**
- `traj_v3_Mistral7B_fc` n_err=2；`traj_v5_Mistral7B_fc_official` n_err=42；`traj_v9_Mistral7B_fc_strict` n_err=3；`traj_v9_Mistral7B_fc_official_strict` n_err=39；`traj_v13_Llama8B_fcstrict_t06_s3` n_err=1。全部 rc=2。
- **错误率普查本身仍然有效**（它本身就是结果），但这些臂被排除在 clean-arm pass-rate 比较之外，不算 clean-arm *p*。
- 早期草稿在其中一个臂上算了 *p*=0.648 → **撤回**。

**A.4 provenance schema 漂移**
- 四代 schema：早期臂缺 `seed`、`temperature`、`fc_schema`。
- 所有结论依赖的字段（`model`、`protocol`、`adapter`、`max_tokens`、`clean_index`）在所有代里都有。
- `script_sha256` 字段从未成功加上（补丁被打断两次）→ 脚本身份由外部保证：启动前哈希探针（`v13_pinned_hashes.txt`），之后再核对。

**A.5 已验证正确的**
- 全部 50 个完整臂用同一个 100 题集合（`clean[:100]`，unique item-set count = 1）→ 每个跨臂比较都是严格配对的。
- 每个 FC 臂都真的在 `max_tokens=2048`。
- 方差实验的采样是真的：temperature 0.6 下两个 seed 在 95/100 题上首轮代码不同，16/100 题结果翻转。

**A.6 第三方组件版本**
- hanXen parser：commit `1b921501…`（`1b92150`，2026-04-29，"fix: buffer partial <tools> prefix in streaming to prevent tag leak"），2026-08-31 下载。
- 文件哈希（前 20 位）：parser `c16bb1f8…`（未修改）；template `736bd175…`（**已修改**）；template.orig `a95b2a9b…`（原版）。
- 修改原因：few-shot 调用 `get_weather`（见 2.4）。

**学习要点**：这一节体现的研究规范是"**每个错误都被枚举，没有一个被悄悄吸收进报告的数字**"（局限 17 原话："None is silently absorbed into a reported number"）。如果被问"你们做过哪些纠错"，这一节就是答案。

<a id="ch4-2"></a>

### 4.2 附录 B：Reproducibility

- 补充包包含：精简的 Llama outcomes、**11,952** 条精简的 7B 评测记录、紧凑事件（包括被排除的段）、lineage 元数据、验证脚本；还有一个小的原始内容机制档案：BFCL 析因的 800 个首响应、100 个 documented + 100 个 repaired 单轮 BFCL 记录（含请求 schema 和官方打分输入）、全部 230 条 τ-bench 任务轨迹（115×2）、每臂 10 条确定性选出的 step-1 RL 轨迹。
- **不包含**：完整训练 rollout 档案、原始评测程序、训练日志、大的 checkpoint。
- arXiv v2 的措辞：补充包 "available from the authors on request"（部分内容不在公开仓库里）。

**能验证 / 不能验证**：

| 验证目标 | 包内支持 |
|---|---|
| 从精简题目记录复算 Llama 配对检验 | 是 |
| 从陈述的不一致计数复算其他历史检验 | 是，仅计数 |
| 7B checkpoint outcomes 和终点配对 | 是 |
| 7B 记录的漏斗和接受的 resume 段 | 是 |
| 检查/重放提供的 BFCL 原始输出 | 是 |
| 检查 τ-bench 和采样的 RL 原始内容 | 是 |
| 重新分类每一个训练 emission | **否** |
| 验证 checkpoint 权重或原始文本哈希 | **否** |

- 三条命令：`analysis/verify_supplement.py`、`analysis/submission_statistics.py`、`analysis/bfcl_fixed_replay.py --bundle mechanism_evidence/bfcl`。
- 审计边界：派发、执行、观测可以唯一关联；抽取缺少每轮请求标识，所以 accepted→dispatch 只能在**轨迹计数粒度**匹配。BFCL 请求字段来自其推理日志，不是逐字节 HTTP 抓包。"**Hashes do not substitute for file contents.**"
- 仅完整档案可跑：`analysis/p3_final_evidence.py`、`analysis/intent.py`、`runs/final/final_table.py`、checkpoint 审计、历史图重绘。
- native FC：`native_fc/outcomes.json`（**1,626** 条 = 542×3）、`native_fc/statistics.json`、`analysis/native_fc_results.py --outcomes …`。CPU 上复算配对对比和超时清单，**不重新执行**生成的程序。

<a id="ch4-3"></a>

### 4.3 附录 C：Prompts and tool schemas（英文渲染）

原始 prompt 在 `paper/sections/C_prompts.tex`（中文版），这里的英文渲染"provided only for readers"。

- **FC_OPTIONAL**："You may call run_tests to submit your code to the test runner…"
- **FC_MANDATORY**："You must first call run_tests to verify your code; do not answer directly. … If any test fails, revise the code and call run_tests again."
- **terse schema**（文档级基线）：`run_tests`，描述 "Submit the Python code you wrote to the test runner; it returns pass/fail status and errors."，参数 `code: string`，required。
- **rich**：加了"这个工具不是你要实现的函数……"的警告 + `def add(a,b)` 示例。仍有 22/100 错。
- **strict**：terse + `"strict": true` + `additionalProperties: false` + 开启 strict tool calling → 错调用归零。
- **thought scaffold**：先写 Thought 再调用 → 23→59。
- **1.5B role-disambiguation**：给出错误示例 `{"name":"join_integers","arguments":{"int_array":[1,2,3]}}` 和正确示例 `{"name":"run_tests","arguments":{"code":"def join_integers…"}}` → 可恢复调用 52→64，但正确命名仍是 **0**，final 15→13。
- **intent criterion**：
  - KodCode：tight regex（见 3.4）。strong 和 weak 用不相交的轨迹标签集；tight 独立计算。
  - BFCL：JSON-call 指示器 = 内嵌可解码 JSON 对象、命名了请求中提供的工具、参数是对象（包括 JSON 字符串编码的参数）。format-marker 和 JSON-marker 是独立的更宽检查，**互相重叠，不能相加**。
  - τ-bench：通用的命名调用模式或 Hermes tool-call marker，出现在没有结构化 tool calls 的 assistant 消息里。**不验证**是否属于提供的工具集，**不保证**语法有效。
  - KodCode 的标注精度**不假设**能迁移到两个外部基准。

<a id="ch4-4"></a>

### 4.4 附录 D：Serving and training configuration

**D.1 每家族的服务参数（表 serving）**：

| 家族 | 参数 / 干预 | 观察到的症状 |
|---|---|---|
| all | `--enable-auto-tool-choice` | 不加 → HTTP 400 |
| Qwen2.5 (Instruct) | `--tool-call-parser hermes` | 各规模 1–88/100 解析；未解析的包括 payload 失败 |
| Qwen2.5-Coder | 基线：hermes；修复：社区 parser + 配套 template | 基线各规模 0 解析；联合适配器 BFCL 196/200 |
| Llama 3.1/3.2 | `--tool-call-parser llama3_json` + `--chat-template tool_chat_template_llama3.1_json.jinja` + schema 里 `"strict": true` | 参数格式错误；不加 strict 时 terse 基线 23% 错调用 |
| Mistral (HF format) | `--tokenizer-mode hf --config-format hf --load-format hf` + `--tool-call-parser mistral` + `tool_chat_template_mistral_parallel.jinja`；`tool_call_id` 必须恰好 9 字符 | HTTP 400；官方推荐的 parallel template 把错误率从 **2% 抬到 42%** |
| DeepSeek-Coder | 无：chat template 从不注入 tools | n/a |

- 关键对比："Missing auto-tool-choice support and the tested Mistral configurations produce **explicit** request errors. Other mismatches can return **HTTP 200 with empty `tool_calls`**, indistinguishable at that field from declining to call a tool." → 显式失败 vs 静默失败。
- baseline provenance：vLLM 0.27.1 文档 "Qwen Models" 一节（2026-09-23 核查）。

**D.2 训练配置（历史 1.5B）**：
- Qwen2.5-Coder-1.5B-Instruct；verl 0.9.0（vLLM 0.27.1 rollout 后端）；GRPO×2 种子 + RLOO×1 种子；16×8=128 条轨迹/step；150 步（历史诊断臂 10/3/75 步）；KodCode 去污染池 7,669 题；held-out：multi 164 HumanEval+ + 378 MBPP+，repair 155 HumanEval+ + 299 MBPP+；分母：正式 542 multi + 454 repair 记录，历史 multi 540；`tool_agent`（FC）或 `react_agent`（ReAct）；`run_tests` 单 `code` 参数沙箱 pytest；`FC_MANDATORY`/ReAct mandatory（强度匹配）；单张 A800 80GB。

**正式 7B 配置**（全部要记住）：
- Qwen2.5-Coder-7B-Instruct，GRPO，LoRA rank = alpha = 32，all linear，seed 0，150 步，16 prompts × 8 rollouts。
- mandatory FC prompt，修好的 verifier，rollout 全文保留。
- 评测 step 0/30/60/90/120/150，542 题。
- 训练设置：lr **1e-6**，temperature **1.0**，top-p 1，top-k −1，prompt 预算 **2,048**、response 预算 **6,144** token，最多 **5** 个 assistant 轮，tool response 上限 **1,200** token，每次生成派发 **1** 个调用。
- 评测设置（不同于训练！）：最多 **4** 轮、每轮 **1,024** token、temperature 0、user 消息反馈。
- 去污染参数（见 2.8）。

**D.3 Tool-call instrumentation**（见 2.9 instrumentation 陷阱）
- 历史诊断运行：FC 计数文件 **0 行**；ReAct **788 行**，全部带 `react` 标签（来源干净）。
- 三个相互独立的信号一致：`num_turns` 钉在最小值 2、tool time 正好 0.00s、专用计数器 0。

**D.4 构建 repaired-FC 训练臂（看起来被三层挡住）**：
1. `multi_turn.format=hermes` 选的是 verl 自己的 `ToolParser` 注册表 → vLLM parser 插件对训练无效。
2. 我们写的 verl 侧 parser 能恢复调用，但**没有一个是那个工具**：36/52 调任务函数，16/52 代码在 body 别处。名字归一化救不了——没有调用携带可归一化的代码。
3. 1.5B checkpoint 根本不产出正确命名的调用；7B 全参单卡放不下。
- 前两点决定了干预方式；第三点决定了规模和 LoRA 配置。

**D.5 通往 repaired-FC 训练臂的路（四个障碍）**：
- (i) vLLM 插件到不了训练循环（verl 的 `tool_parser.py`，hermes 从 vLLM v0.9.1 移植）。
- (ii) verl 侧 parser（接受 `<tools>`、```json、裸 JSON）在 1.5B 上把可恢复调用 0→52/100，但 0 个叫 `run_tests`；全是 `can_form_word({"tiles","word"})`、`check_password_strength({"password"})` 这种——和 Llama 同样的签名，模型小了 5 倍。原始 override 文件本地找不到，无法验证 prompt 字节完全等价。
- (iii) role-disambiguation few-shot 让情况更糟（52→64 可恢复，15→13 final，正确命名仍 0）——§pressure 模式的第三个独立实例。
- (iv) verl 0.9.0 的 vLLM rollout 路径不暴露约束解码。"This is a limitation of the tested rollout path, not an argument against constrained decoding."

<a id="ch4-5"></a>

### 4.5 附录 E：Arm inventory（50 个臂）

- 从轨迹文件自己的 provenance 生成，不是手抄。
- 全部 50 臂共享 `clean[:100]`（distinct-set count 验证 = 1）。
- `---` 表示那一代 schema 没有这个字段（披露而非回填）。
- `strength` 列是**记录的 CLI 选择器**，不一定是生效的 prompt：`a14` 和 `roleprobe` 指定了 `sys_file`，探针在选完内置 prompt 后加载它，覆盖了 `strength=optional`。原始 `sys_fc_mandatory.txt` 本地没找到，a14 的确切 prompt 字节无法独立验证。【学习要点：**配置日志 ≠ 生效配置**。】
- 不列入：`v8_Llama8B_recheck`（23 题定向复跑）和 8 个 n=3 的 `*smoke*` 文件。

<a id="ch4-6"></a>

### 4.6 附录 F：Human validation of the intent criterion（**审稿人最爱追问的部分**）

**目的**：规模趋势依赖自动分类器，所以要人工验证。

**抽样**：98 个输出，**按分类器判决分层**（40 tight、28 strong-but-not-tight、30 neither），故意过采决策边界，使假阳性和假阴性都可估。seeded；标注者只看原始输出，看不到分类器标签。

**第一轮（本身就是一个发现）**：

| | agreement | κ | precision | recall | FP | FN |
|---|---|---|---|---|---|---|
| Round 1（只看原始输出） | 86.7% | 0.713 | 70.0% | 96.6% | 12 | 1 |
| 裁决后 | 96.9% | 0.936 | 95.0% | 97.4% | 2 | 1 |

- 12 个第一轮假阳性的两个原因：
  - **11 个是阅读顺序失败**：调用在文档中间、在一个 ```python 代码块之后（匹配位置 1209–2585，输出中位长度约 2000）；标注者读完代码块就以为"这是直接回答，不是调用"，停下了。
  - **1 个是仪器失败**：一个 32B 题的匹配段从第 2585 字符开始，而标注包把每个输出截断在 **2400** 字符——证据根本不在包里。
- 重新只展示匹配段，13 个争议题里 10 个被推翻。修正后的包带完整输出。
- **第一轮裁决是单边的，κ 不能当可靠性估计**：被裁决的 13 题**恰好**就是标注者和分类器不一致的那些（13/13 重叠），10 个推翻**全部**朝分类器方向，没有一个反向。在这种选择规则下一致性只会上升 → 0.713→0.936 是"选了哪些题复查"的性质，不是可靠性证据。

**第二个独立标注者 A2**：对分类器、A1 的标签、第一包的题目顺序都盲；同样 98 题，逐字相同的 rubric。两个改变：输出**完整**携带；裁决规则**事先固定**为复查**所有**三方不一致（而不是只复查和分类器不一致的）。

| Cohen's κ (95% CI, bootstrap) | 全部 (n=97) | **同字节 (n=62)** |
|---|---|---|
| A1 vs A2 | 0.736 [0.595, 0.866] | **0.967 [0.893, 1.000]** |
| A2 vs classifier | 0.936 [0.852, 1.000] | 0.967 [0.896, 1.000] |
| A1 vs classifier | 0.712 [0.564, 0.849] | 0.934 [0.829, 1.000] |

- 98 题里 97 题有完整的 A1/A2 配对记录。
- 12 个配对分歧里 11 个全文超过 2,400 字符、A1 看到的呈现和 A2 不同。论文**不**把 11 个都归给同一原因。
- 62 个同字节题上两人在 61 个上一致。

**第三方裁决 + 金标准**：
- 15 题在 A1、A2、分类器三者间不一致 → 全部交第三方裁决者（按预注册规则：**任何**三方不一致，绝不是"和分类器不一致"）。裁决者看全文，看不到任何人的标签。
- 裁决者站 A2 11 次、站分类器 9 次、站 A1 6 次；**15 次里 6 次偏离分类器**。这个数才是关键检查——第一轮裁决 13 次里 0 次偏离分类器，那就是"选择规则已经决定了结果"的样子。
- 83 个一致题 + 15 个裁决题 = 98 题金标准。
- 但：分类器影响了原始分层抽样和哪些题需要裁决 → **不称金标准独立于分类器**。

| κ (95% CI) | full n | full sample | 同字节 (n=62) |
|---|---|---|---|
| A1 vs A2 | 97 | 0.736 [0.595, 0.866] | 0.967 [0.893, 1.000] |
| **Classifier vs gold** | 98 | **0.871 [0.756, 0.958]** | 0.934 [0.829, 1.000] |

**敏感性估计**：tight 层精度 36/40 = 0.900；若迁移到 32B，80×0.9 ≈ 72。不恢复假阴性，不是全部真实调用的估计。金标准样本有 4 个 FP 和 2 个 FN，其未加权差不是总体偏差估计。统一乘 0.900 得 0, 3.6, 18.9, 32.4, 72——**按构造**保持趋势，所以这个计算**不能独立验证趋势**（各规模的精度没测）。

**精度估计的历史轨迹**（论文主动报告轨迹而不只报终点）：**0.950**（第一轮单边裁决，被选择规则抬高）→ **0.975**（只用 A2 的标签，依赖单个评分者）→ **0.900**（对裁决金标准，唯一由事先固定规则构建的）。

**测量本身的边界**：每题最多存 4000 字符；98 个样本里 10 个触顶，其中一个灰区题是被截断的调用。全部 10 个触顶都在 32B 子集（0/6, 0/13, 0/15, 0/24, 10/40）。→ 截断造成假阴性的向下压力；但分类器也有假阳性 → 原始计数的净偏差未知 → 同时报告未修正计数和依赖假设的敏感性估计，**不称其为严格下界**。

<a id="ch4-7"></a>

### 4.7 附录 G：Supplementary results（最长的附录，逐小节）

**G.1 Historical observations（动机）**
- 历史 1.5B：pass 数值上涨 2.6–2.8 分（*p*=0.064/0.078/0.077，都不显著）。
- 新通过的题里 91–94% 是在 turn 1 通过的；多轮 rescue 在 540 题里只有 5–10 个。
- 图 fig4：左：pass@1 在三个 run 上都随 150 步上升，由首稿（虚线）驱动而非最终答案；中：rescue 5–10/540 跨 checkpoint、算法、种子；右：seeded-bug 修复通道**确实**在提升（模型在学"修代码"的某些东西），但没检测到多轮 rescue 增加。
- **这些只是促使调查的小探索计数，不识别接口效应，也不独立证明没学到东西。**

**G.2 正文引用的结果表**

*BFCL 首请求漏斗（tab:bfclfunnel）*：

| Arm | cases | HTTP err | first parsed | unparsed | JSON-call | format | JSON |
|---|---|---|---|---|---|---|---|
| hermes (documented) | 200 | 0 | **0** | 200 | 166 | 169 | 187 |
| qwen2_5_coder (repaired) | 200 | 0 | **196** | 4 | 0 | 1 | 2 |

emission 指示器只在解析失败的地方测；三列重叠，**不能相加**。

*τ-bench（tab:tau）*：assistant turns 1389 vs 1971；server-parsed 0 vs 636；emitted but unparsed **789** vs 6；executions/observations 0 vs 636；进入工具循环的任务 0 vs 103；solved 7 vs 10。全部 115 任务配对，无丢弃。
【我的解释】repaired 臂的 assistant turns 更多（1971 vs 1389），因为工具能执行了，对话就会继续。

*Matched envelope 反事实（tab:counterfactual）*：Qwen3-0.6B/1.7B/4B/8B parsed 86/44/70/97、静默 1/1/2/0；Coder 1.5B–32B 错配 0 at every size，静默 0→4→21→36→80。

*Instruct 完整表（tab:instruct）*：

| Size | Instruct parsed | tight | weak | Coder parsed | Coder turn-1 = final | Instruct turn-1 | final | rescued |
|---|---|---|---|---|---|---|---|---|
| 1.5B | 1 | 0 | 7 | 0 | 31 | 13 | 13 | 0 |
| 3B | 11 | 56 | 2 | 0 | 37 | 14 | 17 | +3 |
| 7B | 63 | 21 | 0 | 0 | 53 | 34 | 40 | +6 |
| 14B | 77 | 19 | 0 | 0 | 53 | 38 | 52 | +14 |
| 32B | 88 | 8 | 0 | 0 | 68 | 42 | 65 | +23 |

- 注意 Coder 的 turn-1 = final（31/37/53/53/68）：工具通道断了，没有 rescue。
- Instruct 3B 的 tight=56 很高但 parsed 只有 11：大量 envelope/payload 层失败（对照 failure-layer 表：3B 有 41 个 malformed + 26 个 strictness-only）。
- 【我的解释】有意思的对比：Coder 的 turn-1/final 比 Instruct 高（例如 7B 53 vs 40），说明 Coder 写代码更强；但 Instruct 能用工具反馈救回题目。

*Llama 臂（tab:llama_arms）*：

| Arm | Calls issued | Wrong-tool | Executed | Turn-1 | Final |
|---|---|---|---|---|---|
| terse (parser only) | 97 | 23 | 74 | 34 | 49 |
| + official template | 95 | 22 | 73 | 33 | 44 |
| + official + **strict** | 98 | **0** | **97** | **46** | **61** |
| ReAct (reference) | 98 | 0 | 97 | **61** | 80 |

- Calls issued − Wrong-tool = Executed（97−23=74；95−22=73；98−0=98≈97，有一个没执行，可能是空代码之类）。【这个 98 vs 97 的差论文没解释，别过度解读。】

*Llama 分解（tab:decomposition）*：terse→strict 15/3 *p*=0.0075 +12；strict→ReAct 27/8 *p*=0.0019 +19；terse→ReAct 38/7 *p*=3.1e-06 +31。

*Qwen 修复循环（tab:repair）*：

| Arm | Parsed | Mean tested steps | ≥2 tested | Turn-1 | Final | Rescued |
|---|---|---|---|---|---|---|
| ReAct | 95 | 1.92 | 40 | 57 | 74 | 17 |
| FC + hermes | 0 | 0.96 | 0 | 53 | 53 | 0 |
| FC + adapter | 84 | 1.82 | 37 | 53 | 62 | 9 |

ReAct 行用 v3；统一的 v14 重跑 mean tested steps 1.93，pass/rescue 不变。

*历史 RL rollouts（tab:training）*：FC（tool_agent + hermes）10 步，`num_turns/mean` **2.000（常数）**，tool time **0.00 s**，每 rollout 调用 **0.000**；ReAct（react_agent）3 步，5.883，12.21 s，**2.052**。
【我的解释】`num_turns = 2` 是最小值：1 个 user + 1 个 assistant，说明从来没有工具轮次。

**G.3 离线重解析矩阵 + 失败层分解**

*目的*：排除"你的抽取器就是比 hermes 好"——对**同样存储的字节**用四种规则重解析（`analysis/reparse_matrix.py`，无需 GPU）。

| Arm（server=0） | server | hermes replay | `<tools>` replay | bare JSON | tight |
|---|---|---|---|---|---|
| Qwen-1.5B | 0 | 0 | 0 | 0 | 0 |
| Qwen-3B | 0 | 0 | 0 | 5 | 4 |
| Qwen-7B | 0 | 0 | 0 | 22 | 21 |
| Qwen-14B | 0 | 0 | 0 | 42 | 36 |
| Qwen-32B | 0 | **0** | 0 | **100** | **80** |

- hermes 列每个规模都是 0：模型从不产生那个格式。
- bare JSON 和 tight 不是 0：调用存在，它们和 server 列的差就是格式错配移除的部分。
- server 已经解析成功的臂上，离线 content-only 重解析是**未定义（N/A）而不是 0**。
- 【注意】bare JSON 在 32B 是 100 而 tight 是 80：bare JSON 规则更宽松（不要求 `run_tests` 名字和 Python 词法标记等）。

*失败层（tab:layer）*：用 `analysis/failure_layer.py` 逐行重放 vLLM 0.27.1 的 hermes 抽取器。

| Arm | no envelope | payload malformed | strictness only | **parser loss** |
|---|---|---|---|---|
| Coder 1.5B–32B（每个） | 100 | 0 | 0 | **0** |
| Instruct 1.5B | 93 | 6 | 0 | **0** |
| Instruct 3B | 22 | 41 | 26 | **0** |
| Instruct 7B | 3 | 33 | 1 | **0** |
| Instruct 14B | 13 | 6 | 4 | **0** |
| Instruct 32B | 1 | 9 | 2 | **0** |

- 这些是**所有未解析输出**的首失败类别，不是"识别出的工具尝试"计数。
- 只看缺 envelope 不能识别 censoring：每个规模的 100 个 Coder 输出里既有调用形状的也有非调用的（尤其 1.5B）。需要单独的 emission 判据。
- Instruct 的 payload 失败几乎总是因为把 Python 嵌进 `code` 字符串时 docstring 引号或原始换行没转义。
- **全档案没有一例服务端没解析、而它自己的抽取器会接受的调用。** → "Censoring" 在本文的意思是"**尝试从未以 parser 定义接受的形式到达 parser**"，**不是** "parser 丢弃了有效调用"。

*n_turns 计数器*：只有抽到代码并测试时才 +1。Hermes 臂 100 个首输出里 4 个没代码 → 96/100 tested steps。Hermes 臂平均 assistant 输出数 1.00，adapter 臂 1.90（而 tested-step 均值是 1.82）。

**G.4 BFCL：gating 和完整复现记录**

*fixed-output replay*（`analysis/bfcl_fixed_replay.py`）：
- 对 clean run 的全部 100 个 `simple_python` documented 臂响应做事后、answer-blind 重放。
- 确定性抽取器：从左到右扫描最大 JSON 对象或数组，接受带对象参数的命名调用（包括 JSON 编码的参数字符串），保留所有恢复的调用。**不修复**坏 JSON，**不**用答案选调用。
- 恢复的 97 个响应全部命名了提供的工具并通过结构签名检查（必需字段、类型、声明的参数名、数组项、枚举）；不验证格式注解。
- 100 个原始响应里 **92** 个通过固定 commit 的官方 BFCL AST checker。
- **阳性对照**：100 个 repaired 臂输出复现 96 个官方成功。
- 打分代码不变；离线 loader 提供同样的 `underscore_to_dot=True` 注册表属性，不导入在线推理 SDK。
- "This is an additional offline interpretation rule, **not the community dedicated parser**: none of the original 100 responses contains its required `<tools>` start tag."

*gating*：每个臂完整跑之前先过两个 case 的端到端 smoke test；result count、score 文件的 `total_count`、recording proxy 里不同的 case id 数**必须都等于 200**，且 0 个 HTTP 错误，否则拒绝这次运行。

*两次运行的稳定性*：hermes 臂两次完全一致（0/200 parsed，166 tight，480 请求，0 错误）；repaired 臂差 1 个 case（first parsed 197 vs 196，executed 99 vs 98），success 都是 19。原因：贪心解码在不同 batch 组合下 vLLM 不保证逐位确定；这个臂 8 个线程跑多轮执行器 → 中间漏斗列报 ±1。

*multi_turn_base 漏斗（tab:bfcl5）*：hermes：tight 69，first parsed 0，any parsed 0，executed 0，success 0；qwen2_5_coder：tight 0，first parsed 97，any parsed 98，executed 98，success **19**，rescued 0。

**G.5 四家族 taxonomy（tab:taxonomy_full）**

| Family | Layer | Symptom | 能事先检测？ | Remedy |
|---|---|---|---|---|
| DeepSeek-Coder | template | chat template 从不注入 tools | ✓ 模板检查 | 无 |
| Qwen2.5-Coder | parser | 输出 fenced JSON 而非 `<tool_call>` | × 模板检查**假阳性** | 专用适配器 |
| Llama-3.1-8B | schema | 把**任务函数**当工具调（23%） | × 需检查参数 | `strict: true` |
| Mistral-7B-v0.3 | token | 重复 `[TOOL_CALLS]` → HTTP 400 | ~ 有报错，但只是有时 | **没找到** |

- caption："Four serving diagnostics, **not an exhaustive taxonomy**. Only Mistral's failure announces itself."（只有 Mistral 的失败会"自报家门"，其他都是静默的。）
- 图 fig2：一致的 server-parsed 指标。Llama 是 97 vs 98 parsed；错调用和执行数分开报。灰色 = 按文档配置；绿色 = 找到的最佳配置；Mistral 在四种配置下都没有无错运行 → 橙色，不可比。
- 图 fig1：错配接口下的家族内观察，同表 3 的数据。

**G.6 统一配置下的主表（tab:main）**：所有臂真的 `max_tokens=2048`，同一 100 题，temperature 0。

| Model | ReAct turn-1 | ReAct final | FC turn-1 | FC final | b/c | *p* | FC arm |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | 61 | **80** | 46 | 61 | 27/8 | 0.0019 | official tmpl + strict |
| Qwen2.5-Coder-7B | 57 | **74** | 53 | 62 | 17/5 | 0.0169 | dedicated adapter |
| Mistral-7B-v0.3 | 26 | 33 | N/A | N/A | N/A | N/A | — |

**G.7 四个 Llama 控制的表**（见 3.7）。caption 的最后一句很重要："eliminated it through a schema constraint, **without identifying a unique cause of role confusion**."

**G.8 条件化于成功解析 + 反馈的"接触 vs 使用"**

*both-parsed（tab:conditioned）*：Qwen 83 题 ReAct 63 vs FC 56，11/4，*p*=0.118（n.s.）；Llama 96 题 78 vs 60，25/7，*p*=0.0021。都是**描述性的后处理子集**，都没过 Bonferroni。

*given feedback（tab:given_feedback）*：

| Arm | 到达 ≥2 tested step | Rescued | 给定反馈时的修复成功率 |
|---|---|---|---|
| ReAct | 40 | 17 | 42.5% |
| FC + adapter | 37 | 9 | 24.3% |
| FC + hermes | **0** | **0** | undefined |

子集是处理后选出的，这些比率**不能**定位或因果分解协议差距。

*Adapters are not a universal switch*：同一个 adapter 在 7B 恢复 84/100，在 **3B 只有 1/100**：3B 完全无视 `<tools>` few-shot（100 题里 0 个 `<tools>` 标签，99 个直接写代码），但解题率差不多（final 53）。适配器效果随规模**非单调** → 强化"按 checkpoint"的主张。另外 dedicated 方案同时改了 parser、template、few-shot，我们还改写了 template 的 few-shot（`get_weather`）。

**G.9 Pressure on a broken interface makes things worse（"加压"适得其反）**

两个独立控制指向同一方向：
1. FC 加推理脚手架：角色混淆 23→**59**。
2. Qwen2.5-Coder-1.5B 把可选工具指令换成强制：parser 接受的调用**完全没变**（仍 0/100），final pass **31→15**，无法解析的输出从 ~0 升到 46/100。

两者都不触及接口，都让可观测结果变差。**金句**："When an agent appears reluctant to use tools, strengthening the instruction is the cheapest thing to try and the most likely to mislead"——它能降低任务表现而让工具调用数保持为 0，看起来像一个"既不愿意又没能力"的模型。

*1.5B probe 分类（tab:probe_categories）*（n=100，记录的 mandatory prompt override）：

| Category | n | % |
|---|---|---|
| 错名字，`arguments` 带完整代码 | **0** | 0% |
| 错名字，`arguments` 是任务函数的参数 | **36** | 69% |
| 错名字，可执行代码在 body 别处 | 16 | 31% |

"Not one output names `run_tests`, so at this scale the missing calls are **not a parser mismatch** — there was nothing well-formed to censor."（这就是为什么历史 1.5B 不能作因果证据。）

*加压表（tab:pressure）*：mandatory 基线：可恢复 52、命名 `run_tests` **0**、final 15；+ role-disambiguation few-shot：**64**、**0**、**13**。

**G.10 正式 7B checkpoint 曲线 + 最终谱系审计**

完整曲线（turn-1 / final / rescued，n=542）：

| step | broken FC | repaired FC |
|---|---|---|
| 0 | 418 / 430 / 12 | 418 / 430 / 12 |
| 30 | 418 / 430 / 12 | 418 / 429 / 11 |
| 60 | 420 / 433 / 13 | 417 / 431 / 14 |
| 90 | 418 / 432 / 14 | 415 / 428 / 13 |
| 120 | 419 / 433 / 14 | 415 / 427 / 12 |
| 150 | 418 / 430 / 12 | 415 / 427 / 12 |

- 所有 checkpoint 评测用同一个 542 题 manifest，0 请求失败，核验指示的 native weight step。
- 接受的曲线：原始 repaired step 0–90 + 恢复后的 120 和 150；被放弃的原始 91–116 排除。恢复后重复的 step-90 评测 415/426/11（原 415/428/13），作测试–复测证据，不替换。
- 【我的解释】两条曲线都在 ±3 的范围内波动，broken 臂没训练出工具使用也能有同样波动——这就是"没有检测到差异"的直观样子。

*策略确实更新了*：
- 150 步全部有非零 actor 梯度范数：broken 0.0120–0.0251，repaired 0.00641–0.01599。
- 平均训练奖励：broken 0.150–0.469，repaired 0.168–0.435。
- LoRA 张量直接审计（392 个张量，80,740,352 参数）：broken step120→150 有 45,471,572 个值改变（**56.3%**，ℓ2 delta 0.0787）；repaired step90→恢复后 step150 有 45,296,056 个（**56.1%**，ℓ2 delta 0.1222）。
- → 排除了"零结果是因为根本没训练"的解释。

*事件关联*：broken 19,203 个生成事件、8,689 tight、10 accepted/executed/observed；repaired 33,367 个生成事件、17,608 tight、16,912 accepted 调用在 16,844 个 accepted 生成事件里。40 个多调用生成 → 68 个余量（见 3.9）。
- **tight 和 accepted 是不同的检验**：16,323 个事件两者都满足，1,285 个 tight 但未被接受，521 个被接受但不 tight。
- 【我的解释】521 个"accepted 但不 tight"：parser 接受的调用不一定满足 tight 的严格条件（比如键顺序不同、代码里没有 `def` 等词法标记）；1,285 个"tight 但未接受"：形状像但 parser 不收（比如 JSON 坏了）。这说明 tight 是一个独立的分类器，不是 parser 的复制品。
- 验证脚本：`analysis/p3_final_evidence.py`（需要完整档案）、`analysis/p3_checkpoint_lora_audit.py`（服务端 checkpoint 审计）。

**G.11 历史 ReAct positive control（tab:sufficiency）**

| step | turn-1 | final | rescued by turn ≥2 | repair channel |
|---|---|---|---|---|
| 0 | 359 | 367 | 8 | 263 |
| 15 | 340 | 346 | **6** | 266 |
| 30 | 340 | 346 | **6** | 262 |
| 45 | 341 | 349 | **8** | 261 |
| 60 | 341 | 347 | **6** | 274 |

- 23,676 次执行的工具通道，rescue 也不动。
- 三个历史 1.5B FC 运行（**0** 次工具执行）的 rescue 序列：[7,9,6,6,7,6,10,8,9,8,6]、[8,7,6,7,7,7,7,5,6,6,8]、[7,5,7,5,7,8,9,9,8,9]。
- "This is supporting evidence only; Table p3formal is the single-variable control."
- 图 fig3：FC 下 tool 从未成功调用，`num_turns` 钉在 2，tool time 正好 0；prompt 强度跨臂匹配（都是 mandatory），**步数不匹配（10 vs 3）**。

**G.12 采样方差 + 倒置的噪声结构**

Llama-3.1-8B，temperature 0.6，n=100 × 3 seeds：

| | seed 1 | seed 2 | seed 3 | admissible |
|---|---|---|---|---|
| ReAct turn-1 | 53 | 56 | 43 | 3/3 |
| ReAct final | 72 | 72 | 72 | 3/3 |
| FC turn-1 | 44 | 44 | 45 | — |
| FC final | 62 | 66 | 57† | 2 clean / 3 端到端 |

- FC seed 3 有一个请求错误：模型发了并行工具调用，vLLM 返回 "This model only supports single tool-calls at once"。端到端终点把这题记为失败 → 57/100；预先指定的 clean-arm 分析排除这个 seed → [62, 66]。两个 clean seed 算出的 SD 不稳定，不用来刻画采样变异。
- clean 臂里最坏情况：ReAct 最低（72）vs FC 最高（66）= **+6**。
- **倒置的噪声结构**：ReAct turn-1 波动大（43–56，sd 6.8）而 final 收敛；FC turn-1 几乎恒定（44–45，sd 0.58）而 final 分散。机制是 rescue 数的补偿——19 / 16 / **29**，turn-1 最差的那个 seed rescue 最多。三个 seed 里"turn 1 过了但最终失败"的数都是 **0**：修复循环是单调的。
- "**The three identical finals are a coincidence of totals, not identical item sets**"，sd=0.00 不能读作"ReAct 完美稳定"。

**G.13 总数稳定但题目集变化（探索性）**
- 跨采样 seed：ReAct final 72/72/72，但解集两两 Jaccard 0.71–0.80，15–24 个题不同。
- 跨训练步：ReAct 运行 step 15→30 之间，542 个首轮完成里 85 个逐字改变，6 个题翻转结果（3 得 3 失），turn-1 和 final 数值完全相同（两个 checkpoint 都是 340/346）。"The policy is demonstrably moving; the aggregate is not."
- 只说明等总数可以掩盖不同的题目级结果；不识别修复的普遍稳定效应，也不证明部署收益。→ 报告题目级变化。

**G.14 训练 verifier**（见 2.9 Reward）。补充：held-out 通过率**不用**任何一个 verifier 路径，用 `mode="script"`、exit code 0、`status=="ok"`。

**G.15 Post-hoc common native FC evaluation（审稿后加的）**

*设计*：
- 用现有 base 和最终谱系 step-150 broken/repaired checkpoint，**不是新训练**。审稿后设计，在训练预注册之外。
- 三个权重看同样 542 个冻结 EvalPlus 题、原始 tokenizer/chat template、`run_tests` schema、同样的修好的 verl 抽取逻辑。**不用** BFCL few-shot adapter。
- 每个加载的 LoRA：392 张量、80,740,352 参数；有序 float32 张量摘要和审计过的 checkpoint 导出匹配；rank/alpha 32，scaling 1。base/tokenizer 配置哈希检查了；完整 base 权重分片没有独立重新哈希。
- 生成：顺序的 Transformers，bf16/SDPA，贪心，seed 0，最多 4 次 assistant 生成、每次 1,024 新 token，8,192 token 上下文预算，1,200 **字符**观测上限。上下文溢出显式失败，不静默截断。
- 后端：PyTorch 2.13.0、Transformers 5.10.4、PEFT 0.20.0、safetensors 0.8.0。
- 配置记录在公开仓库 `submission_packages/native_fc/configurations.json`。

*控制器规则*：
- 只有**被接受的、正确命名的、代码非空**的调用才触发交互执行。每次生成派发第一个被接受的调用（这里没出现多调用）。
- 反馈是 **tool-role** 消息，每轮重新渲染完整历史。
- 工具测试通过**不会**停止循环；无调用的回复或生成上限才结束。
- 之后显式的最终代码提交覆盖最后的工具代码；纯结束语不覆盖。
- 对话后，最终代码**独立评分一次**，不返回反馈（终端测试不是工具调用）。EvalPlus script mode，30 秒超时（不是训练奖励的 pytest 部分得分）。
- 对齐了 tool-role 反馈，但**不是** AgentLoop 的逐字节重放，也没有消除任务/测试分布偏移。

*结果（tab:nativefc）*：

| Weight | Final pass | Initiating tasks | Calls | Tool rescues | Timeouts |
|---|---|---|---|---|---|
| Base | 239 | 46 | 91 | 2 | 4 |
| Broken step 150 | 248 | 50 | 93 | 1 | 5 |
| Repaired step 150 | 248 | 47 | 90 | 1 | 6 |

- **rescue 定义不同**：这里要求"第一次**实际工具测试**失败，之后某次工具测试通过"。没被测试的首回复不算"之后被救回的失败"。所以和之前的自动反馈 rescue 指标**不可互换、不可合并**。
- 补充的描述性计数："失败的首次工具测试之后跟着通过的最终提交（包括没有交互重测的提交）"：2/2/1。例：broken 的 MBPP/612 三次工具测试失败，然后不再调用工具直接提交了通过的最终代码。
- 两种定义都**没有**显示 repaired 有更多 rescue。
- initiation = 至少一次被接受且执行的调用，不是所有尝试或像调用的输出。
- accepted 调用 = 执行 = 返回观测（91/93/90）；生成数 619/622/619；没有输出触及生成长度上限；每次交互执行状态都是 `ok`（表示测试**执行成功**，不一定代码正确）。
- 终端超时仍计为失败（完整 542 分母，无 clean-arm 过滤）。base 超时 MBPP 84、126、388、739；broken 额外 603；repaired 额外 603 和 765。这些超时**没有**被独立诊断为本质的不终止。
- 两个历史标记题：native FC 下 HumanEval/32 对所有权重都失败；MBPP/599 对 base 失败、对两个训练后权重**通过**。保留冻结的 542 题主分析；排除两题后 239/540、247/540、247/540，训练臂不一致仍是 16/16。这个敏感性**不**把任一测试重新归类为无缺陷。

*跨协议的绝对分差*：训练后权重在程序化反馈下 430/542 和 427/542，在 native FC 下都是 248/542。**这不是自动反馈收益的隔离估计**：后端、历史渲染、代码提交处理、停止规则都变了。只解读**协议内**的权重对比。

*统计*：
- repaired − broken：16/16，差 0pp，CI [−3.47, 3.47]，*p*=1。
- broken vs base：20 gains / 11 losses，+1.66pp，CI [−1.74, 5.03]，*p*=0.150。
- repaired vs base：22 gains / 13 losses，+1.66pp，CI [−1.94, 5.23]，*p*=0.175。
- 这些区间覆盖题目配对的不确定性，不覆盖训练种子或推理复测变异。等总数不意味着等通过集或等价策略。"Neither protocol detects a success benefit of repair in this setting; this **does not establish that interface repair cannot improve learning**."
- 9 个探索性检验（3 个权重对比 × final/initiation/rescue），连调整前都不显著；独立于 27-test 族；36 个合并修正也不会让任何新结果显著。区间是逐点的。确定性的 6 题和 24 题工程 pilot 与全集重叠，不算额外观察或种子。
- 公开仓库提供配置、1,626 条题目记录、`native_fc_results.py`（`--outcomes` 模式 CPU 复算；`--archive` 模式检查原始哈希、匹配输入、加载的 adapter 身份、执行到观测的链接）。精简记录不含生成代码或完整对话。

<a id="ch4-8"></a>

### 4.8 附录 H：Limitations, in full（17 条，"按怀疑型读者应该应用的顺序"）

**A. 证据范围**
1. **规模和训练结果只有一个任务家族**。外部效度结果不是代码专属（BFCL multi_turn_base 涵盖文件系统/车辆/交易 API；τ-bench 是客服），但规模 ladder 和 RL 结果是代码专属。前者是真缺口；后者不期望能补（reward 是 pytest 通过率，移植会同时改变环境、奖励和任务）。单工具 schema 不检验多工具间的工具选择；单工具是简单的 parser 场景，但只作为做更广测试的动机，不作定量下界论证。
   - 【注意】原文这里写 τ-bench 789 个未解析回合 "decompose to 100% envelope-layer with zero parser loss"——"zero parser loss" 是真的，但因为全部在 envelope 层就失败了，这个"0"在证据上是平凡的（见 2.3 首失败分类）。如果将来出 v3 可以改措辞。
2. **题目抽样**：`clean[:100]` 不是随机样本；配对是严格的所以研究内比较可靠。泛化靠随机 300 题复制（seed 20260901）：1500 题上 server-parsed 每个规模都是 0；严格 emitted 0, 4, 30, 40, 81 vs 原来 0, 4, 21, 36, 80。7B 21→30，原子集低估了那一档。
3. **规模结果只在一个家族内**。不声称 censoring 普遍随规模增长，不把参数量等同于能力。Instruct ladder 是对**机制**的控制，不是第二条欠计数 ladder——它的执行层最初是空的（主机缺 `pytest`），后来用能工作的执行器重跑，解析层逐值复现。
4. **Taxonomy 是四个实例，不是枚举**。证据故意不对称：Qwen 有规模 ladder、重放矩阵和修复；Llama 有单变量干预；Mistral 没有可准入的 100 题 FC 比较；DeepSeek 只是模板检查。
5. **单样本头条臂**：表 repair 和 ladders 是 temperature 0 的单样本。方差只在 Llama 上估计，且只有 2/3 FC 臂是 clean 的。
6. **多重比较**：27-test 回顾族，0.00185，非预注册。**通过的**：official+strict 错调用（4.77e-7）和 final（7.63e-5）、300 随机题的协议差距（3.1e-6）、official+strict turn-1（0.000977）。**没通过的**：Llama strict→ReAct（0.0019）、Llama both-parsed（0.0021）、修复循环 pass 增益（n=100 0.093，n=300 0.0385）、Llama n=300 协议对比（0.0352）、残差估计（0.118、0.125）。报告 n=300 值是为了让读者看到方向稳定、但不声称显著。
7. **模型规模止于 32B**，单张 A800，无前沿模型。

**B. 对因果解读的威胁**
8. **clean repaired-FC 控制只有一个种子一个预算**。识别了这个设定下接口对采样经验分布的效应；零 held-out 结果不证明工作通道普遍不足；LoRA 训练和历史 1.5B 全参不严格可比。
9. **条件化于成功解析是后处理条件化**（见 2.10(h)）。"**we do not detect a residual difference, which is not the same as establishing there is none**".
10. **接近零的观察频率不是零概率**。broken 7B 臂执行了 10 次而不是 0；仪器描述的是采样分布，不是策略的支撑集。所以说"几乎关闭的通道"，不是"数学上不可达的动作"。
11. **尝试和丢弃只在新运行中分开**。正式 7B 臂保留完整 rollout 文本 + 哈希、parser verdict、执行事件、返回观测，所以 8,689 个 regex 阳性 emission 和 10 个被接受的动作分开记录；分类器不是语义有效性神谕，它和派发的联系是轨迹级计数检查。历史 1.5B 没保留全文。
12. **零学习观察是设定特定的**。一个种子、7B、LoRA、一个代码任务、一个训练长度。旧 ReAct 臂方向一致但被协议变化混淆。共同评测把 tool-role 反馈换成 user-message 反馈 → 测的是向程序化反馈协议的迁移，不是 native FC 行为。格式偏移、有限的适配预算/容量、多轮信用分配都是没迁移的合理解释，数据分不开。评测器在有可解码的 `<tool_call>` code 参数时取最后一个，否则取最后一个代码块或原文——不是通用解码器。native FC 评测补上了发起测量，但不是所有分布偏移（script-mode 测试、控制器重新渲染历史、只有一个训练种子）；不是逐字节训练 rollout，不是等价性检验；两种 rescue 定义不能和程序化反馈 rescue 合并；跨协议绝对分差不能只归因于是否自动发起测试。
13. **人工验证只覆盖一个判据一个样本**（见 4.6）。
14. **历史训练 verifier 有两个可利用的缝隙**（见 2.9）。两臂用同一个修好的 verifier，不能解释对比。
**C. 环境依赖与 provenance**
15. **服务栈版本依赖**：vLLM 0.27.1、verl 0.9.0；parser 行为、模板默认值、`strict: true` 语义都是这些版本的属性。没在第二个服务栈（如 SGLang）上复制。native FC 用 Transformers/PEFT，不是第二个服务栈的复制。
16. **仓库修订依赖**：chat template 在 `tokenizer_config.json` 里，发布者可以不改模型就改模板；Qwen2.5-Coder 的假阳性正是继承模板的产物。
17. **仪器勘误，披露而非吸收**（见 4.1）。

**What would change our conclusions (restated)**：更多种子或更长预算；第二个家族的 ladder（带工作的执行器）；同样 ladder 在非代码任务上（五个 Coder checkpoint 跑 τ-bench retail，documented 配置——emitted 列随规模上升而 parsed 列保持 0 → 接口的属性；否则 → 代码形状 payload 的属性；小规模需要 rescue 臂，理由同 Granite）；更多标准基准（已做 BFCL v4 和 τ-bench；大 schema 或依赖参数级匹配的套件未测）。

<a id="ch4-9"></a>

### 4.9 附录 I：Statistical estimands and retrospective test family

**I.1 RL 终点的有限样本区间**（见 2.10(c)）。

**I.2 显式报告族**：
- **回顾性报告审计，不是预注册的多重性策略**。
- 包括：5 个被审计的 Llama 控制对比里的全部 15 个指标（即使只强调了一个）+ 12 个不同的比较。同一对比重复出现只算一次。
- 名义 *p* = 精确双侧 McNemar；调整 *p* = min(1, 27p)；阈值 0.05/27 = 0.00185185。
- 历史 1.5B 动机运行和被放弃/无效运行**不在族内**。
- 两个 both-parsed 对比保留在族内，但仍是后处理描述性比较。
- 这个显式族替代了早期"大约 12 个"的近似计数；特别地，Llama strict→ReAct 和 both-parsed 对比在修正阈值下**不再通过**。

**27 个检验（tab:testfamily）**——G/L 是后一个臂在该二元指标上的 gains/losses；错调用指标上 gain 表示**更多错调用**（不是改善）。★ = *p* ≤ 0.05/27：

| ID | 对比与指标 | n | G/L | 名义 *p* |
|---|---|---|---|---|
| T01 | Llama terse→rich: wrong_tool | 100 | 5/6 | 1.00 |
| T02 | Llama terse→rich: turn1_pass | 100 | 4/6 | 0.754 |
| T03 | Llama terse→rich: final_pass | 100 | 10/10 | 1.00 |
| T04 | Llama terse→thought: wrong_tool | 100 | 37/1 | 2.84e-10 ★ |
| T05 | Llama terse→thought: turn1_pass | 100 | 1/23 | 2.98e-6 ★ |
| T06 | Llama terse→thought: final_pass | 100 | 4/29 | 1.09e-5 ★ |
| T07 | Llama terse→official: wrong_tool | 100 | 0/1 | 1.00 |
| T08 | Llama terse→official: turn1_pass | 100 | 0/1 | 1.00 |
| T09 | Llama terse→official: final_pass | 100 | 1/6 | 0.125 |
| T10 | Llama terse→strict: wrong_tool | 100 | 0/23 | 2.38e-7 ★ |
| T11 | Llama terse→strict: turn1_pass | 100 | 13/1 | 0.00183 ★ |
| T12 | Llama terse→strict: final_pass | 100 | 15/3 | 0.00754 |
| T13 | Llama official→strict: wrong_tool | 100 | 0/22 | 4.77e-7 ★ |
| T14 | Llama official→strict: turn1_pass | 100 | 14/1 | 0.000977 ★ |
| T15 | Llama official→strict: final_pass | 100 | 18/1 | 7.63e-5 ★ |
| T16 | Llama strict→ReAct final | 100 | 27/8 | 0.00188 |
| T17 | Llama terse→ReAct final | 100 | 38/7 | 3.12e-6 ★ |
| T18 | Qwen Hermes→adapter final | 100 | 16/7 | 0.0931 |
| T19 | Qwen adapter→ReAct final | 100 | 17/5 | 0.0169 |
| T20 | Qwen Hermes→adapter final (random) | 300 | 43/25 | 0.0385 |
| T21 | Qwen adapter→ReAct final (random) | 300 | 59/18 | 3.06e-6 ★ |
| T22 | Llama strict→ReAct final (random) | 300 | 51/31 | 0.0352 |
| T23 | Qwen both-parsed FC→ReAct final | 83 | 11/4 | 0.118 |
| T24 | Llama both-parsed FC→ReAct final | 96 | 25/7 | 0.00210 |
| T25 | τ documented→repaired success | 115 | 3/0 | 0.250 |
| T26 | P3 broken→repaired final @150 | 542 | 0/3 | 0.250 |
| T27 | P3 repaired step0→150 final | 542 | 1/4 | 0.375 |

**读这张表的几个练习**：
- 数一数 ★：T04, T05, T06, T10, T11, T13, T14, T15, T17, T21 —— **10 个**。
- T11（0.00183）勉强过、T16（0.00188）勉强没过——差 0.00005。论文对两边都如实报告。
- T09（0.125）就是"之前被误引成错调用 *p* 的那个 final-pass *p*"。
- T05、T06 是 thought scaffold 让 turn-1 和 final **变差**（1/23、4/29），显著地变差。
- T25、T26 的 *p* 都是 0.25，但方向相反：τ-bench 3/0 朝修复方向；P3 0/3 朝 broken 方向。都是 3 个不一致对，都不能下结论。
- 复算工具：`submission_statistics.py`（从题目级记录复算 15 个 Llama 检验，其余从显式配对计数复算，来源在 `submission_statistics.json`）。count-based 复算不是对每个历史比较的新题目级审计。

---

<a id="ch5"></a>

## 5. 数字速查表

<sub>[↑ 回到目录](#toc) ｜ [← 上一章：4. 附录逐个精读](#ch4) ｜ [下一章：6. 代码与数据地图 →](#ch6)</sub>

> 按主题分组。"层"一列：E=emission，P=parse，X=execution，O=observation，R=result/outcome。

<a id="ch5-1"></a>

### 5.1 BFCL v4（Qwen2.5-Coder-7B-Instruct，Gorilla `6ea5797`）

| 数字 | 含义 | 层 | 出处 |
|---|---|---|---|
| 0.00 → 0.96 | simple_python 官方 accuracy（=96/100） | R | 表 2 |
| 0.00 → 0.19 | multi_turn_base 官方 accuracy（=19/100） | R | 表 2 |
| 0 / 0 / 0 / 196 | 2×2 首请求解析（n=200） | P | 表 1 |
| +98 / +98 | parser、template 的主效应（自己算的，论文不报） | P | — |
| 0 | 在基线处两个简单效应 | P | 表 1 |
| 166 / 169 / 187 | documented 臂未解析里满足 JSON-call / format-marker / JSON-marker 的数（重叠，不能相加） | E | tab:bfclfunnel |
| 0 | documented 臂 HTTP 错误 | — | tab:bfclfunnel |
| 97 / 92 | 固定字节离线重放：结构合规调用 / 通过官方 AST | P/R | G.4 |
| 96 | repaired 臂阳性对照复现的官方成功 | R | G.4 |
| 98 / 19 | multi_turn 修复后执行 / 成功 | X/R | 表 2、tab:bfcl5 |
| 69 | multi_turn hermes 臂 tight（未解析） | E | tab:bfcl5 |
| 197 vs 196；99 vs 98 | repaired 臂两次运行的 first parsed / executed（±1） | P/X | G.4 |
| 480 | hermes 臂请求数（两次一样） | — | G.4 |

<a id="ch5-2"></a>

### 5.2 τ-bench retail（115 任务，本地 Llama-3.1-8B 用户模拟器）

| 数字 | 含义 | 层 |
|---|---|---|
| 1,389 vs 1,971 | assistant 回合数 | — |
| 0 → 636 | 解析并执行的调用 | P/X/O |
| 789 vs 6 | 有 call-shaped payload 但未解析的回合 | E |
| 789/789 | 归入 missing-envelope（都没到 payload 验证） | P |
| 0 → 103 | 至少执行过一次的任务 | X |
| 7 → 10 | solved，3/0，*p*=0.25，不声明 | R |

<a id="ch5-3"></a>

### 5.3 规模

| 数字 | 含义 |
|---|---|
| 0, 4, 21, 36, 80 | Coder 1.5/3/7/14/32B tight 未解析 /100（`clean[:100]`） |
| 0, 4, 30, 40, 81 | 随机 300 题复制 /100 |
| ≈72 | 32B 敏感性估计（80×0.900） |
| 0, 3.6, 18.9, 32.4, 72 | 统一乘 0.9（按构造保持趋势，不验证趋势） |
| 0/6, 0/13, 0/15, 0/24, 10/40 | 分层样本里各规模 4,000 字符截断命中 |
| 0, 0, 1, 0, 18 | 全部 100 个首输出里各规模截断命中 |
| 0, 5, 22, 42, 100 | bare-JSON 离线重解析 |
| 86, 44, 70, 97 | Qwen3 0.6/1.7/4/8B parsed（预注册、matched） |
| 1, 1, 2, 0 | Qwen3 静默数 |
| 1, 11, 63, 77, 88 | Instruct parsed |
| 13/13, 14/17, 34/40, 38/52, 42/65 | Instruct turn-1/final |
| 0, 3, 6, 14, 23 | Instruct rescues |
| 31, 37, 53, 53, 68 | Coder turn-1 = final |
| 4,254 | 去重首轮数，parser loss = 0 |
| 84 vs 1 | 同一个 adapter 在 7B vs 3B 的恢复数 |

<a id="ch5-4"></a>

### 5.4 失败层（no envelope / malformed / strictness / parser loss）

| Arm | 数字 |
|---|---|
| Coder 每个规模 | 100 / 0 / 0 / 0 |
| Instruct 1.5B | 93 / 6 / 0 / 0 |
| Instruct 3B | 22 / 41 / 26 / 0 |
| Instruct 7B | 3 / 33 / 1 / 0 |
| Instruct 14B | 13 / 6 / 4 / 0 |
| Instruct 32B | 1 / 9 / 2 / 0 |

<a id="ch5-5"></a>

### 5.5 Llama-3.1-8B

| 数字 | 含义 |
|---|---|
| 23 | terse FC 错调用 |
| 22（6/5，*p*=1） | rich schema |
| 59（*p*=2.84e-10） | thought scaffold |
| 22（1/0，*p*=1） | official template |
| 0（22/0，*p*=4.77e-7） | official + strict |
| 73 → 97 | 有效执行（official → +strict） |
| 33 → 46 | turn-1 |
| 44 → 61（18/1，*p*=7.63e-5，★） | final，+17 |
| 49 → 61（15/3，*p*=0.0075） | terse → strict，+12 interface |
| 61 → 80（27/8，*p*=0.0019） | strict → ReAct，+19 protocol |
| 49 → 80（38/7，*p*=3.1e-6，★） | terse → ReAct，+31 |
| 97 → 74 | initiation 修正 |
| 23/23 | 定向复跑复现错调用 |
| 96 题 78 vs 60（25/7，*p*=0.0021） | both-parsed |
| 72/72/72 | ReAct final 三个 seed（Jaccard 0.71–0.80） |
| 53/56/43 | ReAct turn-1 三个 seed（sd 6.8） |
| 44/44/45 | FC turn-1（sd 0.58） |
| 62/66/57† | FC final（seed 3 有一个并行调用请求错误） |
| 19/16/29 | ReAct 三个 seed 的 rescue |
| +6 | clean 臂最坏情况的协议差 |

<a id="ch5-6"></a>

### 5.6 Qwen2.5-Coder-7B 评测修复

| 数字 | 含义 |
|---|---|
| 0 → 84 | parsed |
| 0 → 37 | ≥2 tested steps |
| 0 → 9 | rescues |
| 53 → 62（16/7，*p*=0.093） | final，n=100 |
| 160 → 178（43/25，*p*=0.0385） | final，n=300 随机 |
| 53 = 53 | turn-1（一致性检查） |
| 95 / 40 / 57/74 / 17 | ReAct reference：parsed / ≥2 / turn-1/final / rescue |
| 17/5，*p*=0.0169 | adapter → ReAct（n=100） |
| 59/18，*p*=3.06e-6（★） | adapter → ReAct（n=300） |
| 83 题 63 vs 56，+8.4 [−0.5, 17.4]，*p*=0.118 | both-parsed 残差 |
| 42.5% vs 24.3% | 给定反馈的修复成功率（后处理，描述性） |
| 0.96 / 1.82 / 1.92 | mean tested steps：hermes / adapter / ReAct |

<a id="ch5-7"></a>

### 5.7 正式 7B RL（P3）

| 数字 | 含义 |
|---|---|
| 150 步，16×8，seed 0 | 规模 |
| LoRA r=α=32，all-linear，scaling 1 | 配置 |
| 392 张量，80,740,352 参数 | LoRA 规模 |
| lr 1e-6，temp 1.0，prompt 2,048 / response 6,144，≤5 轮，tool response ≤1,200 token，1 调用/生成 | 训练预算 |
| 4 轮 × 1,024 token，temp 0，user 消息反馈 | 评测协议 |
| 542（164 HumanEval+ + 378 MBPP+） | 评测题数 |
| 19,203 / 8,689 / 10 / 10 / 10 | broken：生成事件 / tight / accepted / executed / observed |
| 33,367 / 17,608 / 16,912 / 16,844 | repaired：生成事件 / tight / accepted 调用 / accepted 生成事件（= 派发 = 执行 = 观测） |
| 40 个生成，68 个多余调用 | 多调用 |
| 16,323 / 1,285 / 521 | tight∩accepted / tight-only / accepted-only |
| 430 → 430 / 12 → 12 | broken final / rescue（step 0→150） |
| 430 → 427 / 12 → 12 | repaired |
| −0.55pp，[−1.79, 0.72]，0/3，*p*=0.25 | repaired vs broken @150 |
| 1/4，*p*=0.375，[−2.06, 1.01] | repaired step0 vs 150 |
| 418/430/12 | step 0 turn-1/final/rescue（两臂相同） |
| 415/428/13 vs 415/426/11 | step 90 原始 vs 恢复后复测 |
| step 116 中断，从 step 90 恢复，排除 26 次更新的分支 | resume |
| 0.0120–0.0251 / 0.00641–0.01599 | 梯度范数范围（broken / repaired） |
| 0.150–0.469 / 0.168–0.435 | 平均训练奖励范围 |
| 56.3%（ℓ2 0.0787）/ 56.1%（ℓ2 0.1222） | LoRA 值改变比例（broken 120→150 / repaired 90→150） |

<a id="ch5-8"></a>

### 5.8 Native FC 评测（post-hoc）

| 数字 | 含义 |
|---|---|
| 239 / 248 / 248 | base / broken / repaired final |
| 16/16，0pp，[−3.47, 3.47]，*p*=1 | repaired vs broken |
| 20/11，+1.66，[−1.74, 5.03]，*p*=0.150 | broken vs base |
| 22/13，+1.66，[−1.94, 5.23]，*p*=0.175 | repaired vs base |
| 46 / 50 / 47 | 发起任务数 |
| 91 / 93 / 90 | 调用 = 执行 = 观测 |
| 2 / 1 / 1 | tool-test rescue |
| 2 / 2 / 1 | 补充：失败首测后最终提交通过 |
| 4 / 5 / 6 | 终端超时 |
| 619 / 622 / 619 | 生成数 |
| 239/540, 247/540, 247/540 | 排除两个标记题的敏感性 |
| 1,626 | 精简题目记录（542×3） |
| 9 | 探索性检验数（另一个族） |
| 8,192 token 上下文，1,200 **字符**观测上限，30s 超时 | 配置 |

<a id="ch5-9"></a>

### 5.9 历史 1.5B

| 数字 | 含义 |
|---|---|
| +2.6–2.8，*p*=0.064/0.078/0.077 | pass 增益（不显著） |
| 91–94% | 新通过题里 turn-1 就过的比例 |
| 5–10 / 540 | rescue 范围 |
| GRPO×2 + RLOO×1 | 算法 |
| 10 步，2.000，0.00s，0.000 | FC 诊断：步数、num_turns、tool time、调用/rollout |
| 3 步，5.883，12.21s，2.052 | ReAct 诊断 |
| 0 行 vs 788 行 | 计数器文件 |
| 23,676 | 旧 ReAct 臂执行次数 |
| 359/367/8 → 341/347/6 | 旧 ReAct 曲线 step 0 → 60 |
| 52 → 64，0 命名，15 → 13 | verl parser 恢复 / role-disambig / 正确命名 / final |
| 36 / 16 | 任务函数参数 / 代码在别处 |
| 31 → 15，~0 → 46 | 1.5B optional→mandatory：final / 无法解析 |

<a id="ch5-10"></a>

### 5.10 人工验证

| 数字 | 含义 |
|---|---|
| 98（40/28/30） | 分层样本 |
| 0.713，86.7%，precision 70.0%，recall 96.6% | 第一轮 |
| 2,400 | 第一轮标注包截断字符数 |
| 11 + 1 | 第一轮 12 个 FP：阅读顺序 + 仪器失败 |
| 13/13，10 个全朝分类器 | 第一轮单边裁决 |
| 0.736 [0.595, 0.866] | A1–A2 全部 97 题 |
| 0.967 [0.893, 1.000]，61/62 | A1–A2 同字节 62 题 |
| 15，6/15 | 三方裁决题数，偏离分类器次数 |
| 0.871 [0.756, 0.958] | 分类器 vs 金标准 |
| 36/40 = 0.900 | tight 层精度 |
| 4 FP，2 FN | 金标准样本 |
| 0.950 → 0.975 → 0.900 | 精度估计历史 |
| 10/98 | 样本中 4,000 字符触顶数（全在 32B） |
| 5,000 次，seed 7 | bootstrap |

<a id="ch5-11"></a>

### 5.11 杂项

| 数字 | 含义 |
|---|---|
| 50 | 完整臂数（全部 `clean[:100]`） |
| 7 | 记错 max_tokens 的 ReAct 臂（记 2048 实际 1024） |
| 5 | 不可准入臂（Mistral 2/42/3/39，Llama t06_s3 1） |
| *p*=0.648 | 撤回的 *p* |
| 27，0.00185185 | Bonferroni 族与阈值 |
| 10 | 族中通过阈值的检验数 |
| 17 | 局限条数 |
| 7,669 | KodCode 去污染池 |
| 8/5，10/6，5-gram，0.10 | 去污染参数 |
| 20260901 | manifest / 随机抽样 seed |
| 1b92150 | hanXen parser commit |
| 2% → 42% | Mistral 官方 parallel template 错误率 |
| 9 | Mistral `tool_call_id` 必须的字符数 |
| 98 行 | preflight 脚本 |
| vLLM 0.27.1，verl 0.9.0 | 主栈 |
| PyTorch 2.13.0，Transformers 5.10.4，PEFT 0.20.0，safetensors 0.8.0 | native FC 栈 |

---

<a id="ch6"></a>

## 6. 代码与数据地图

<sub>[↑ 回到目录](#toc) ｜ [← 上一章：5. 数字速查表](#ch5) ｜ [下一章：7. 常见误读与正确说法 →](#ch7)</sub>

> 都是仓库 `code_agent/` 下的相对路径。✅ = 在公开仓库里；📦 = 只在补充包/完整档案里。

<a id="ch6-1"></a>

### 6.1 服务与探针

| 文件 | 作用 |
|---|---|
| `preflight_toolcall.py` ✅ | 98 行 conformance 预检：规范请求 → 验证 name/arguments → `tool_choice: required` 再来一遍 |
| `probe_toolcall.py`、`probe_react_full.py`、`probe_multiturn.py` ✅ | KodCode 受控探针（FC / ReAct / 多轮）；`runs/final/probe_react_full.py` 是定稿版本 |
| `validate_arms.py` ✅ | 臂准入检查（100 题、rc=0、0 错误、provenance） |
| `runs/final/plugin/` ✅ | hanXen parser（未改）+ template（已改）+ template.orig（原版） |
| `runs/final/ERRATA.md` ✅ | 附录 A 的来源 |
| `runs/final/final_table.py` 📦 | 主表生成（需完整档案） |
| `decontaminate.py`、`clean_ids.json` ✅ | 去污染 |

<a id="ch6-2"></a>

### 6.2 分析脚本（`analysis/`）

| 文件 | 对应论文内容 |
|---|---|
| `failure_layer.py`、`failure_layer_table.py` | 逐行重放 vLLM hermes 抽取器，生成 tab:layer |
| `reparse_matrix.py` | 四规则离线重解析，tab:reparse |
| `intent.py` | tight/strong/weak 分类器（完整档案才能跑全量） |
| `bfcl_fixed_replay.py`（+ `test_bfcl_fixed_replay.py`） | BFCL 固定字节重放（97/92） |
| `submission_statistics.py` + `.json` | 27-test 族复算 |
| `native_fc_results.py` | native FC 配对对比与超时清单 |
| `p3_final_evidence.py` | P3 哈希、谱系重建、per-file manifest（完整档案） |
| `p3_checkpoint_lora_audit.py` | LoRA 张量改变比例审计（服务端） |
| `verify_supplement.py` | 补充包验证 |
| `llama_paired_audit.py` | Llama 配对审计 |
| `funnel.py`、`funnel_data.json` | 图 1 漏斗 |
| `MECHANISM_EVIDENCE.md`、`VERIFICATION_SCOPE.md` | 机制证据和验证范围说明 |

<a id="ch6-3"></a>

### 6.3 训练

| 文件 | 作用 |
|---|---|
| `qwen_tools_parser.py` | verl 侧 `qwen2_5_coder` parser（`<tools>` / ```json / 裸 JSON 三种依次尝试，同段文本只按第一种命中的格式解析，避免重复计数） |
| `code_tool.py`（`CodeTool.execute`） | FC 路径的工具，**计数器放这里** |
| `react_agent_loop.py`（`ReActAgentLoop._run_tests`） | ReAct 路径，**计数器放这里** |
| `sandbox.py` | 沙箱 pytest；**计数器不能放这里**（reward 也走它），自带的叫 `sandbox_exec_count` |
| `reward_code.py`（`extract_final_code`、`compute_score`） | outcome-only pytest 通过率奖励 |
| `p3/launch_arm.py`、`p3/run_p3_arm.sh`、`p3/run_p3_pair.sh` | 正式 7B 两臂启动（唯一差异 `multi_turn.format`） |
| `p3/parser_diagnostics.py`、`p3/summarise_p3.py` | P3 事件统计 |
| `p3/native_fc_eval.py`（+ tests） | native FC 评测控制器 |
| `p3/FINAL_RESULT_20260918.md`、`p3/FINAL_EVIDENCE_AUDIT_20260919.md`、`p3/PREREG_AMENDMENT_20260903.md` | P3 结果、审计、预注册修正 |
| `submission_packages/native_fc/{configurations,outcomes,statistics}.json` ✅ | native FC 配置与 1,626 条记录 |

<a id="ch6-4"></a>

### 6.4 图

`figures/make_figs.py` 生成：`fig0_funnel`（图 1 漏斗）、`fig1_intent_parse_gap`（规模）、`fig2_family_taxonomy`（四家族）、`fig3_training_causal`（历史 1.5B 诊断）、`fig4_training_curves`（历史曲线）。

<a id="ch6-5"></a>

### 6.5 论文源码

- 投稿版：`paper/iclr2027/`（未改动）。
- arXiv v2：`paper/iclr2027/arxiv_v2/source/`（`main.tex` + `sections/`），打包 `arxiv_v2.tar.gz`。
- 中文 prompt 原文：`paper/sections/C_prompts.tex`。

---

<a id="ch7"></a>

## 7. 常见误读与正确说法

<sub>[↑ 回到目录](#toc) ｜ [← 上一章：6. 代码与数据地图](#ch6) ｜ [下一章：8. 自测题（附答案） →](#ch8)</sub>

| # | ❌ 误读 | ✅ 正确说法 | 为什么 |
|---|---|---|---|
| 1 | "vLLM 的 hermes parser 有 bug" | "契约不匹配：parser 正确地拒绝了不符合它语法的字节；4,254 个去重首轮里 parser loss = 0" | 见 G.3 |
| 2 | "两个主效应为零" | "在文档基线处，单换任一组件恢复为 0（简单效应为 0）；两个都换恢复 196/200（强互补）" | 主效应各 +98 |
| 3 | "BFCL 分数从 0 提到 96 分" | "官方 scorer 下 accuracy 0.00 → 0.96（96/100），simple_python 子集" | 指标和尺度要讲清 |
| 4 | "我们发现了 Qwen2.5-Coder 的兼容性问题" | "这个不兼容在 vLLM issue #32926 已被报告；我们的贡献是量化、分解、跟进到交互评测和 RL" | 附录 D |
| 5 | "τ-bench 上 0 个 payload 是坏的" | "789 个全部在 missing-envelope 层失败，没有一个到达 payload 验证阶段" | 首失败分类 |
| 6 | "修复接口提升了模型性能" | "修复恢复了机制（解析/执行/观测），没检测到结果提升" | *p*=0.093 / 0.25 / 1 |
| 7 | "RL 证明了修复接口没用" | "单种子、150 步、7B LoRA 下未检测到 held-out 提升；不是等价性结论，不证明修复无法改善学习" | 局限 8、12 |
| 8 | "broken 臂的通道完全关闭" | "几乎关闭：8,689 个 tight 生成事件只对应 10 次执行" | 局限 10 |
| 9 | "broken 臂没有工具经验，为什么评测还有 12 个 rescue？矛盾" | "评测协议自动执行代码、以 user 消息给反馈，不依赖模型发起调用" | §5.5 |
| 10 | "censoring 随规模增长" | "在一个家族（Qwen2.5-Coder）、一个错配接口、一个任务内，classifier-positive 的未解析 emission 随规模增长；不是 scaling law" | §5.2 |
| 11 | "32B 至少有 80 个真实调用被吞" | "80 是 tight 分类器计数；截断造成假阴性、精度 0.9 造成假阳性，净偏差无法识别；≈72 是依赖假设的敏感性估计" | 附录 F |
| 12 | "Qwen3 证明了 envelope 匹配就不会有问题" | "与契约错配驱动欠计数一致，但家族和 envelope 一起变了，不能隔离 envelope；不是严格规模反事实" | §3、§5.2 |
| 13 | "turn-1 53 = 53 证明了失败在 parser 层" | "只是一致性检查；adapter 同时改了 template 和 few-shot。层定位靠固定字节重放和事件追踪" | 签名 1 |
| 14 | "strict:true 找回了被删掉的调用" | "strict 是 constructive suppression：在生成时阻止错误动作，改变的是 emission 本身" | §1 |
| 15 | "Llama 的 ReAct 优势 +19 是显著的协议效应" | "*p*=0.0019，刚好没过 0.00185；描述性分差" | 附录 I T16 |
| 16 | "加强 prompt 能让模型更愿意调用工具" | "在坏接口上加压适得其反：1.5B mandatory 让 final 31→15，调用仍 0；thought scaffold 让错调用 23→59" | G.9 |
| 17 | "both-parsed 子集上 Qwen 没有残差，说明协议等价" | "后处理条件化；+8.4 [−0.5, 17.4] 未解决，没检测到 ≠ 不存在" | 局限 9 |
| 18 | "27 个检验的 Bonferroni 是预注册的" | "回顾性报告审计，非预注册" | 附录 I |
| 19 | "7B LoRA 结果可以和历史 1.5B 全参曲线直接比" | "不能严格比较；两臂都用 LoRA 是因为 7B 全参单卡 OOM" | §5.5 |
| 20 | "repaired 臂事件多，所以比率更低/更高说明……" | "事件总数不同是因为修复改变了轨迹；每臂比率只描述该臂漏斗，因果归因靠单一 parser 干预" | §5.5 |
| 21 | "`timing_s/agent_loop/tool_calls` 是 0，所以没调用" | "那是秒数不是次数；我们在 `CodeTool.execute` 和 `_run_tests` 自己埋了计数器" | 附录 D |
| 22 | "历史 1.5B 的 0 执行证明了 parser masking" | "过度决定：1.5B 根本没产出正确命名的调用（52 个可恢复里 0 个叫 `run_tests`），只作动机" | §1、G.9 |
| 23 | "native FC 下 248 vs 248，说明两个策略一样" | "16/16 不一致；等总数 ≠ 等通过集 ≠ 等价策略" | G.15 |
| 24 | "程序化反馈 430 vs native FC 248，差的 180 题是自动反馈的收益" | "后端、历史渲染、提交处理、停止规则都变了；只解读协议内对比" | G.15 |
| 25 | "ReAct final 72/72/72，非常稳定" | "总数巧合，解集 Jaccard 0.71–0.80，15–24 题不同" | G.12 |
| 26 | "τ-bench 分数说明 Qwen 在 τ-bench 上的水平" | "用户模拟器是本地 Llama-3.1-8B 不是 gpt-4o；绝对分不可与排行榜比，只看配对差" | §4 |
| 27 | "documented 配置 = 官方保证兼容" | "documented 指配置选择的来源（Qwen2.5 家族级文档），不是 Coder 专属保证，也不是部署普遍性的证据" | 附录 D |

---

<a id="ch8"></a>

## 8. 自测题（附答案）

<sub>[↑ 回到目录](#toc) ｜ [← 上一章：7. 常见误读与正确说法](#ch7) ｜ [下一章：9. 术语表 →](#ch9)</sub>

> 建议先遮住答案作答。分三级：⭐ 基础、⭐⭐ 理解、⭐⭐⭐ 审稿人级追问。

<a id="ch8-1"></a>

### ⭐ 基础

**Q1. 用一句话定义 interface-induced trajectory censoring。**
<details><summary>答案</summary>模型输出和把输出转成动作的接口（template + parser）之间的错配，导致字节里存在的工具意图在结构化轨迹里消失；服务端返回 HTTP 200 + 空 tool_calls，和"模型不调用"不可区分。</details>

**Q2. masked 的形式定义是什么？**
<details><summary>答案</summary>C(Y)=1 但 P(Y)=∅：独立判据认为字节里有工具意图，但配置的 parser 什么也没返回。</details>

**Q3. 漏斗的六层是哪六层？**
<details><summary>答案</summary>emission → serialization → parse → execution → observation → score/reward。</details>

**Q4. BFCL 的 0.96 是什么指标？分母是多少？**
<details><summary>答案</summary>官方 scorer 的 accuracy（correct_count/total_count），simple_python 子集 100 个 case，即 96/100；AST checker 打分。</details>

**Q5. BFCL 2×2 四个格子是多少？**
<details><summary>答案</summary>Hermes parser × documented template = 0；Hermes × dedicated template = 0；dedicated parser × documented template = 0；dedicated × dedicated = 196（/200 首请求解析）。</details>

**Q6. Coder ladder 的五个数？Qwen3 ladder 的静默数？**
<details><summary>答案</summary>0, 4, 21, 36, 80（/100，tight 未解析）；Qwen3 静默 1, 1, 2, 0。</details>

**Q7. P3 两臂唯一的差异是什么？**
<details><summary>答案</summary>verl 注册的 parser：`multi_turn.format=hermes` vs 注册的 `qwen2_5_coder`。</details>

**Q8. P3 的执行次数和 held-out 结果？**
<details><summary>答案</summary>执行 10 vs 16,844；step 150 final 430 vs 427，0/3 不一致，*p*=0.25，CI [−1.79, 0.72]；rescue 都是 12→12。</details>

**Q9. Llama 的 23% 错调用是什么错？**
<details><summary>答案</summary>把题目要求实现的函数（如 can_form_word）当工具来调用，参数是那个函数的参数，而不是调用 run_tests(code=...)——角色混淆。</details>

**Q10. Bonferroni 阈值是多少，怎么来的？**
<details><summary>答案</summary>0.05/27 ≈ 0.00185；27 = 5 个 Llama 控制对比 × 3 个指标（15）+ 12 个其他对比；回顾性、非预注册。</details>

<a id="ch8-2"></a>

### ⭐⭐ 理解

**Q11. 为什么"两个主效应为零"是错的？请给出主效应、简单效应、交互的数值。**
<details><summary>答案</summary>主效应 = 在另一因素各水平上平均：parser 主效应 = (0+196)/2 − (0+0)/2 = 98，template 同理 98。简单效应在基线处：0−0 = 0。交互 = (196−0) − (0−0) = 196。正确说法：基线处两个简单效应为 0，联合替换强互补。</details>

**Q12. 为什么 parser loss = 0 不能推出"Coder 的 payload 都是合法 JSON"？**
<details><summary>答案</summary>failure-layer 是首失败分类，Coder 100% 在 no-envelope 层就返回了，payload 层根本没被检验。parser loss = 0 只对带 envelope 的输出（主要是 Instruct）是真正的检验。</details>

**Q13. 为什么 Qwen 修复后 turn-1 53 = 53 不能用来定位失败层？什么能？**
<details><summary>答案</summary>签名 1 说 parser-only 干预不改变已生成字节、所以 turn-1 不变；但 dedicated adapter 同时改了 template 和 few-shot，理论上会改变 turn-1 字节，所以相等只是一致性检查。定位靠固定字节离线重放（BFCL 97/92、reparse matrix）和事件追踪。</details>

**Q14. strict:true 和换 parser 有什么本质区别？**
<details><summary>答案</summary>换 parser 是对已有字节的重新解释（masking 的修复）；strict 是约束解码，在生成时阻止违反 schema 的 token，改变 emission 本身（constructive suppression），所以能改变 turn-1（33→46）。</details>

**Q15. 为什么 broken 训练臂在评测里也有 12 个 rescue？**
<details><summary>答案</summary>held-out 评测故意绕开训练 parser：通过 native rollout RPC 查询，生成的代码被自动执行、测试输出作为 user 消息追加。rescue 不依赖模型发起调用。</details>

**Q16. 保守配对区间 [−1.79, 0.72] 是怎么构造的？覆盖什么、不覆盖什么？**
<details><summary>答案</summary>对 p₊（只 repaired 过）和 p₋（只 broken 过）各算 97.5% Clopper–Pearson 区间，用 union bound 得 [L₊−U₋, U₊−L₋]，至少 95% 覆盖。只覆盖题目配对抽样不确定性（假设独立）；不覆盖训练种子、推理复测、基准选择、任务簇；是逐点不是同时区间。</details>

**Q17. 为什么 80 不是下界？**
<details><summary>答案</summary>4,000 字符截断造成假阴性（向上的真实值），分类器精度 0.900 造成假阳性（向下的真实值），两个方向相反，净偏差无法识别。</details>

**Q18. 第一轮人工验证 κ 从 0.713 到 0.936，为什么不能当可靠性证据？**
<details><summary>答案</summary>裁决是单边的：被复查的 13 题恰好是标注者和分类器不一致的那些，10 个推翻全朝分类器方向，在这种选择规则下一致性只能上升。所以换成第二个盲标注者 + 预先固定的"任何三方不一致都复查"规则。</details>

**Q19. 同一个 adapter 在 7B 恢复 84、在 3B 只恢复 1，说明什么？**
<details><summary>答案</summary>适配器不是万能开关，效果随规模非单调；3B 完全无视 `<tools>` few-shot（0 个标签，99 个直接写代码）。强化"兼容性要按 checkpoint 验证"的主张。</details>

**Q20. 为什么事件总数（19,203 vs 33,367）不同不影响因果归因？**
<details><summary>答案</summary>事件数不同是因为修复改变了轨迹（调用执行后对话更长、更多机会再调用）。论文只用每臂内比率描述漏斗，因果归因用单一注册 parser 干预，不要求原始分母相等。</details>

**Q21. "Pressure on a broken interface makes things worse" 的两个证据？实践启示？**
<details><summary>答案</summary>(1) FC + thought scaffold：Llama 错调用 23→59；(2) 1.5B optional→mandatory：调用仍 0，final 31→15，无法解析 ~0→46。启示：agent 看似不愿用工具时，加强指令是最便宜也最可能误导的尝试——可能降低任务表现而调用数仍为 0，看起来"既不愿意又没能力"。</details>

**Q22. Qwen2.5-Instruct 控制排除了什么解释？**
<details><summary>答案</summary>排除了"同一 template/parser 组合本身就坏了"：同样接口下 Instruct 能被解析（1→88）且 rescue 随之上升（0→23），Coder 始终 0。差异在 checkpoint 特定的后训练，不在参数量或共享家族/模板；但不能认定工具格式训练是唯一原因。</details>

<a id="ch8-3"></a>

### ⭐⭐⭐ 审稿人级追问

**Q23. "你的 bare-JSON / dedicated 抽取器只是比 hermes 好，这不叫 censoring。"怎么回应？**
<details><summary>答案</summary>(1) 论文不声称 parser 有 bug，定义 masked 只需 C(Y)=1 且 P(Y)=∅；(2) 重解析矩阵对同样存储的字节用四种规则，hermes 重放每个规模都是 0——模型从不产生那个格式；(3) 固定字节 BFCL 重放：answer-blind、不修复坏 JSON，97 个结构合规、92 个过官方 AST，说明正确调用就在原始字节里；(4) 论文结论是"接口是测量函数的一部分"，不是"hermes 不好"。</details>

**Q24. "RL 零结果说明你的主线（接口影响 RL 经验）不重要。"**
<details><summary>答案</summary>论文把两个主张分开：经验接触（事件计数，10→16,844，确定性识别）和行为改变（held-out，未检测到）。前者成立；后者在单种子、150 步、7B LoRA、一个代码任务下未检测到，可能原因包括 tool-role→user-message 格式偏移、适配容量/预算、多轮信用分配，数据分不开。另外策略确实更新了（梯度非零、56% LoRA 值改变）。不是等价性结论。</details>

**Q25. "程序化反馈评测不测自愿发起，所以对 FC 训练不公平。"**
<details><summary>答案</summary>论文承认这点并加了 post-hoc native FC 评测：tool-role 反馈、自愿调用、同样 542 题。结果 248 vs 248，16/16，*p*=1，发起 47 vs 50，rescue 1 vs 1。仍是 post-hoc、单种子、非逐字节训练重放，不是等价检验；9 个探索检验单独成族。</details>

**Q26. "`clean[:100]` 不是随机样本，结果可能有偏。"**
<details><summary>答案</summary>承认（局限 2）。随机 300 题复制（seed 20260901）：server-parsed 1500 题全 0；tight 0, 4, 30, 40, 81 vs 0, 4, 21, 36, 80；7B 档移动了（单个 100 题子集对中间档不稳定），但方向和单调性不变。50 个臂共享同一集合，所以研究内配对比较可靠。</details>

**Q27. "Qwen3 和 Coder 是不同家族，你怎么知道是 envelope 而不是家族？"**
<details><summary>答案</summary>不知道，论文明说不能隔离 envelope（家族和 envelope 一起变了），只是"与契约错配驱动一致"的支持性跨家族观察。更强的证据是同谱系 Instruct 控制（同接口不同 checkpoint）和"what would change"里提出的第二家族 ladder、非代码 ladder。</details>

**Q28. "Bonferroni 很保守，你为什么还要用？为什么不预注册？"**
<details><summary>答案</summary>回顾性报告审计，目的是诚实标出哪些结论经得起族修正（10 个），哪些不能（如 T16 0.00188）；不把任何结论建立在没通过的对比上。因为是事后整理，明确说非预注册，不冒充。新的 native FC 检验单独成族，合并成 36 个也不改变结论。</details>

**Q29. "历史 verifier 有漏洞，你的 RL 奖励可能被 hack。"**
<details><summary>答案</summary>两个缝隙（未锚定 `(\d+) passed`、skipped/xfail 不计分母）在历史日志里审计 0 次出现，奖励曲线无阶跃；正式 7B 前已修复（锚定 summary、分母含 skipped/xfailed/xpassed、全文保留 + SHA-256），两臂用同一个修好的 verifier，不能解释对比；held-out 也不用 verifier（script mode、exit 0、status ok）。</details>

**Q30. "resume 了一次，lineage 是不是脏的？"**
<details><summary>答案</summary>step 116 后中断，从验证过的 step-90 恢复；核对了模型、optimizer、scheduler、RNG、dataloader、源码、配置、weight-step 哈希；正式事件用 1–90 + 恢复后 91–150，排除被放弃的 26 次更新分支；重复 step-90 评测差 2 题，作为测试–复测变异报告、不替换曲线；每个评测请求核验加载的 global weight step。</details>

**Q31. "为什么不直接在 RL 里用约束解码？"**
<details><summary>答案</summary>verl 0.9.0 的 vLLM rollout 路径不暴露（唯一的 strict key 属于 profiler）。论文说这是被测 rollout 路径的限制，不是反对约束解码的论据；约束解码改变输出分布，是另一种补救，本身也不保证参数正确或任务成功。</details>

**Q32. "精度估计变了三次（0.950→0.975→0.900），是不是在挑数字？"**
<details><summary>答案</summary>正相反：论文主动报告整个轨迹而不是只报终点。0.950 被单边选择规则抬高；0.975 只依赖单个评分者；0.900 是唯一由事先固定规则构建的金标准，而且是最低的那个——用最保守的。</details>

**Q33. 如果让你设计下一个实验来"推翻"规模结论，你做什么？**
<details><summary>答案</summary>论文已写：(1) 第二个家族带工作执行器的 ladder——欠计数不随规模增长则规模结论只是 Qwen+hermes 的事实；(2) 五个 Coder checkpoint 跑 τ-bench retail（documented 配置）——emitted 随规模上升而 parsed 保持 0 则是接口属性，否则是代码形状 payload 的属性；小规模需要 rescue 臂（像 Granite 那样不尝试调用的 checkpoint 只给无信息的 0）。</details>

**Q34. 解释"A serving interface is part of both the benchmark's measurement function and an agent's effective action space"。**
<details><summary>答案</summary>测量函数：基准记录的是 P(Y)/E(P(Y)) 的函数而不是 Y，所以 parser 是测量仪器的一部分（Qwen：同样字节，换接口分数 0→0.96）。动作空间：接口决定哪些动作能到达环境（Llama：strict 让错误动作无法到达；RL：parser 决定哪些动作出现在经验里）。所以工具使用结论必须针对"模型–接口配置"而不是"模型"。</details>

**Q35. "Mistral 和 DeepSeek 为什么在 taxonomy 里？证据这么弱。"**
<details><summary>答案</summary>论文明说只作为"还有其他失败层"的动机（template 层、token 层），taxonomy 是四个实例不是枚举、故意不对称；Mistral 四种配置全有请求错误、无可准入比较；DeepSeek 只有模板检查。都不支持可比的性能声明。它们的价值是对比：只有 Mistral 的失败会"自报家门"，其他都静默。</details>

**Q36. 手算：Llama terse→strict final 15/3 的精确 McNemar *p*。**
<details><summary>答案</summary>n=18，k=3：P(X≤3) = (C(18,0)+C(18,1)+C(18,2)+C(18,3))/2¹⁸ = (1+18+153+816)/262144 = 988/262144 ≈ 0.003769；双侧 ×2 ≈ 0.00754。</details>

**Q37. tight 和 accepted 的交叉 16,323 / 1,285 / 521 说明什么？**
<details><summary>答案</summary>tight 分类器不是 parser 的复制品：有 1,285 个形状像但 parser 不收的，有 521 个 parser 收了但不满足 tight 严格条件的（如键顺序、缺 Python 词法标记）。两者是独立的检验，所以 tight 才能作为独立的 C(Y)。</details>

**Q38. 论文的"measurement card"至少要记录什么？**
<details><summary>答案</summary>模型与 tokenizer revision、template 与 parser 哈希、解码参数、请求错误、emitted-intent 判据、acceptance、dispatch、execution、observation 计数、outcome delta。把报告单位从"模型调用率"变成有版本的"模型–接口测量卡"。</details>

---

<a id="ch9"></a>

## 9. 术语表

<sub>[↑ 回到目录](#toc) ｜ [← 上一章：8. 自测题（附答案）](#ch8) ｜ [下一章：10. 延伸阅读 →](#ch10)</sub>

| 英文 | 中文 | 一句话解释 |
|---|---|---|
| interface-induced trajectory censoring | 接口诱导的轨迹删失 | 本文核心概念 |
| emission | 发射 / 原始输出 | 模型生成的字节 Y |
| envelope | 包装层 | 如 `<tool_call>…</tool_call>`、`<tools>…</tools>` |
| payload | 载荷 | 包装层内的 JSON |
| serialization | 序列化 | template 规定的格式 |
| masking | 遮蔽 | 已吐出的动作被接口抹掉 |
| constructive suppression | 建设性抑制 | 生成时就阻止（如 strict） |
| contract mismatch | 契约不匹配 | 各组件按自己的契约工作但契约互相不一致 |
| strong complementarity | 强互补 | 单换都无效、联合替换有效 |
| simple effect / main effect / interaction | 简单效应 / 主效应 / 交互 | 见 2.10(g) |
| documented configuration | 文档配置 | 按 vLLM 家族级文档选的配置 |
| dedicated adapter | 专用适配器 | hanXen parser + 配套 template |
| tight / strong / weak | 紧 / 强 / 弱判据 | KodCode emission 分类器的三个级别 |
| silent count | 静默数 | tight 且未解析 |
| sensitivity estimate | 敏感性估计 | ≈72，依赖精度可迁移假设 |
| fixed-output replay | 固定输出重放 | 不重新生成，对原字节重解析 |
| failure layer / first-failure | 失败层 / 首失败 | 按第一个失败点归类 |
| parser loss | 解析器丢失 | 按规则应接受却没接受（0 例） |
| positive control | 阳性对照 | 如 `tool_choice: required` |
| admissibility gate | 准入门 | 100 题、rc 0、0 错误、provenance |
| clean-arm / end-to-end | 干净臂 / 端到端 | 前者排除有错误的臂，后者保留完整分母 |
| turn-1 / final pass | 首轮 / 最终通过 | — |
| rescue | 救回 | final 过但 turn-1 没过 |
| tested step | 测试步 | 抽出代码并测试的一次生成 |
| initiation | 发起 | 至少一次被接受并执行的调用 |
| role confusion | 角色混淆 | 把要实现的函数当工具调 |
| experience access | 经验接触 | RL 轨迹里出现工具交互 |
| programmatic-feedback evaluation | 程序化反馈评测 | 自动执行代码、user 消息给反馈 |
| native FC evaluation | 原生函数调用评测 | 自愿调用、tool-role 反馈 |
| lineage | 谱系 | 最终模型的训练步来源 |
| test–retest | 测试–复测 | 同一 checkpoint 重复评测的变异 |
| discordant pairs | 不一致对 | McNemar 只看它们 |
| post-treatment conditioning | 后处理条件化 | 按受处理影响的变量选子集 |
| retrospective reporting family | 回顾性报告族 | 27-test，非预注册 |
| preregistration / protocol deviation | 预注册 / 协议偏离 | — |
| union bound | 并集界 | 两个 97.5% 区间合成 ≥95% |
| measurement card / conformance report | 测量卡 / 合规报告 | 本文的实践建议 |
| decontamination / containment | 去污染 / 包含度 | n-gram 交集 / 较小集合 |

---

<a id="ch10"></a>

## 10. 延伸阅读

<sub>[↑ 回到目录](#toc) ｜ [← 上一章：9. 术语表](#ch9)</sub>

按"读懂这篇论文"的优先级：

1. **vLLM tool calling 文档**（v0.27.1，"Qwen Models"一节）+ **vLLM issue #32926**：问题的源头。
2. **hanXen/vllm-qwen2.5-coder-tool-parser**：社区修复，看 parser 和 template 怎么配合。
3. **BFCL**（Patil et al., 2025）：AST checker 和 multi-turn state-based 打分怎么做。
4. **τ-bench**（Yao et al., 2025）：用户模拟器和最终状态打分。
5. **DeepSeekMath**（Shao et al., 2024）：GRPO 的出处。
6. **LoRA**（Hu et al., 2022）。
7. **HybridFlow / verl**（Sheng et al., 2025）：AgentLoop 和 rollout 架构。
8. **Garware 2026（broken ruler）**：最接近的类比，"同样字节重解析 0%→76%"。
9. **Sclar et al. 2024（format spread）**、**Dehghani et al. 2021（benchmark lottery）**：评测对"讨厌因素"敏感的经典工作。
10. **Tam et al. 2024（Speak Freely）**：约束生成伤害推理——和 Llama 结果对照着读。
11. **Harness-Bench（Yao 2026）**、**Agent-Reactive Bugs（Chen 2026）**：最近的 model–harness 边界工作，搞清楚本文和它们的区别。
12. 统计：McNemar 检验、Clopper–Pearson 区间、Bonferroni、Cohen's κ、析因设计里主效应 vs 简单效应——任何一本实验设计教材（如 Montgomery, *Design and Analysis of Experiments*）的相关章节。

---

*本文档依据 arXiv v2 源码（`paper/iclr2027/arxiv_v2/source/`）编写。标【我的解释】的部分是辅助理解的背景，不是论文原文；如与论文冲突，以论文为准。*

---

<sub>[↑ 回到目录](#toc)</sub>
