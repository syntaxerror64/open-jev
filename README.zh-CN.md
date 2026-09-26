# Open Jev

[![许可证](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)
[![发布版本](https://img.shields.io/github/v/release/syntaxerror64/open-jev?sort=semver)](https://github.com/syntaxerror64/open-jev/releases)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2%2B-ee4c2c)](https://pytorch.org/)
[![欢迎 PR](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/syntaxerror64/open-jev/pulls)

[English](README.md) | [Русский](README.ru.md) | **简体中文**

一个从零开始、基于第一性原理的 PyTorch 开源复现，复现
[TypeSafe AI 的 Jev / System One
模型](https://typesafe.ai/blog/introducing-system-one-models-and-jev)的思想：
输入是非结构化的程序状态，输出是带类型的概率决策。

> [!IMPORTANT]
> 这是一个非官方的研究性实现。发布的检查点通过蒸馏训练——属于研究级别，
> 没有外部基准测试。它不是生产环境的 Jev 模型，不复现 TypeSafe AI 的训练数据
> 或权重，也不声称与其公开结果一致。

## 目录

- [核心思想](#核心思想)
- [主要特性](#主要特性)
- [安装](#安装)
- [快速开始](#快速开始)
- [训练](#训练)
- [已发布的检查点](#已发布的检查点)
- [评估](#评估)
- [开发](#开发)
- [文档](#文档)
- [局限性](#局限性)
- [参与贡献](#参与贡献)
- [许可证](#许可证)
- [致谢](#致谢)

## 核心思想

Jev 被定位为一个 *system one* 模型：输入非结构化的程序状态，输出带类型的概率
决策。

本实现不逐个 token 地生成文本，而是对状态只编码一次，再通过小型的类型化
readout 头回答每一个问题。

```text
类 JSON 状态 ──> 双向状态编码器 ──> 共享状态缓存
                                          │
                    问题槽位 ── 交叉注意力 ┘
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
           noul          choice         score
          p(true)      p(options)     p(levels)
```

问题之间彼此隔离，并被折叠进批处理维度。它们可以关注共享状态，但不能关注
其他问题。结果是一次并行的 forward 计算，输出被各自声明的类型所约束。

## 主要特性

- **类型化输出。** `Noul` 返回 p(true)，`Choice` 只在运行时声明的选项上做
  softmax，`Score` 返回评分等级分布及其期望——`Choice` 的答案绝不会是你没有
  提供过的选项。
- **一次编码，多个问题。** 双向变换器将嵌套状态一次性编码为可缓存的共享
  表示；每个问题通过可学习的 query 槽位交叉注意力访问该缓存。缓存复用与
  多问题扩展性已在 [benchmarks/RESULTS.md](benchmarks/RESULTS.md) 中测量。
- **可以推理的置信度。** 证据（evidential）头将认知不确定性与类别概率分开
  建模；另一种基于离散度（spread）的方式可通过配置选择。
- **软标签训练。** `RLCDLoss` 在完整分布上组合 soft-target NLL、Brier 分数、
  一致性（consistency）、证据项与 ECE——绝不用 one-hot 标签——因此分歧与
  模糊性保持可见。
- **可复现的产物。** 每个发布版本都附带 `.pt` 检查点和带 `sha256` 的 JSON
  清单；模型配置随产物一同保存，`JevConfig(**manifest["config"])` 即可重建
  精确的模型结构——没有任何数字需要手工信任。
- **由测试保证的结构。** 150 个测试检查结构性保证（类型化输出、归一化分布、
  问题隔离、缓存复用），并拦截每一次 `git push`。

实现刻意保持小巧可读。默认分词器是确定性哈希分词器；训练得到的 BPE 分词器
（用 `python -m scripts.train_bpe` 仅从本仓库文本构建）实现相同接口，可按需
传给 `Jev`。

## 安装

```bash
git clone https://github.com/syntaxerror64/open-jev.git
cd open-jev
python -m pip install "torch>=2.0"
```

建议使用 Python 3.10 或更高版本。

## 快速开始

```python
import torch

from open_jev.main import Choice, Jev, JevConfig, Noul, Score

model = Jev(
    JevConfig(
        vocab_size=4096,
        d_model=128,
        n_heads=4,
        d_ff=512,
        n_state_layers=3,
        n_readout_layers=4,
    )
).eval()

state = {
    "customer": {"tier": "enterprise", "tenure_months": 34},
    "message": "This is the third duplicate charge. Please fix it.",
    "policy": "Duplicate charges are refundable within 60 days.",
}

questions = [
    Noul("The customer is requesting a refund.", key="wants_refund"),
    Choice(
        "Which team should handle this?",
        options=["billing", "technical", "account"],
        key="route",
    ),
    Score(
        "How frustrated is the customer?",
        labels=["calm", "annoyed", "frustrated", "very frustrated"],
        key="frustration",
    ),
]

with torch.no_grad():
    answers = model([state], questions)[0]

for answer in answers:
    print(answer)
```

这段代码构建的是全新随机初始化的模型，从不加载检查点，因此上面的数值没有
意义；发布的检查点（见[已发布的检查点](#已发布的检查点)）是通过蒸馏训练的。
这里真正有用的保证是结构性的：`Choice` 的答案只能是你提供的选项之一。

运行完整演示（包含一次面向校准的训练步骤）：

```bash
python example.py
```

`forward.py` 是最小示例——只有推理，没有训练步骤。

## 训练

`RLCDLoss` 期望的是概率分布而不是 one-hot 标签。这样分歧与模糊性保持可见，
而不是把每个样本都推向确定。

```python
from open_jev.main import RLCDLoss

targets = [
    torch.tensor([[0.08, 0.92]]),
    torch.tensor([[0.80, 0.05, 0.15]]),
    torch.tensor([[0.02, 0.10, 0.48, 0.40]]),
]

loss_fn = RLCDLoss()
loss = loss_fn(model, [state], questions, targets)
loss.backward()
```

做一致性训练时，把语义等价的状态经 `augmented_states` 传入——例如改写或打乱
字典键。

训练管线把这个损失包装进可恢复的循环，由教师层提供软标签（绝不用数据集里的
标签）：

```bash
python -m pipeline.train --config pipeline/examples/tiny.json --steps 30
python -m pipeline.train --config pipeline/examples/tiny.json --steps 10 \
       --resume runs/tiny/checkpoint.pt
```

## 已发布的检查点

训练运行被导出为带版本的 v1 格式（`state_dict`、配置、步数、指标、`sha256`），
并作为 GitHub Release 资产发布：

| 版本 | 资产 | 状态 |
|---|---|---|
| [v0.2.0](https://github.com/syntaxerror64/open-jev/releases/tag/v0.2.0) | [`jev-real.pt`](https://github.com/syntaxerror64/open-jev/releases/download/v0.2.0/jev-real.pt) + [`jev-real.json`](https://github.com/syntaxerror64/open-jev/releases/download/v0.2.0/jev-real.json)（6.3 MB） | 蒸馏训练：教师 Qwen2.5-0.5B，500 条 Alpaca 数据，200 步 |
| [v0.1.0](https://github.com/syntaxerror64/open-jev/releases/tag/v0.1.0) | `jev-tiny.pt` + `jev-tiny.json`（2.3 MB） | 随机初始化——已被取代 |

按清单校验下载的文件，或直接从发布版本加载：

```bash
sha256sum jev-real.pt                       # 必须与 jev-real.json 中的 "sha256" 一致
JEV_CHECKPOINT_URL=<asset-url> pytest tests/test_checkpoint_release.py -v
``

自行导出并发布一次运行：

```bash
python -m scripts.export_checkpoint --run runs/tiny --out dist/jev-tiny.pt
python scripts/release.py --tag v0.1.0 --asset dist/jev-tiny.pt --dry-run
```

`--dry-run` 以 JSON 打印发布清单而不联网；真正的命令会在存在训练产物时执行
`gh release create`。[MODEL_CARD.md](MODEL_CARD.md) 描述了产物的内容及其当前
局限。

## 评估

在保留集上比较随机初始化与发布的 v0.2.0 检查点（100 条，seed 0，`n_bins`
10——细节见 [eval/COMPARISON.md](eval/COMPARISON.md)）：

| 指标 | 随机 | 检查点（第 200 步） |
|---|---|---|
| ECE | 0.3073 | 0.1532 |
| Brier | 0.3133 | 0.1509 |
| NLL | 0.8850 | 0.6695 |
| 一致性（对称 KL，越低越好） | 0.006497 | 0.0001031 |

**这些数字意味着什么。** 保留集没有 ground-truth 标签：ECE、Brier 与 NLL 是
针对教师蒸馏出的目标计算的，因此它们衡量的是*与教师在这份保留集上的一致
性*——不是正确性，也不是外部基准。确切措辞与来源见
[MODEL_CARD.md](MODEL_CARD.md)。

## 开发

```bash
python -m venv .venv
.venv/bin/pip install "torch>=2.0" --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

测试套件位于 `tests/`，只检查结构性保证（类型化输出、归一化分布、问题隔离、
缓存复用）——这些性质对任何权重都成立，无论训练与否。标记为 `slow` 的测试
（`example.py` 演示）会被下面的推送门禁跳过；用普通 `pytest` 运行它们。

推送受该套件门禁保护：每个克隆启用一次内置钩子，测试不过 `git push` 就不会
上传任何东西。

```bash
git config core.hooksPath .githooks
```

紧急情况可用 `git push --no-verify`。

## 文档

- [MODEL_CARD.md](MODEL_CARD.md) —— 发布产物包含什么、指标来源与局限
- [eval/COMPARISON.md](eval/COMPARISON.md) —— 保留集上随机与检查点的对比
- [benchmarks/RESULTS.md](benchmarks/RESULTS.md) —— 状态缓存复用与多问题
  扩展性的测量
- `docs/ARCHITECTURE.md` —— 复现方案的完整推理与权衡（本地开发文档）

## 局限性

- **类型化输出防止的是 schema 违规，而不是错误答案。** 在无人值守使用之前，
  必须在分布漂移下重新测量校准。
- **小型蒸馏模型。** 157 万参数、200 步、500 条训练指令；「已训练」≠「好」——
  上面的指标是与教师的一致性，不是正确性。没有外部基准。
- **研究范围。** 基于公开材料的独立复现——不复现 TypeSafe AI 的数据、权重或
  公布结果。
- **仅限 CPU、仅限英文。** 开发与测试基于 `torch` CPU 版本和英文数据；其他
  语言与加速器未经测试。

## 参与贡献

欢迎 issue 与 pull request。推送前请运行 `.venv/bin/python -m pytest`——内置的
pre-push 钩子（`git config core.hooksPath .githooks`，每个克隆一次）在每次
推送时执行同一套测试。

## 许可证

采用 [Apache License 2.0](LICENSE)。

简言之：这个代码你几乎可以做任何想做的事——使用、修改、再发布、销售、私用
——商业或个人目的皆可。唯一的条件是保留许可证与其声明，并标注你修改过的
文件。不提供任何担保。

## 致谢

- 感谢 [TypeSafe AI](https://typesafe.ai) 发布 [System One Models and
  Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)，以及
  足够多的公开细节启发独立实验。
- 感谢 [Kye Gomez](https://github.com/kyegomez/open-jev) 的 open-jev 初始
  脚手架，本仓库在其之上构建。
