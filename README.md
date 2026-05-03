# Football Predictor v4.0
A Python Program which use Monte Carlo and Poisson to predict the result of football match

# ⚽ Football Predictor v4.0

**蒙特卡洛 × 泊松分布 × 自适应多因子足球比赛预测系统**

一个纯 Python 命令行工具，无需训练即可预测足球比赛结果。支持联赛单场、欧冠两回合淘汰赛、多队锦标赛三种模式，自动适配有无高级数据（xG、射正率等）。

---

## 目录

- [功能概览](#功能概览)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [数据文件格式](#数据文件格式)
- [使用详解](#使用详解)
- [模型原理](#模型原理)
- [注意事项与局限性](#注意事项与局限性)
- [开源协议](#开源协议)

---

## 功能概览

| 功能 | 说明 |
|------|------|
| 单场预测 | 输入主客队，输出胜/平/负概率及预期比分 |
| 两回合预测 | 欧冠淘汰赛模式，支持输入已知首回合比分 |
| 锦标赛模拟 | 输入多组对阵，模拟完整淘汰赛路径至决赛 |
| 自动数据适配 | 检测到 xG/射正/扑救等列时使用13因子模型，否则自动降级为7因子基础模型 |
| 文件导入 | 支持 CSV 和 JSON 两种格式导入球队数据 |
| 手动输入 | 无文件时可逐项手动输入球队信息 |
| 模板生成 | 一键生成带示例数据的 CSV 模板 |

---

## 环境要求

- **Python 3.7+**（仅使用标准库，无需安装任何第三方依赖）

验证版本：

```bash
python3 --version
```

---

## 快速开始

### 1. 启动程序

```bash
python3 football_predictor.py
```

程序启动后进入交互菜单，不会自动执行任何计算。

### 2. 生成模板

在菜单中选择 `[8]`，会在当前目录生成两个模板文件：

- `teams_template.csv` — 含高级数据（xG、射正率等）的完整模板
- `teams_template_basic.csv` — 仅基础数据的模板

### 3. 填写数据 → 加载 → 预测

将你的球队数据填入 CSV，然后在菜单中选择 `[1]` 加载文件，再选择 `[4]`/`[5]`/`[6]` 进行预测。

### 完整流程示例

```
$ python3 football_predictor.py

  主菜单:
    [1] 加载球队数据 (从文件)
    ...

  请选择 > 1
  输入文件路径: ./my_teams.csv
  [OK] 已加载 4 支球队

  请选择 > 4            ← 单场预测
  主队缩写: PSG
  客队缩写: BMU
  中立场地? [y/N]: n

  ── 单场预测 (巴黎圣日耳曼主场) ──
  │ 巴黎圣日耳曼胜  │  52.1%  │
  │ 平局           │  21.9%  │
  │ 拜仁慕尼黑胜    │  25.9%  │
  │ 预期比分        │ 2.0-1.3 │
```

---

## 数据文件格式

### CSV 格式

第一行为表头，每行一支球队。编码建议使用 UTF-8。

#### 必填列

| 列名 | 类型 | 说明 |
|------|------|------|
| `name` | 字符串 | 球队全名（如 `巴黎圣日耳曼`） |
| `abbr` | 字符串 | 球队缩写，3字母（如 `PSG`），用于输入和匹配 |

#### 基础数据列（建议填写）

| 列名 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `league_pts` | 整数 | 联赛/小组赛积分 | 0 |
| `wins` | 整数 | 胜场数 | 0 |
| `draws` | 整数 | 平场数 | 0 |
| `losses` | 整数 | 负场数 | 0 |
| `goals_for` | 整数 | 总进球数 | 0 |
| `goals_against` | 整数 | 总失球数 | 0 |
| `uefa_coeff` | 浮点数 | UEFA 俱乐部系数 | 50 |
| `ucl_titles` | 整数 | 历史欧冠冠军次数 | 0 |
| `form_1` ~ `form_5` | 整数 | 最近5场结果（`3`=胜, `1`=平, `0`=负） | 全部为1 |

#### 高级数据列（可选，全部留空则自动使用基础模型）

| 列名 | 类型 | 范围 | 说明 |
|------|------|------|------|
| `xg` | 浮点数 | 通常 0.5~3.5 | 场均期望进球 |
| `xga` | 浮点数 | 通常 0.3~2.5 | 场均期望失球 |
| `shots_on_pct` | 浮点数 | 0~1 | 射正率（射正次数/总射门次数） |
| `possession` | 浮点数 | 0~1 | 场均控球率 |
| `pass_acc` | 浮点数 | 0~1 | 传球准确率 |
| `tackle_int` | 浮点数 | 通常 10~25 | 场均抢断+拦截次数 |
| `save_pct` | 浮点数 | 0~1 | 门将扑救率 |

#### CSV 示例

完整模式：

```csv
name,abbr,league_pts,wins,draws,losses,goals_for,goals_against,uefa_coeff,ucl_titles,form_1,form_2,form_3,form_4,form_5,xg,xga,shots_on_pct,possession,pass_acc,tackle_int,save_pct
巴黎圣日耳曼,PSG,14,4,2,2,20,8,88,0,3,3,3,3,1,2.10,0.72,0.37,0.59,0.89,16.8,0.80
拜仁慕尼黑,BMU,21,7,0,1,31,12,108,6,3,3,3,3,0,2.85,1.10,0.40,0.61,0.90,16.5,0.70
```

基础模式（无高级列）：

```csv
name,abbr,league_pts,wins,draws,losses,goals_for,goals_against,uefa_coeff,ucl_titles,form_1,form_2,form_3,form_4,form_5
巴黎圣日耳曼,PSG,14,4,2,2,20,8,88,0,3,3,3,3,1
拜仁慕尼黑,BMU,21,7,0,1,31,12,108,6,3,3,3,3,0
```

### JSON 格式

```json
{
  "teams": [
    {
      "name": "巴黎圣日耳曼",
      "abbr": "PSG",
      "league_pts": 14,
      "wins": 4,
      "draws": 2,
      "losses": 2,
      "goals_for": 20,
      "goals_against": 8,
      "uefa_coeff": 88,
      "ucl_titles": 0,
      "recent_form": [3, 3, 3, 3, 1],
      "xg": 2.10,
      "xga": 0.72,
      "shots_on_pct": 0.37,
      "possession": 0.59,
      "pass_acc": 0.89,
      "tackle_int": 16.8,
      "save_pct": 0.80
    }
  ]
}
```

JSON 中高级字段同样全部可选。

---

## 使用详解

### 菜单选项说明

#### [1] 加载球队数据

输入 CSV 或 JSON 文件路径。可以多次加载不同文件，球队数据会合并（相同缩写的球队会被覆盖）。

```
  请选择 > 1
  输入文件路径: /path/to/teams.csv
```

#### [2] 手动输入球队

通过命令行逐项输入一支球队的数据，适合临时添加或没有文件时使用。会询问是否输入高级数据。

#### [3] 查看已加载球队

显示当前所有球队及其实力评分，按实力从高到低排列。同时显示当前是「基础模式」还是「完整模式」。

#### [4] 单场预测（联赛模式）

输入主队和客队的缩写，模拟 N 次比赛（默认50,000次），输出：

- 主胜 / 平局 / 客胜 概率
- 预期比分（平均进球数）
- 可选中立场地（取消主场优势）

#### [5] 两回合预测（欧冠淘汰赛）

输入球队A（首回合主场）和球队B（首回合客场）。

- 如果首回合已踢，输入比分（如 `2-1`），程序只模拟次回合
- 如果首回合未踢，直接回车，程序模拟两回合
- 总比分平局时模拟加时/点球（基于实力概率）

输出两队晋级概率和预期总比分。

#### [6] 锦标赛模拟

输入多组对阵（逗号分隔，如 `PSG,BMU`），空行结束。程序自动模拟多轮淘汰直到决赛（决赛为中立场地单场决胜）。

对阵数量建议为2的幂（2、4、8），否则只预测第一轮。

#### [7] 设置模拟次数

默认 50,000 次。增加次数可以提高结果稳定性但会变慢：

| 次数 | 耗时约 | 适用场景 |
|------|--------|----------|
| 10,000 | <1s | 快速测试 |
| 50,000 | 1~3s | 日常使用（推荐） |
| 200,000 | 5~15s | 需要精确概率 |
| 500,000 | 15~40s | 最高精度 |

#### [8] 生成模板文件

在当前目录生成 `teams_template.csv` 和 `teams_template_basic.csv`，填入你的数据即可。

---

## 模型原理

### 预测方法

采用**蒙特卡洛模拟 + 泊松分布**。每次模拟中：

1. 根据两队的多维实力指标计算各自的期望进球数（λ参数）
2. 从泊松分布中随机采样得出比分
3. 重复数万次，统计各结果出现的频率作为概率

### 实力评分

根据数据完整度自动选择模型：

**基础模式（7因子）** — 当高级数据缺失时使用

| 因子 | 权重 | 说明 |
|------|------|------|
| UEFA系数 | 20% | 反映球队长期竞争力 |
| 联赛积分 | 20% | 反映当前赛季表现 |
| 近期状态 | 18% | 最近5场胜/平/负 |
| 攻击力 | 15% | 场均进球率 |
| 防守力 | 12% | 场均失球率（越低越好） |
| 历史底蕴 | 10% | 欧冠冠军次数 |
| 基线 | 5% | 防止极端值 |

**完整模式（13因子）** — 当超过半数球队有高级数据时使用

在基础因子之上增加 xG、xGA、射正率、控球率、传球准确率、抢断拦截、扑救率共7个高级因子，基础因子权重相应降低以保持总和为1。

### 进球模型修正

在完整模式下，期望进球数还会受到两个修正因子影响：

- **xG 修正**：基于进攻方 xG 与防守方 xGA 的比值调整
- **射正-扑救修正**：基于进攻方射正率与防守方扑救率的对抗关系调整

### 主场优势

默认主场优势系数为 0.30，体现在主队的期望进球数上。中立场地时取消此加成。联赛平均进球参数取自历史数据（主场 1.55，客场 1.15）。

---

## 注意事项与局限性

### 数据质量

- **「垃圾进，垃圾出」**：预测准确度完全取决于输入数据的质量。请确保数据来源可靠且及时更新。
- 建议使用最近一个赛季或最近 N 场（而非历史全部）的统计数据，以反映球队当前实力。
- 不同数据源的统计口径可能不同（比如 xG 模型各家计算方式有差异），尽量使用同一来源的数据。

### 模型局限

- **不考虑球员级别信息**：伤病、停赛、转会、轮换等球员层面的变化无法体现。如果核心球员缺阵，预测结果可能偏差较大。
- **不考虑战术和心理因素**：如赛程疲劳、淘汰赛经验、更衣室氛围等。
- **不考虑赛季阶段**：赛季初和赛季末球队状态不同，模型不区分。
- **泊松分布的局限**：泊松模型假设进球事件独立，实际比赛中一个进球可能改变双方策略（领先方收缩、落后方压上）。
- **平局后的加时/点球**：两回合总比分平局时，晋级方由实力概率随机决定，并非真实模拟加时赛和点球大战。

### 使用建议

- 预测结果是概率而非确定结论，"60%胜率"意味着仍有40%的可能性不胜。
- 适合用于趋势分析和参考，不建议作为任何决策的唯一依据。
- 模拟次数建议 ≥ 50,000 次以获得稳定结果。低于 10,000 次时概率波动明显。
- 多次运行同样参数会因随机种子不同产生略微不同的结果，这是正常现象。

### 数据获取参考

以下免费数据源可用于获取本程序所需的各项指标：

| 数据源 | 网址 | 提供内容 |
|--------|------|----------|
| Football-Data.co.uk | football-data.co.uk | 比赛统计、赔率，CSV直接下载 |
| Understat | understat.com | xG、xGA、npxG |
| FBref | fbref.com | 射门、传球、防守等详细统计 |
| football-data.org | football-data.org | 赛程、积分榜（免费API） |
| StatsBomb Open Data | github.com/statsbomb/open-data | 事件级数据（部分比赛） |

---

## 开源协议

```
MIT License

Copyright (c) 2026 james2077

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 免责声明

本项目仅供学习和研究用途。预测结果基于统计模型，不构成任何投注、赌博或其他决策的建议。足球比赛存在大量不可预测的因素，任何模型都无法保证准确性。使用者应自行承担基于本工具输出所做决策的全部风险。
