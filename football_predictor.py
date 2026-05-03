#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
足球比赛预测系统 v4.0
======================
Monte Carlo · Poisson · 自适应多因子模型

功能：
  1. 单场预测（联赛单循环）：输入两队，预测胜/平/负概率
  2. 两回合预测（欧冠淘汰赛双循环）：模拟主客两回合，预测晋级概率
  3. 锦标赛预测：多组对阵，模拟完整淘汰赛路径
  4. 支持从 CSV/JSON 文件导入球队数据
  5. 自动检测高级数据(xG/射正/扑救等)，无则用基础模式

用法：
  python football_predictor.py
  启动后进入交互菜单，按提示操作。

数据文件格式：
  CSV 必须包含列: name, abbr
  可选列: league_pts, wins, draws, losses, goals_for, goals_against,
          uefa_coeff, ucl_titles, form_1..form_5,
          xg, xga, shots_on_pct, possession, pass_acc, tackle_int, save_pct
  JSON 格式见 --generate-template
"""

import random
import math
import json
import csv
import sys
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  数据结构
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class Team:
    name: str
    abbr: str
    league_pts: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0
    uefa_coeff: float = 50.0
    ucl_titles: int = 0
    recent_form: list = field(default_factory=lambda: [1, 1, 1, 1, 1])
    # 高级数据（全部可选，为0表示无数据）
    xg: float = 0.0
    xga: float = 0.0
    shots_on_pct: float = 0.0
    possession: float = 0.0
    pass_acc: float = 0.0
    tackle_int: float = 0.0
    save_pct: float = 0.0

    @property
    def total_games(self):
        return self.wins + self.draws + self.losses

    @property
    def gpg(self):
        return self.goals_for / max(self.total_games, 1)

    @property
    def cpg(self):
        return self.goals_against / max(self.total_games, 1)

    @property
    def has_advanced(self):
        return self.xg > 0 or self.shots_on_pct > 0 or self.save_pct > 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  预测引擎
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 权重配置
W_BASIC = {
    "uefa_coeff": .20, "league_phase": .20, "recent_form": .18,
    "attack": .15, "defense": .12, "history": .10, "baseline": .05,
}

W_FULL = {
    "uefa_coeff": .15, "league_phase": .15, "recent_form": .12,
    "attack": .08, "defense": .08, "history": .05,
    "xg": .10, "xga": .06, "shots_on_pct": .05, "possession": .04,
    "pass_acc": .03, "tackle_int": .02, "save_pct": .02, "baseline": .05,
}

# 联赛平均进球参数
AVG_HOME_GOALS = 1.55
AVG_AWAY_GOALS = 1.15
HOME_ADVANTAGE = 0.30


class PredictionEngine:
    """蒙特卡洛 + 泊松分布预测引擎"""

    def __init__(self, teams: Dict[str, Team], seed=None):
        """
        teams: {"缩写": Team对象} 的字典
        """
        self.teams = teams
        if seed is not None:
            random.seed(seed)

        # 自动检测是否有高级数据
        adv_count = sum(1 for t in teams.values() if t.has_advanced)
        self.advanced_mode = adv_count > len(teams) / 2
        self.W = dict(W_FULL) if self.advanced_mode else dict(W_BASIC)

        # 归一化参数
        vals = list(teams.values())
        self._max_coeff = max((t.uefa_coeff for t in vals), default=1) or 1
        self._max_pts = max((t.league_pts for t in vals), default=1) or 1
        self._max_xg = max((t.xg for t in vals), default=1) or 1
        self._max_ti = max((t.tackle_int for t in vals), default=1) or 1

    def strength(self, t: Team) -> float:
        """计算球队综合实力分"""
        W = self.W
        r = 0.0

        # 基础因子
        r += W.get("uefa_coeff", 0) * (t.uefa_coeff / self._max_coeff)
        r += W.get("league_phase", 0) * (t.league_pts / self._max_pts)
        form_max = max(len(t.recent_form) * 3, 1)
        r += W.get("recent_form", 0) * (sum(t.recent_form) / form_max)
        r += W.get("attack", 0) * min(t.gpg / 3.5, 1.0)
        r += W.get("defense", 0) * max(0, 1 - t.cpg / 2.5)
        r += W.get("history", 0) * min(t.ucl_titles * 0.08, 1.0)

        # 高级因子
        if self.advanced_mode:
            if t.xg > 0:
                r += W.get("xg", 0) * min(t.xg / self._max_xg, 1)
            if t.xga > 0:
                r += W.get("xga", 0) * max(0, 1 - t.xga / 2)
            if t.shots_on_pct > 0:
                r += W.get("shots_on_pct", 0) * min(t.shots_on_pct / 0.5, 1)
            if t.possession > 0:
                r += W.get("possession", 0) * min(t.possession / 0.7, 1)
            if t.pass_acc > 0:
                r += W.get("pass_acc", 0) * min(t.pass_acc / 0.95, 1)
            if t.tackle_int > 0:
                r += W.get("tackle_int", 0) * min(t.tackle_int / self._max_ti, 1)
            if t.save_pct > 0:
                r += W.get("save_pct", 0) * min(t.save_pct / 0.85, 1)

        return r + W.get("baseline", 0.05)

    def _expected_goals(self, team: Team, opp: Team, home: bool, neutral: bool) -> float:
        """计算一场比赛中 team 的期望进球数（泊松λ参数）"""
        base = self.strength(team)

        # xG 修正因子
        xf = 1.0
        if self.advanced_mode and team.xg > 0 and opp.xga > 0:
            xf = 0.7 + 0.3 * min(team.xg / max(opp.xga, 0.3) / 3, 1)

        # 射正 vs 扑救修正因子
        fm = 1.0
        if self.advanced_mode and team.shots_on_pct > 0 and opp.save_pct > 0:
            fm = 0.85 + 0.15 * (team.shots_on_pct / 0.45) * (1 - opp.save_pct * 0.5)

        if neutral:
            lam = base * (AVG_HOME_GOALS + AVG_AWAY_GOALS) / 2 * 1.8 * xf * fm
        elif home:
            lam = (base + HOME_ADVANTAGE) * AVG_HOME_GOALS * 1.5 * xf * fm
        else:
            lam = base * AVG_AWAY_GOALS * 1.5 * xf * fm

        return max(0.25, lam)

    @staticmethod
    def _poisson_sample(lam: float) -> int:
        """从泊松分布中采样一个进球数"""
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while True:
            k += 1
            p *= random.random()
            if p < L:
                break
        return k - 1

    def simulate_match(self, home_abbr: str, away_abbr: str, neutral=False):
        """模拟一场比赛，返回 (主队进球, 客队进球)"""
        h = self.teams[home_abbr]
        a = self.teams[away_abbr]
        hg = self._poisson_sample(self._expected_goals(h, a, True, neutral))
        ag = self._poisson_sample(self._expected_goals(a, h, False, neutral))
        return hg, ag

    # ── 单场预测 ──

    def predict_single(self, home_abbr: str, away_abbr: str, n_sims=50000, neutral=False):
        """
        单场比赛预测（联赛模式）
        返回: {"home_win": %, "draw": %, "away_win": %, "avg_home_goals": x, "avg_away_goals": x}
        """
        hw = dw = aw = 0
        total_hg = total_ag = 0
        for _ in range(n_sims):
            hg, ag = self.simulate_match(home_abbr, away_abbr, neutral)
            total_hg += hg
            total_ag += ag
            if hg > ag:
                hw += 1
            elif hg == ag:
                dw += 1
            else:
                aw += 1
        return {
            "home_win": hw / n_sims * 100,
            "draw": dw / n_sims * 100,
            "away_win": aw / n_sims * 100,
            "avg_home_goals": total_hg / n_sims,
            "avg_away_goals": total_ag / n_sims,
        }

    # ── 两回合预测 ──

    def predict_two_legs(self, team_a: str, team_b: str, n_sims=50000,
                         leg1_score=None):
        """
        两回合淘汰赛预测（欧冠模式）
        team_a 先主后客，team_b 先客后主
        leg1_score: 已知首回合比分 (a_goals, b_goals)，None表示未踢
        返回: {"a_advance": %, "b_advance": %, "avg_agg_a": x, "avg_agg_b": x}
        """
        a_adv = b_adv = 0
        total_agg_a = total_agg_b = 0

        for _ in range(n_sims):
            if leg1_score:
                l1_ha, l1_ab = leg1_score
            else:
                l1_ha, l1_ab = self.simulate_match(team_a, team_b)

            l2_hb, l2_aa = self.simulate_match(team_b, team_a)

            agg_a = l1_ha + l2_aa
            agg_b = l1_ab + l2_hb
            total_agg_a += agg_a
            total_agg_b += agg_b

            if agg_a > agg_b:
                a_adv += 1
            elif agg_b > agg_a:
                b_adv += 1
            else:
                # 平局 → 用实力概率决定（模拟加时/点球）
                sa = self.strength(self.teams[team_a])
                sb = self.strength(self.teams[team_b])
                if random.random() < sa / (sa + sb + 0.03):
                    a_adv += 1
                else:
                    b_adv += 1

        return {
            "a_advance": a_adv / n_sims * 100,
            "b_advance": b_adv / n_sims * 100,
            "avg_agg_a": total_agg_a / n_sims,
            "avg_agg_b": total_agg_b / n_sims,
        }

    # ── 锦标赛模拟 ──

    def predict_tournament(self, matchups: list, n_sims=50000, two_legs=True):
        """
        淘汰赛锦标赛预测
        matchups: [(team_a, team_b), ...] 必须是2的幂数量
        two_legs: True=两回合, False=单场决胜
        最后一场始终为单场决胜（中立场地决赛）
        返回: {abbr: {"champion": %, "final": %, "semi": %}}
        """
        champ_cnt = defaultdict(int)
        final_cnt = defaultdict(int)

        for _ in range(n_sims):
            alive = []
            # 第一轮
            for a, b in matchups:
                if two_legs:
                    hg1, ag1 = self.simulate_match(a, b)
                    hg2, ag2 = self.simulate_match(b, a)
                    agg_a = hg1 + ag2
                    agg_b = ag1 + hg2
                    if agg_a > agg_b:
                        alive.append(a)
                    elif agg_b > agg_a:
                        alive.append(b)
                    else:
                        sa = self.strength(self.teams[a])
                        sb = self.strength(self.teams[b])
                        alive.append(a if random.random() < sa / (sa + sb + .03) else b)
                else:
                    hg, ag = self.simulate_match(a, b)
                    if hg > ag:
                        alive.append(a)
                    elif ag > hg:
                        alive.append(b)
                    else:
                        sa = self.strength(self.teams[a])
                        sb = self.strength(self.teams[b])
                        alive.append(a if random.random() < sa / (sa + sb) else b)

            # 后续轮次（两回合），直到决赛
            while len(alive) > 2:
                next_round = []
                for i in range(0, len(alive), 2):
                    a, b = alive[i], alive[i + 1]
                    hg1, ag1 = self.simulate_match(a, b)
                    hg2, ag2 = self.simulate_match(b, a)
                    agg_a = hg1 + ag2
                    agg_b = ag1 + hg2
                    if agg_a > agg_b:
                        next_round.append(a)
                    elif agg_b > agg_a:
                        next_round.append(b)
                    else:
                        sa = self.strength(self.teams[a])
                        sb = self.strength(self.teams[b])
                        next_round.append(a if random.random() < sa / (sa + sb + .03) else b)
                alive = next_round

            # 决赛（中立场地单场决胜）
            fa, fb = alive[0], alive[1]
            final_cnt[fa] += 1
            final_cnt[fb] += 1
            hg, ag = self.simulate_match(fa, fb, neutral=True)
            if hg > ag:
                champ_cnt[fa] += 1
            elif ag > hg:
                champ_cnt[fb] += 1
            else:
                sa = self.strength(self.teams[fa])
                sb = self.strength(self.teams[fb])
                if random.random() < sa / (sa + sb):
                    champ_cnt[fa] += 1
                else:
                    champ_cnt[fb] += 1

        results = {}
        all_teams = set()
        for a, b in matchups:
            all_teams.add(a)
            all_teams.add(b)

        for t in all_teams:
            results[t] = {
                "champion": champ_cnt.get(t, 0) / n_sims * 100,
                "final": final_cnt.get(t, 0) / n_sims * 100,
            }

        return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  数据加载
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _safe_float(val, default=0.0):
    try:
        v = float(val)
        return v if v == v else default  # NaN check
    except (ValueError, TypeError):
        return default

def _safe_int(val, default=0):
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


def load_teams_csv(filepath: str) -> Dict[str, Team]:
    """从 CSV 文件加载球队数据"""
    teams = {}
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        print(f"  [信息] CSV 列: {', '.join(sorted(cols))}")

        for row in reader:
            abbr = row.get("abbr", "").strip()
            name = row.get("name", "").strip()
            if not abbr or not name:
                continue

            # 近5场状态
            form = []
            for i in range(1, 6):
                v = row.get(f"form_{i}", "")
                form.append(_safe_int(v, 1))
            if all(f == 1 for f in form) and "recent_form" in row:
                # 尝试解析 JSON 格式的 recent_form
                try:
                    form = json.loads(row["recent_form"])
                except:
                    pass

            t = Team(
                name=name, abbr=abbr,
                league_pts=_safe_int(row.get("league_pts")),
                wins=_safe_int(row.get("wins")),
                draws=_safe_int(row.get("draws")),
                losses=_safe_int(row.get("losses")),
                goals_for=_safe_int(row.get("goals_for")),
                goals_against=_safe_int(row.get("goals_against")),
                uefa_coeff=_safe_float(row.get("uefa_coeff"), 50),
                ucl_titles=_safe_int(row.get("ucl_titles")),
                recent_form=form,
                xg=_safe_float(row.get("xg")),
                xga=_safe_float(row.get("xga")),
                shots_on_pct=_safe_float(row.get("shots_on_pct")),
                possession=_safe_float(row.get("possession")),
                pass_acc=_safe_float(row.get("pass_acc")),
                tackle_int=_safe_float(row.get("tackle_int")),
                save_pct=_safe_float(row.get("save_pct")),
            )
            teams[abbr] = t
    return teams


def load_teams_json(filepath: str) -> Dict[str, Team]:
    """从 JSON 文件加载球队数据"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    team_list = data if isinstance(data, list) else data.get("teams", [])
    teams = {}
    for t in team_list:
        abbr = t.get("abbr", "")
        if not abbr:
            continue
        teams[abbr] = Team(
            name=t.get("name", abbr), abbr=abbr,
            league_pts=_safe_int(t.get("league_pts")),
            wins=_safe_int(t.get("wins")),
            draws=_safe_int(t.get("draws")),
            losses=_safe_int(t.get("losses")),
            goals_for=_safe_int(t.get("goals_for")),
            goals_against=_safe_int(t.get("goals_against")),
            uefa_coeff=_safe_float(t.get("uefa_coeff"), 50),
            ucl_titles=_safe_int(t.get("ucl_titles")),
            recent_form=t.get("recent_form", [1, 1, 1, 1, 1]),
            xg=_safe_float(t.get("xg")),
            xga=_safe_float(t.get("xga")),
            shots_on_pct=_safe_float(t.get("shots_on_pct")),
            possession=_safe_float(t.get("possession")),
            pass_acc=_safe_float(t.get("pass_acc")),
            tackle_int=_safe_float(t.get("tackle_int")),
            save_pct=_safe_float(t.get("save_pct")),
        )
    return teams


def load_teams_file(filepath: str) -> Dict[str, Team]:
    """根据扩展名自动选择加载方式"""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".json":
        return load_teams_json(filepath)
    elif ext in (".csv", ".tsv"):
        return load_teams_csv(filepath)
    else:
        print(f"  [错误] 不支持的文件格式: {ext}，请使用 .csv 或 .json")
        return {}


def generate_template():
    """生成示例模板文件"""
    # CSV
    csv_path = "teams_template.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "name", "abbr", "league_pts", "wins", "draws", "losses",
            "goals_for", "goals_against", "uefa_coeff", "ucl_titles",
            "form_1", "form_2", "form_3", "form_4", "form_5",
            "xg", "xga", "shots_on_pct", "possession",
            "pass_acc", "tackle_int", "save_pct"
        ])
        writer.writerow([
            "巴黎圣日耳曼", "PSG", 14, 4, 2, 2, 20, 8, 88, 0,
            3, 3, 3, 3, 1,
            2.10, 0.72, 0.37, 0.59, 0.89, 16.8, 0.80
        ])
        writer.writerow([
            "拜仁慕尼黑", "BMU", 21, 7, 0, 1, 31, 12, 108, 6,
            3, 3, 3, 3, 0,
            2.85, 1.10, 0.40, 0.61, 0.90, 16.5, 0.70
        ])
        writer.writerow([
            "阿森纳", "ARS", 24, 8, 0, 0, 23, 4, 95, 0,
            1, 3, 3, 3, 3,
            2.40, 0.48, 0.41, 0.57, 0.89, 18.5, 0.82
        ])
        writer.writerow([
            "马德里竞技", "ATM", 13, 4, 1, 3, 17, 14, 86, 0,
            0, 3, 3, 3, 0,
            1.50, 1.20, 0.32, 0.46, 0.82, 22.0, 0.72
        ])
    print(f"  [OK] 已生成 CSV 模板: {csv_path}")

    # CSV without advanced data
    csv_basic = "teams_template_basic.csv"
    with open(csv_basic, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "name", "abbr", "league_pts", "wins", "draws", "losses",
            "goals_for", "goals_against", "uefa_coeff", "ucl_titles",
            "form_1", "form_2", "form_3", "form_4", "form_5",
        ])
        writer.writerow(["巴黎圣日耳曼", "PSG", 14, 4, 2, 2, 20, 8, 88, 0, 3, 3, 3, 3, 1])
        writer.writerow(["拜仁慕尼黑", "BMU", 21, 7, 0, 1, 31, 12, 108, 6, 3, 3, 3, 3, 0])
    print(f"  [OK] 已生成基础模板（无高级数据）: {csv_basic}")
    print()
    print("  说明:")
    print("    - form_1~form_5: 最近5场结果, 3=胜 1=平 0=负")
    print("    - xg/xga: 场均期望进球/期望失球")
    print("    - shots_on_pct: 射正率(0~1), possession: 控球率(0~1)")
    print("    - pass_acc: 传球准确率(0~1), tackle_int: 场均抢断+拦截次数")
    print("    - save_pct: 扑救率(0~1)")
    print("    - 高级数据列全部可选，留空或不填则自动切换为基础模式")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  命令行手动输入球队
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def input_team_manual(prompt="") -> Team:
    """通过命令行交互输入一支球队的数据"""
    print(f"\n  === 输入球队数据{' (' + prompt + ')' if prompt else ''} ===")
    name = input("  球队名称: ").strip()
    abbr = input("  缩写(3字母): ").strip().upper()
    if not abbr:
        abbr = name[:3].upper()

    print("  -- 基础数据 (直接回车跳过使用默认值) --")
    league_pts = _safe_int(input("  联赛/小组赛积分 [0]: ") or 0)
    wins = _safe_int(input("  胜场 [0]: ") or 0)
    draws = _safe_int(input("  平场 [0]: ") or 0)
    losses = _safe_int(input("  负场 [0]: ") or 0)
    goals_for = _safe_int(input("  进球数 [0]: ") or 0)
    goals_against = _safe_int(input("  失球数 [0]: ") or 0)
    uefa_coeff = _safe_float(input("  UEFA系数 [50]: ") or 50, 50)
    ucl_titles = _safe_int(input("  欧冠冠军数 [0]: ") or 0)

    form_str = input("  最近5场(逗号分隔, 3=胜 1=平 0=负, 如3,3,1,0,3) [1,1,1,1,1]: ").strip()
    if form_str:
        form = [_safe_int(x.strip(), 1) for x in form_str.split(",")][:5]
        while len(form) < 5:
            form.append(1)
    else:
        form = [1, 1, 1, 1, 1]

    # 高级数据
    use_adv = input("  是否输入高级数据(xG/射正等)? [y/N]: ").strip().lower()
    xg = xga = shots_on_pct = possession = pass_acc = tackle_int = save_pct = 0.0

    if use_adv in ("y", "yes", "是"):
        print("  -- 高级数据 (直接回车跳过) --")
        xg = _safe_float(input("  场均xG [0]: ") or 0)
        xga = _safe_float(input("  场均xGA [0]: ") or 0)
        shots_on_pct = _safe_float(input("  射正率(0~1, 如0.35) [0]: ") or 0)
        possession = _safe_float(input("  控球率(0~1, 如0.55) [0]: ") or 0)
        pass_acc = _safe_float(input("  传球准确率(0~1, 如0.87) [0]: ") or 0)
        tackle_int = _safe_float(input("  场均抢断+拦截 [0]: ") or 0)
        save_pct = _safe_float(input("  扑救率(0~1, 如0.72) [0]: ") or 0)

    return Team(
        name=name, abbr=abbr, league_pts=league_pts,
        wins=wins, draws=draws, losses=losses,
        goals_for=goals_for, goals_against=goals_against,
        uefa_coeff=uefa_coeff, ucl_titles=ucl_titles, recent_form=form,
        xg=xg, xga=xga, shots_on_pct=shots_on_pct,
        possession=possession, pass_acc=pass_acc,
        tackle_int=tackle_int, save_pct=save_pct,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  输出格式化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def print_team_list(teams: Dict[str, Team], engine: PredictionEngine):
    """打印已加载的球队列表"""
    print(f"\n  已加载 {len(teams)} 支球队 | 模式: {'完整(含高级数据)' if engine.advanced_mode else '基础'}")
    print(f"  {'─' * 70}")
    print(f"  {'缩写':<6} {'名称':<16} {'积分':>4} {'战绩':>10} {'进/失':>8} {'实力':>6}", end="")
    if engine.advanced_mode:
        print(f" {'xG':>6} {'xGA':>6} {'射正%':>6}", end="")
    print()
    print(f"  {'─' * 70}")

    sorted_teams = sorted(teams.values(), key=lambda t: engine.strength(t), reverse=True)
    for t in sorted_teams:
        record = f"{t.wins}W {t.draws}D {t.losses}L"
        gd = f"{t.goals_for}/{t.goals_against}"
        s = engine.strength(t)
        line = f"  {t.abbr:<6} {t.name:<16} {t.league_pts:>4} {record:>10} {gd:>8} {s:>6.3f}"
        if engine.advanced_mode:
            xg_str = f"{t.xg:.2f}" if t.xg > 0 else "  - "
            xga_str = f"{t.xga:.2f}" if t.xga > 0 else "  - "
            sot_str = f"{t.shots_on_pct * 100:.0f}%" if t.shots_on_pct > 0 else " - "
            line += f" {xg_str:>6} {xga_str:>6} {sot_str:>6}"
        print(line)
    print()


def print_single_result(home: Team, away: Team, result: dict, neutral=False):
    """打印单场预测结果"""
    venue = "中立场地" if neutral else f"{home.name}主场"
    print(f"\n  ── 单场预测 ({venue}) ──")
    print(f"  {home.name} vs {away.name}")
    print(f"  ┌────────────┬──────────┐")
    print(f"  │ {home.name}胜  │ {result['home_win']:>6.1f}%  │")
    print(f"  │ 平局       │ {result['draw']:>6.1f}%  │")
    print(f"  │ {away.name}胜  │ {result['away_win']:>6.1f}%  │")
    print(f"  ├────────────┼──────────┤")
    print(f"  │ 预期比分   │ {result['avg_home_goals']:.1f} - {result['avg_away_goals']:.1f}  │")
    print(f"  └────────────┴──────────┘")

    # 最可能结果
    probs = [("home_win", result["home_win"], f"{home.name}胜"),
             ("draw", result["draw"], "平局"),
             ("away_win", result["away_win"], f"{away.name}胜")]
    best = max(probs, key=lambda x: x[1])
    print(f"  → 最可能结果: {best[2]} ({best[1]:.1f}%)")


def print_two_legs_result(team_a: Team, team_b: Team, result: dict, leg1=None):
    """打印两回合预测结果"""
    print(f"\n  ── 两回合淘汰赛预测 ──")
    if leg1:
        print(f"  首回合({team_a.name}主场): {leg1[0]}-{leg1[1]} [已踢]")
        print(f"  次回合({team_b.name}主场): 待预测")
    else:
        print(f"  首回合: {team_a.name}(主) vs {team_b.name}(客)")
        print(f"  次回合: {team_b.name}(主) vs {team_a.name}(客)")

    print(f"  ┌──────────────┬──────────┐")
    print(f"  │ {team_a.name}晋级  │ {result['a_advance']:>6.1f}%  │")
    print(f"  │ {team_b.name}晋级  │ {result['b_advance']:>6.1f}%  │")
    print(f"  ├──────────────┼──────────┤")
    print(f"  │ 预期总比分   │ {result['avg_agg_a']:.1f} - {result['avg_agg_b']:.1f}  │")
    print(f"  └──────────────┴──────────┘")

    fav = team_a.name if result["a_advance"] > result["b_advance"] else team_b.name
    fav_pct = max(result["a_advance"], result["b_advance"])
    print(f"  → 预测晋级: {fav} ({fav_pct:.1f}%)")


def print_tournament_result(teams: Dict[str, Team], results: dict):
    """打印锦标赛预测结果"""
    print(f"\n  ── 锦标赛夺冠预测 ──")
    print(f"  ┌────┬──────────────────┬──────────┬──────────┐")
    print(f"  │ #  │ 球队             │ 夺冠率   │ 进决赛   │")
    print(f"  ├────┼──────────────────┼──────────┼──────────┤")

    sorted_r = sorted(results.items(), key=lambda x: x[1]["champion"], reverse=True)
    for i, (abbr, r) in enumerate(sorted_r, 1):
        name = teams[abbr].name
        print(f"  │ {i:<2} │ {name:<16} │ {r['champion']:>6.1f}%  │ {r['final']:>6.1f}%  │")

    print(f"  └────┴──────────────────┴──────────┴──────────┘")

    if sorted_r:
        best_abbr, best_r = sorted_r[0]
        print(f"  → 预测冠军: {teams[best_abbr].name} ({best_r['champion']:.1f}%)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  选择球队的辅助函数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def pick_team(teams: Dict[str, Team], prompt: str) -> Optional[str]:
    """让用户选择一支球队（输入缩写或编号）"""
    abbr_input = input(f"  {prompt}: ").strip().upper()
    if not abbr_input:
        return None

    # 直接匹配缩写
    if abbr_input in teams:
        return abbr_input

    # 模糊匹配名称
    for abbr, t in teams.items():
        if abbr_input in t.name.upper() or abbr_input in t.abbr:
            return abbr

    print(f"  [错误] 找不到球队 '{abbr_input}'")
    print(f"  可用球队: {', '.join(teams.keys())}")
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  主菜单
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print()
    print("=" * 60)
    print("  足球比赛预测系统 v4.0")
    print("  Monte Carlo · Poisson · 自适应多因子模型")
    print("=" * 60)
    print()

    teams: Dict[str, Team] = {}
    engine: Optional[PredictionEngine] = None
    n_sims = 50000

    while True:
        print("─" * 50)
        print("  主菜单:")
        print("    [1] 加载球队数据 (从文件)")
        print("    [2] 手动输入球队")
        print("    [3] 查看已加载球队")
        print("    [4] 单场预测 (联赛)")
        print("    [5] 两回合预测 (欧冠淘汰赛)")
        print("    [6] 锦标赛模拟 (多队淘汰赛)")
        print("    [7] 设置模拟次数 (当前: {:,})".format(n_sims))
        print("    [8] 生成模板文件")
        print("    [q] 退出")
        print()

        choice = input("  请选择 > ").strip().lower()

        # ── 1. 加载数据 ──
        if choice == "1":
            filepath = input("  输入文件路径 (.csv 或 .json): ").strip()
            filepath = filepath.strip('"').strip("'")
            if not filepath:
                continue
            if not os.path.exists(filepath):
                print(f"  [错误] 文件不存在: {filepath}")
                continue

            loaded = load_teams_file(filepath)
            if not loaded:
                print("  [错误] 加载失败或文件为空")
                continue

            teams.update(loaded)
            engine = PredictionEngine(teams)
            print(f"  [OK] 已加载 {len(loaded)} 支球队 (总计 {len(teams)} 支)")
            print_team_list(teams, engine)

        # ── 2. 手动输入 ──
        elif choice == "2":
            t = input_team_manual()
            teams[t.abbr] = t
            engine = PredictionEngine(teams)
            print(f"  [OK] 已添加: {t.name} ({t.abbr})")

        # ── 3. 查看球队 ──
        elif choice == "3":
            if not teams:
                print("  [提示] 还没有加载任何球队，请先使用 [1] 或 [2]")
                continue
            engine = PredictionEngine(teams)
            print_team_list(teams, engine)

        # ── 4. 单场预测 ──
        elif choice == "4":
            if len(teams) < 2:
                print("  [提示] 至少需要2支球队，请先加载数据")
                continue
            engine = PredictionEngine(teams)

            print(f"\n  可用球队: {', '.join(teams.keys())}")
            home = pick_team(teams, "主队缩写")
            if not home:
                continue
            away = pick_team(teams, "客队缩写")
            if not away:
                continue
            if home == away:
                print("  [错误] 主客队不能相同")
                continue

            neutral_input = input("  中立场地? [y/N]: ").strip().lower()
            neutral = neutral_input in ("y", "yes", "是")

            print(f"\n  模拟中... ({n_sims:,} 次)")
            t0 = time.time()
            result = engine.predict_single(home, away, n_sims, neutral)
            elapsed = time.time() - t0
            print(f"  完成 ({elapsed:.2f}s)")

            print_single_result(teams[home], teams[away], result, neutral)

        # ── 5. 两回合预测 ──
        elif choice == "5":
            if len(teams) < 2:
                print("  [提示] 至少需要2支球队，请先加载数据")
                continue
            engine = PredictionEngine(teams)

            print(f"\n  可用球队: {', '.join(teams.keys())}")
            print("  (team_a 先主后客)")
            ta = pick_team(teams, "球队A缩写(首回合主场)")
            if not ta:
                continue
            tb = pick_team(teams, "球队B缩写(首回合客场)")
            if not tb:
                continue
            if ta == tb:
                print("  [错误] 两队不能相同")
                continue

            leg1_input = input("  首回合已踢? 输入比分如 '2-1'，未踢直接回车: ").strip()
            leg1 = None
            if leg1_input and "-" in leg1_input:
                parts = leg1_input.split("-")
                try:
                    leg1 = (int(parts[0].strip()), int(parts[1].strip()))
                    print(f"  首回合比分: {teams[ta].name} {leg1[0]} - {leg1[1]} {teams[tb].name}")
                except ValueError:
                    print("  [警告] 比分格式错误，将模拟两回合")

            print(f"\n  模拟中... ({n_sims:,} 次)")
            t0 = time.time()
            result = engine.predict_two_legs(ta, tb, n_sims, leg1)
            elapsed = time.time() - t0
            print(f"  完成 ({elapsed:.2f}s)")

            print_two_legs_result(teams[ta], teams[tb], result, leg1)

        # ── 6. 锦标赛模拟 ──
        elif choice == "6":
            if len(teams) < 2:
                print("  [提示] 至少需要2支球队，请先加载数据")
                continue
            engine = PredictionEngine(teams)

            print(f"\n  可用球队: {', '.join(teams.keys())}")
            print("  输入对阵（每行一组，格式: A,B 用缩写，空行结束）:")
            print("  例:  PSG,BMU")
            print("       ATM,ARS")
            print()

            matchups = []
            while True:
                line = input("  对阵: ").strip()
                if not line:
                    break
                parts = [x.strip().upper() for x in line.split(",")]
                if len(parts) != 2:
                    print("  [错误] 格式应为: A,B")
                    continue
                a, b = parts
                if a not in teams:
                    print(f"  [错误] 找不到 '{a}'")
                    continue
                if b not in teams:
                    print(f"  [错误] 找不到 '{b}'")
                    continue
                matchups.append((a, b))
                print(f"    + {teams[a].name} vs {teams[b].name}")

            if len(matchups) < 1:
                print("  [错误] 至少需要一组对阵")
                continue

            # 检查是否是2的幂
            n_matchups = len(matchups)
            if n_matchups & (n_matchups - 1) != 0:
                print(f"  [警告] {n_matchups} 组对阵不是2的幂，只预测第一轮晋级")

            legs_input = input("  两回合制? [Y/n]: ").strip().lower()
            two_legs = legs_input not in ("n", "no", "否")

            print(f"\n  模拟中... ({n_sims:,} 次)")
            t0 = time.time()
            result = engine.predict_tournament(matchups, n_sims, two_legs)
            elapsed = time.time() - t0
            print(f"  完成 ({elapsed:.2f}s)")

            print_tournament_result(teams, result)

        # ── 7. 设置模拟次数 ──
        elif choice == "7":
            try:
                new_n = int(input(f"  输入模拟次数 (当前 {n_sims:,}): ").strip())
                if new_n < 100:
                    print("  [警告] 最少100次")
                    new_n = 100
                elif new_n > 500000:
                    print("  [警告] 最多500,000次")
                    new_n = 500000
                n_sims = new_n
                print(f"  [OK] 已设置为 {n_sims:,} 次")
            except ValueError:
                print("  [错误] 请输入整数")

        # ── 8. 生成模板 ──
        elif choice == "8":
            generate_template()

        # ── 退出 ──
        elif choice in ("q", "quit", "exit"):
            print("  再见!")
            break

        else:
            print("  [提示] 无效选项，请输入 1-8 或 q")

        print()


if __name__ == "__main__":
    main()
