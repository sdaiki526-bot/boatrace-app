"""
競艇予想ロジックモジュール

スコアリング方式で各艇を採点し、予想順位・推奨買い目を出力する。

採点項目（カスタマイズ可能）:
  - コース別基礎勝率（1コースが圧倒的有利）
  - 選手全国勝率・当地勝率
  - 級別（A1 > A2 > B1 > B2）
  - モーター2連率
  - ボート2連率
  - 平均スタートタイム（小さいほど良い）
  - F/L数（多いと減点）
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

from boatrace_scraper import RacerInfo

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# コース別期待勝率（全国統計ベース）
# ─────────────────────────────────────────────
COURSE_WIN_RATE = {
    1: 0.555,
    2: 0.154,
    3: 0.114,
    4: 0.087,
    5: 0.057,
    6: 0.033,
}

# 級別スコア
RANK_SCORE = {
    "A1": 10.0,
    "A2":  6.0,
    "B1":  3.0,
    "B2":  0.0,
}

# ─────────────────────────────────────────────
# 設定（重み）
# ─────────────────────────────────────────────
@dataclass
class PredictorConfig:
    """各スコアの重み設定。合計が100になるよう調整推奨。"""
    w_course:      float = 35.0   # コース有利不利
    w_win_rate:    float = 20.0   # 全国勝率
    w_local_rate:  float = 15.0   # 当地勝率
    w_rank:        float = 10.0   # 級別
    w_motor:       float = 10.0   # モーター2連率
    w_boat:        float =  5.0   # ボート2連率
    w_start:       float =  5.0   # 平均ST（小さいほど良）

    # ペナルティ
    penalty_per_flying: float = 3.0   # F1本あたり減点
    penalty_per_late:   float = 1.5   # L1本あたり減点

    # 基準値（正規化用）
    win_rate_max:   float = 8.0   # 勝率の最大想定値
    motor_rate_max: float = 60.0  # モーター2連率の最大想定値（%）
    boat_rate_max:  float = 60.0
    st_worst:       float = 0.25  # STが遅いとみなすしきい値


# ─────────────────────────────────────────────
# 予想結果
# ─────────────────────────────────────────────
@dataclass
class LaneScore:
    lane: int
    name: str
    rank: str
    total_score: float
    breakdown: dict = field(default_factory=dict)  # 各項目のスコア内訳
    predicted_rank: int = 0  # 予想順位（1〜6）


@dataclass
class PredictionResult:
    race_date: str
    venue_name: str
    race_no: int
    scores: list[LaneScore] = field(default_factory=list)

    # 推奨買い目
    tansho: str = ""          # 単勝（1点）
    fukusho: list = field(default_factory=list)  # 複勝（2点）
    niren_tan: list = field(default_factory=list)  # 2連単（3点）
    niren_fuku: list = field(default_factory=list) # 2連複（3点）
    sanren_tan: list = field(default_factory=list) # 3連単（3点）
    sanren_fuku: str = ""     # 3連複（1点）

    def summary(self) -> str:
        lines = [
            f"{'='*50}",
            f"  {self.venue_name} {self.race_no}R 予想結果  ({self.race_date})",
            f"{'='*50}",
            f"{'順':>3} {'枠':>3} {'選手名':<10} {'級':>3}  {'スコア':>7}",
            f"{'-'*50}",
        ]
        for s in self.scores:
            lines.append(
                f"{s.predicted_rank:>3}着 {s.lane:>2}枠 {s.name:<10} {s.rank:>3}  {s.total_score:>7.2f}"
            )
        lines += [
            f"{'─'*50}",
            f"【推奨買い目】",
            f"  単勝    : {self.tansho}",
            f"  複勝    : {' / '.join(self.fukusho)}",
            f"  2連単   : {' / '.join(self.niren_tan)}",
            f"  2連複   : {' / '.join(self.niren_fuku)}",
            f"  3連単   : {' / '.join(self.sanren_tan)}",
            f"  3連複   : {self.sanren_fuku}",
            f"{'='*50}",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────
# 予想エンジン
# ─────────────────────────────────────────────
class BoatracePredictor:
    def __init__(self, config: PredictorConfig = None):
        self.cfg = config or PredictorConfig()

    # ── スコア計算 ─────────────────────────────
    def _score_racer(self, racer: RacerInfo) -> tuple[float, dict]:
        cfg = self.cfg
        breakdown = {}

        # 1. コーススコア（0〜1 を重みでスケール）
        course_raw = COURSE_WIN_RATE.get(racer.lane, 0.03)
        # 最大(1コース=0.555)を1.0として正規化
        course_score = (course_raw / 0.555) * cfg.w_course
        breakdown["コース"] = round(course_score, 3)

        # 2. 全国勝率（0〜win_rate_max を 0〜w_win_rate にマップ）
        wr = racer.win_rate_all or 0.0
        wr_score = min(wr / cfg.win_rate_max, 1.0) * cfg.w_win_rate
        breakdown["全国勝率"] = round(wr_score, 3)

        # 3. 当地勝率
        lr = racer.local_win_rate or 0.0
        lr_score = min(lr / cfg.win_rate_max, 1.0) * cfg.w_local_rate
        breakdown["当地勝率"] = round(lr_score, 3)

        # 4. 級別
        rank_score = RANK_SCORE.get(racer.rank, 0.0) / 10.0 * cfg.w_rank
        breakdown["級別"] = round(rank_score, 3)

        # 5. モーター2連率
        m2 = racer.motor_2rate or 30.0  # デフォルト平均値
        motor_score = min(m2 / cfg.motor_rate_max, 1.0) * cfg.w_motor
        breakdown["モーター"] = round(motor_score, 3)

        # 6. ボート2連率
        b2 = racer.boat_2rate or 30.0
        boat_score = min(b2 / cfg.boat_rate_max, 1.0) * cfg.w_boat
        breakdown["ボート"] = round(boat_score, 3)

        # 7. 平均ST（小さいほど良い。0.10が最速、0.25が遅め）
        st = racer.avg_start_time
        if st is not None and 0 <= st <= cfg.st_worst:
            st_score = (1.0 - st / cfg.st_worst) * cfg.w_start
        else:
            st_score = 0.0
        breakdown["ST"] = round(st_score, 3)

        # ペナルティ
        penalty = (
            (racer.flying_count or 0) * cfg.penalty_per_flying
            + (racer.late_count or 0) * cfg.penalty_per_late
        )
        breakdown["ペナルティ"] = round(-penalty, 3)

        total = sum(breakdown.values())
        return round(total, 3), breakdown

    # ── 予想メイン ─────────────────────────────
    def predict(
        self,
        racers: list[RacerInfo],
        race_date: str = "",
        venue_name: str = "",
        race_no: int = 0,
    ) -> PredictionResult:
        """
        Parameters
        ----------
        racers     : get_racelist() の戻り値
        race_date  : 表示用（"YYYYMMDD"）
        venue_name : 表示用
        race_no    : 表示用

        Returns
        -------
        PredictionResult
        """
        if not racers:
            logger.warning("選手データが空です")
            return PredictionResult(race_date, venue_name, race_no)

        scores: list[LaneScore] = []
        for r in racers:
            total, bd = self._score_racer(r)
            scores.append(LaneScore(
                lane=r.lane, name=r.name, rank=r.rank,
                total_score=total, breakdown=bd,
            ))

        # 降順でソートして予想順位を付与
        scores_sorted = sorted(scores, key=lambda s: s.total_score, reverse=True)
        for i, s in enumerate(scores_sorted):
            s.predicted_rank = i + 1

        # 元の枠番順に戻す（表示用）
        scores_by_lane = sorted(scores_sorted, key=lambda s: s.lane)

        # 買い目生成（上位3艇を使用）
        top3 = [s.lane for s in scores_sorted[:3]]
        top2 = top3[:2]

        result = PredictionResult(
            race_date=race_date,
            venue_name=venue_name,
            race_no=race_no,
            scores=scores_by_lane,
        )

        result.tansho   = str(top3[0])
        result.fukusho  = [str(top3[0]), str(top3[1])]
        result.niren_tan = [
            f"{top2[0]}-{top2[1]}",
            f"{top2[1]}-{top2[0]}",
            f"{top3[0]}-{top3[2]}",
        ]
        result.niren_fuku = [
            f"{min(top2)}-{max(top2)}",
            f"{min(top3[0], top3[2])}-{max(top3[0], top3[2])}",
            f"{min(top3[1], top3[2])}-{max(top3[1], top3[2])}",
        ]
        a, b, c = top3
        result.sanren_tan = [
            f"{a}-{b}-{c}",
            f"{a}-{c}-{b}",
            f"{b}-{a}-{c}",
        ]
        s3 = sorted(top3)
        result.sanren_fuku = f"{s3[0]}-{s3[1]}-{s3[2]}"

        return result

    # ── 複数レース一括予想 ──────────────────────
    def predict_all_races(self, races_data: list[dict]) -> list[PredictionResult]:
        """
        get_all_races() の戻り値を受け取って全レース予想する。

        Parameters
        ----------
        races_data : [{"race_no": 1, "racers": [...], ...}, ...]
        """
        results = []
        for race in races_data:
            rno = race.get("race_no", 0)
            raw_racers = race.get("racers", [])

            # dict → RacerInfo に復元
            racers = []
            for d in raw_racers:
                try:
                    racers.append(RacerInfo(**d))
                except TypeError:
                    pass

            odds = race.get("odds", {})
            vname = odds.get("venue_name", "")
            rdate = odds.get("race_date", "")

            pred = self.predict(racers, rdate, vname, rno)
            results.append(pred)
        return results

    # ── 保存 ───────────────────────────────────
    @staticmethod
    def save_predictions(results: list[PredictionResult], path: str):
        data = [asdict(r) for r in results]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"予想結果 → {path}")
