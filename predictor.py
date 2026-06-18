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

    # 展示タイム（任意・データがある場合のみ加算）
    w_exhibition:     float = 10.0  # 展示タイムの重み
    exhibition_range: float = 0.20  # 最速との差がこの値以上なら0点（秒）


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
    def _score_racer(self, racer: RacerInfo, exhibition_time: Optional[float] = None,
                      fastest_time: Optional[float] = None) -> tuple[float, dict]:
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

        # 8. 展示タイム（データがある場合のみ。最速艇との差分でスコア化）
        if exhibition_time is not None and fastest_time is not None:
            diff = exhibition_time - fastest_time  # 0 = 最速
            if diff <= 0:
                ex_score = cfg.w_exhibition
            elif diff >= cfg.exhibition_range:
                ex_score = 0.0
            else:
                ex_score = (1.0 - diff / cfg.exhibition_range) * cfg.w_exhibition
            breakdown["展示タイム"] = round(ex_score, 3)

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
        exhibition_times: Optional[dict] = None,
    ) -> PredictionResult:
        """
        Parameters
        ----------
        racers     : get_racelist() の戻り値
        race_date  : 表示用（"YYYYMMDD"）
        venue_name : 表示用
        race_no    : 表示用
        exhibition_times : {lane: exhibition_time(float)} の辞書（任意）
                           指定すると予想スコアに展示タイム補正を加える

        Returns
        -------
        PredictionResult
        """
        if not racers:
            logger.warning("選手データが空です")
            return PredictionResult(race_date, venue_name, race_no)

        # 展示タイムの最速値を算出
        fastest_time = None
        if exhibition_times:
            valid_times = [t for t in exhibition_times.values() if t is not None]
            if valid_times:
                fastest_time = min(valid_times)

        scores: list[LaneScore] = []
        for r in racers:
            ex_time = exhibition_times.get(r.lane) if exhibition_times else None
            total, bd = self._score_racer(r, exhibition_time=ex_time, fastest_time=fastest_time)
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


# ─────────────────────────────────────────────
# LightGBMモデル + 展示タイム補正を組み合わせた予想エンジン
# ─────────────────────────────────────────────
import pickle
import numpy as np
import pandas as pd

ML_FEATURE_COLS = [
    "lane", "rank_num", "age", "weight",
    "flying_count", "late_count", "avg_start_time",
    "win_rate_all", "win_rate_2", "win_rate_3",
    "local_win_rate", "local_win_rate_2", "local_win_rate_3",
    "motor_2rate", "boat_2rate",
]
ML_RANK_MAP = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}


class MLPredictor:
    """
    LightGBM学習済みモデル（model_win.pkl / model_top3.pkl）で
    各艇の1着確率・3着内確率を予測し、展示タイム補正を加味して
    PredictionResult（predictor.pyの既存フォーマット）を返す。

    既存の BoatracePredictor（ルールベース）と同じインターフェースで
    使えるよう predict() を実装している。
    """

    def __init__(self, model_dir: str | Path = "models",
                 exhibition_weight: float = 0.15,
                 exhibition_range: float = 0.20):
        model_dir = Path(model_dir)
        win_path = model_dir / "model_win.pkl"
        top3_path = model_dir / "model_top3.pkl"

        if not win_path.exists() or not top3_path.exists():
            raise FileNotFoundError(
                f"モデルが見つかりません: {model_dir}\n"
                "先に python train_model.py を実行してください。"
            )

        with open(win_path, "rb") as f:
            self.model_win = pickle.load(f)
        with open(top3_path, "rb") as f:
            self.model_top3 = pickle.load(f)

        # 展示タイム補正の重み（1着確率に対する加算比率の最大値）
        self.exhibition_weight = exhibition_weight
        self.exhibition_range = exhibition_range

    @staticmethod
    def _racer_to_features(racer: RacerInfo) -> dict:
        return {
            "lane":               racer.lane,
            "rank_num":           ML_RANK_MAP.get(racer.rank, 1),
            "age":                racer.age,
            "weight":             racer.weight,
            "flying_count":       racer.flying_count or 0,
            "late_count":         racer.late_count or 0,
            "avg_start_time":     racer.avg_start_time,
            "win_rate_all":       racer.win_rate_all,
            "win_rate_2":         racer.win_rate_2,
            "win_rate_3":         racer.win_rate_3,
            "local_win_rate":     racer.local_win_rate,
            "local_win_rate_2":   racer.local_win_rate_2,
            "local_win_rate_3":   racer.local_win_rate_3,
            "motor_2rate":        racer.motor_2rate,
            "boat_2rate":         racer.boat_2rate,
        }

    def predict(
        self,
        racers: list[RacerInfo],
        race_date: str = "",
        venue_name: str = "",
        race_no: int = 0,
        exhibition_times: Optional[dict] = None,
    ) -> PredictionResult:
        if not racers:
            logger.warning("選手データが空です")
            return PredictionResult(race_date, venue_name, race_no)

        rows = [self._racer_to_features(r) for r in racers]
        X = pd.DataFrame(rows, columns=ML_FEATURE_COLS).apply(pd.to_numeric, errors="coerce")

        prob_win = self.model_win.predict(X)
        prob_top3 = self.model_top3.predict(X)

        # 展示タイム補正の準備
        fastest_time = None
        if exhibition_times:
            valid_times = [t for t in exhibition_times.values() if t is not None]
            if valid_times:
                fastest_time = min(valid_times)

        scores: list[LaneScore] = []
        for racer, pw, pt in zip(racers, prob_win, prob_top3):
            breakdown = {
                "1着確率": round(float(pw) * 100, 3),
                "3着内確率": round(float(pt) * 100, 3),
            }
            total = float(pw) * 100  # 基本スコアは1着確率(%)

            # 展示タイム補正
            if exhibition_times and fastest_time is not None:
                ex_time = exhibition_times.get(racer.lane)
                if ex_time is not None:
                    diff = ex_time - fastest_time
                    max_bonus = self.exhibition_weight * 100  # %スケールに合わせる
                    if diff <= 0:
                        ex_score = max_bonus
                    elif diff >= self.exhibition_range:
                        ex_score = 0.0
                    else:
                        ex_score = (1.0 - diff / self.exhibition_range) * max_bonus
                    breakdown["展示タイム補正"] = round(ex_score, 3)
                    total += ex_score

            scores.append(LaneScore(
                lane=racer.lane, name=racer.name, rank=racer.rank,
                total_score=round(total, 3), breakdown=breakdown,
            ))

        # 降順でソートして予想順位を付与
        scores_sorted = sorted(scores, key=lambda s: s.total_score, reverse=True)
        for i, s in enumerate(scores_sorted):
            s.predicted_rank = i + 1

        scores_by_lane = sorted(scores_sorted, key=lambda s: s.lane)

        top3 = [s.lane for s in scores_sorted[:3]]
        top2 = top3[:2]

        result = PredictionResult(
            race_date=race_date,
            venue_name=venue_name,
            race_no=race_no,
            scores=scores_by_lane,
        )

        result.tansho = str(top3[0])
        result.fukusho = [str(top3[0]), str(top3[1])]
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
        result.sanren_fuku = "-".join(map(str, sorted(top3)))

        return result