"""
③ 予想適用スクリプト

当日の出走表を取得 → 学習済みモデルで予想 → 買い目を出力する。

使い方:
  # 今日の全開催会場を自動検出して予想
  python predict_today.py

  # 会場・レースを指定
  python predict_today.py --venue 01 --race 1

  # 全レースまとめて予想してJSONに保存
  python predict_today.py --venue 12 --out predictions/今日の予想.json
"""

import argparse
import json
import pickle
import logging
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from boatrace_scraper import BoatraceScraper, RacerInfo, VENUE_MAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "lane", "course_base_winrate", "rank_num", "age", "weight",
    "flying_count", "late_count", "avg_start_time",
    "win_rate_all", "win_rate_2", "win_rate_3",
    "local_win_rate", "local_win_rate_2", "local_win_rate_3",
    "motor_2rate", "boat_2rate",
]

COURSE_WIN_RATE = {1: 0.555, 2: 0.154, 3: 0.114, 4: 0.087, 5: 0.057, 6: 0.033}
RANK_MAP = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}


# ─────────────────────────────────────────────
# モデルロード
# ─────────────────────────────────────────────
class ModelPredictor:
    def __init__(self, model_dir: Path = Path("models")):
        win_path  = model_dir / "model_win.pkl"
        top3_path = model_dir / "model_top3.pkl"

        if not win_path.exists() or not top3_path.exists():
            raise FileNotFoundError(
                f"モデルが見つかりません: {model_dir}\n"
                "先に python train_model.py を実行してください。"
            )

        with open(win_path,  "rb") as f: self.model_win  = pickle.load(f)
        with open(top3_path, "rb") as f: self.model_top3 = pickle.load(f)
        logger.info(f"モデル読み込み完了: {model_dir}")

    def racer_to_features(self, racer: RacerInfo) -> dict:
        return {
            "lane":               racer.lane,
            "course_base_winrate": COURSE_WIN_RATE.get(racer.lane, 0.03),
            "rank_num":           RANK_MAP.get(racer.rank, 1),
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

    def predict_race(
        self,
        racers: list[RacerInfo],
        venue_name: str = "",
        race_no: int = 0,
        race_date: str = "",
    ) -> dict:
        """1レース分の予想を返す"""
        if not racers:
            return {}

        rows = [self.racer_to_features(r) for r in racers]
        X = pd.DataFrame(rows, columns=FEATURE_COLS).apply(pd.to_numeric, errors="coerce")

        prob_win  = self.model_win.predict(X)
        prob_top3 = self.model_top3.predict(X)

        results = []
        for racer, pw, pt in zip(racers, prob_win, prob_top3):
            results.append({
                "lane":       racer.lane,
                "name":       racer.name,
                "rank":       racer.rank,
                "prob_win":   round(float(pw),  4),
                "prob_top3":  round(float(pt),  4),
            })

        # スコア順にソート
        results_sorted = sorted(results, key=lambda x: x["prob_win"], reverse=True)
        for i, r in enumerate(results_sorted):
            r["predicted_rank"] = i + 1

        # 推奨買い目（確率上位3艇）
        top3 = [r["lane"] for r in results_sorted[:3]]
        top2 = top3[:2]
        a, b, c = top3[0], top3[1], top3[2]

        buy = {
            "単勝":   [str(a)],
            "複勝":   [str(a), str(b)],
            "2連単":  [f"{a}-{b}", f"{a}-{c}", f"{b}-{a}"],
            "2連複":  [f"{min(a,b)}-{max(a,b)}", f"{min(a,c)}-{max(a,c)}"],
            "3連単":  [f"{a}-{b}-{c}", f"{a}-{c}-{b}", f"{b}-{a}-{c}"],
            "3連複":  ["-".join(map(str, sorted(top3)))],
        }

        return {
            "race_date":   race_date,
            "venue_name":  venue_name,
            "race_no":     race_no,
            "lane_scores": sorted(results, key=lambda x: x["lane"]),  # 枠番順
            "ranking":     results_sorted,
            "buy":         buy,
        }

    def print_result(self, pred: dict):
        if not pred:
            return
        print(f"\n{'='*55}")
        print(f"  {pred['venue_name']} {pred['race_no']}R  ({pred['race_date']})")
        print(f"{'='*55}")
        print(f"  {'枠':>2} {'選手名':<10} {'級':>3}  {'1着確率':>7}  {'3着内確率':>8}  {'予想順'}")
        print(f"  {'-'*50}")
        for r in pred["lane_scores"]:
            print(
                f"  {r['lane']:>2} {r['name']:<10} {r['rank']:>3} "
                f" {r['prob_win']:>7.1%}  {r['prob_top3']:>8.1%}  "
                f"  {r['predicted_rank']}着予想"
            )
        print(f"\n  【推奨買い目】")
        for bet_type, combos in pred["buy"].items():
            print(f"    {bet_type:>4}: {' / '.join(combos)}")
        print(f"{'='*55}")


# ─────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="競艇 当日予想")
    parser.add_argument("--venue", help="会場コード 例: 01")
    parser.add_argument("--race",  type=int, help="レース番号 1〜12")
    parser.add_argument("--date",  help="日付 YYYYMMDD（省略時=今日）")
    parser.add_argument("--max-race", type=int, default=12)
    parser.add_argument("--out",   help="出力JSONパス")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--delay", type=float, default=1.5)
    args = parser.parse_args()

    # モデルロード
    try:
        predictor = ModelPredictor(Path(args.model_dir))
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    scraper = BoatraceScraper(delay=args.delay)

    if args.date:
        from datetime import datetime
        race_date = datetime.strptime(args.date, "%Y%m%d").date()
    else:
        race_date = date.today()

    date_str = race_date.strftime("%Y%m%d")
    all_preds = []

    if args.venue:
        venues = [args.venue.zfill(2)]
    else:
        # 省略時は主要会場（実際の開催は日によって変わる）
        venues = [f"{i:02d}" for i in range(1, 25)]

    for venue in venues:
        vname = VENUE_MAP.get(venue, venue)
        race_range = [args.race] if args.race else range(1, args.max_race + 1)

        for rno in race_range:
            racers = scraper.get_racelist(race_date, venue, rno)
            if not racers:
                continue

            pred = predictor.predict_race(
                racers,
                venue_name=vname,
                race_no=rno,
                race_date=date_str,
            )
            predictor.print_result(pred)
            all_preds.append(pred)

    if args.out and all_preds:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(all_preds, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n💾 予想を保存しました: {out.resolve()}")


if __name__ == "__main__":
    main()
