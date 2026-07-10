"""
① データ整形スクリプト

crawler.py で収集した data/ 以下のJSONを
1行1艇（6艇×レース数）の学習用CSVに変換する。

出力: dataset/training_data.csv

使い方:
  python build_dataset.py
  python build_dataset.py --data-dir data --out dataset/training_data.csv
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ─────────────────────────────────────────────
# 特徴量定義
# ─────────────────────────────────────────────
COURSE_WIN_RATE = {1: 0.555, 2: 0.154, 3: 0.114, 4: 0.087, 5: 0.057, 6: 0.033}
RANK_MAP = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}

def dedup_racers(racers: list) -> list:
    """
    クローラーの不具合で1レースに同じ艇が重複(空データ含む)して入るため、
    laneごとに「中身のある艇(win_rate_allがある)」を優先して1つだけ残す。
    これで18艇などの破損レコードを正しい6艇に復元する。
    """
    best = {}
    for r in racers:
        lane = r.get("lane")
        if lane is None:
            continue
        has_data = r.get("win_rate_all") is not None
        if lane not in best:
            best[lane] = r
        elif best[lane].get("win_rate_all") is None and has_data:
            best[lane] = r
    return [best[l] for l in sorted(best.keys())]

def load_exhibition_data(path: str = "dataset/exhibition_data.csv") -> dict:
    """展示データCSVを読んで {(race_date, venue_code, race_no, lane): {...}} の辞書を返す"""
    from pathlib import Path
    ex_map = {}
    p = Path(path)
    if not p.exists():
        print(f"⚠ 展示データ {path} が見つかりません。展示なしで続行します。")
        return ex_map
    df = pd.read_csv(p, dtype={"venue_code": str})
    for _, r in df.iterrows():
        key = (str(r["race_date"]), str(r["venue_code"]).zfill(2), int(r["race_no"]), int(r["lane"]))
        ex_map[key] = {
            "exhibition_time": r.get("exhibition_time"),
            "start_course": r.get("start_course"),
            "start_st": r.get("start_st"),
            "wind_speed": r.get("wind_speed"),
            "wave_height": r.get("wave_height"),
        }
    print(f"📊 展示データ読み込み: {len(ex_map)}件")
    return ex_map

def racer_to_row(racer: dict, result: dict | None, race_meta: dict, ex_map: dict = None) -> dict | None:
    """1艇分のデータを特徴量rowに変換する"""
    lane = racer.get("lane")
    if not lane:
        return None
    # 着順（目的変数）
    arrival = result.get("arrival", []) if result else []
    if lane in arrival:
        finish = arrival.index(lane) + 1   # 1〜6
    else:
        finish = None  # 結果未確定・欠場など
    row = {
        # メタ情報
        "race_date":  race_meta["race_date"],
        "venue_code": race_meta["venue_code"],
        "race_no":    race_meta["race_no"],
        "lane":       lane,
        "racer_no":   racer.get("racer_no", ""),
        # 目的変数
        "finish":     finish,           # 着順 (1〜6)
        "is_win":     1 if finish == 1 else (0 if finish else None),   # 1着か
        "is_top3":    1 if finish and finish <= 3 else (0 if finish else None),
        # コース特徴
        "course_base_winrate": COURSE_WIN_RATE.get(lane, 0.03),
        # 選手特徴
        "rank_num":      RANK_MAP.get(racer.get("rank", "B2"), 1),
        "age":           racer.get("age"),
        "weight":        racer.get("weight"),
        "flying_count":  racer.get("flying_count", 0),
        "late_count":    racer.get("late_count", 0),
        "avg_start_time": racer.get("avg_start_time"),
        # 全国成績
        "win_rate_all":  racer.get("win_rate_all"),
        "win_rate_2":    racer.get("win_rate_2"),
        "win_rate_3":    racer.get("win_rate_3"),
        # 当地成績
        "local_win_rate":   racer.get("local_win_rate"),
        "local_win_rate_2": racer.get("local_win_rate_2"),
        "local_win_rate_3": racer.get("local_win_rate_3"),
        # 機材
        "motor_2rate": racer.get("motor_2rate"),
        "boat_2rate":  racer.get("boat_2rate"),
        # 展示・進入・気象（あれば結合、なければNone）
        "exhibition_time": None,
        "start_course":    None,
        "start_st":        None,
        "ex_wind_speed":   None,
        "ex_wave_height":  None,
    }

    # 展示データを結合
    if ex_map:
        ex_key = (race_meta["race_date"], race_meta["venue_code"], race_meta["race_no"], lane)
        ex = ex_map.get(ex_key)
        if ex:
            row["exhibition_time"] = ex.get("exhibition_time")
            row["start_course"]    = ex.get("start_course")
            row["start_st"]        = ex.get("start_st")
            row["ex_wind_speed"]   = ex.get("wind_speed")
            row["ex_wave_height"]  = ex.get("wave_height")

    return row


def build_dataset(data_dir: Path, out_path: Path):
    json_files = sorted(data_dir.rglob("*.json"))
    if not json_files:
        print(f"❌ {data_dir} にJSONファイルが見つかりません")
        return
    print(f"\n📂 JSONファイル数: {len(json_files)}")

    # 展示データを読み込む（日付・会場・レース・艇番で引けるようにする）
    ex_map = load_exhibition_data()

    rows = []
    for jf in tqdm(json_files, desc="整形中", ncols=70):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        # ファイル名から日付・会場を取得（例: 20260501_桐生.json）
        stem = jf.stem  # "20260501_桐生"
        parts = stem.split("_", 1)
        race_date = parts[0] if len(parts) >= 1 else ""
        venue_name = parts[1] if len(parts) >= 2 else ""
        for race_no_str, race in data.items():
            if not isinstance(race, dict):
                continue
            race_no = race.get("race_no", race_no_str)
            racers  = race.get("racers", [])
            racers  = dedup_racers(racers)   # 重複・空データを除去して正しい6艇に復元
            result  = race.get("result")
            meta = {
                "race_date":  race_date,
                "venue_code": result.get("venue_code", "") if result else "",
                "race_no":    race_no,
            }
            for racer in racers:
                row = racer_to_row(racer, result, meta, ex_map)
                if row:
                    rows.append(row)
    if not rows:
        print("❌ 変換できるデータがありませんでした")
        return
    df = pd.DataFrame(rows)
    # 数値型に変換
    num_cols = [c for c in df.columns if c not in ("race_date", "venue_code", "racer_no")]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 完了!")
    print(f"   行数  : {len(df):,} 行（1行=1艇）")
    print(f"   列数  : {len(df.columns)} 列")
    print(f"   保存先: {out_path.resolve()}")
    print(f"\n   着順データあり: {df['finish'].notna().sum():,} 件")
    print(f"   着順データなし: {df['finish'].isna().sum():,} 件（取得時未確定など）")

    # 展示タイムが結合できた件数を確認
    if "exhibition_time" in df.columns:
        ex_count = df["exhibition_time"].notna().sum()
        print(f"   展示タイムあり: {ex_count:,} 件（結合成功）")
    return df


def main():
    parser = argparse.ArgumentParser(description="競艇データ整形スクリプト")
    parser.add_argument("--data-dir", default="data", help="crawlerの保存先ディレクトリ")
    parser.add_argument("--out", default="dataset/training_data.csv", help="出力CSVパス")
    args = parser.parse_args()

    build_dataset(Path(args.data_dir), Path(args.out))


if __name__ == "__main__":
    main()
