"""
過去レースの3連単全120通りオッズを取得し、キャッシュ(JSON)に保存するスクリプト。

対象レース: prediction_recordsでhitが確定済み、かつrace_dateが
dataset/training_data.csvの最終日付以前(=モデルスコアを再計算できる)のレースのうち、
直近N件(デフォルト400件)。

キャッシュ済みのレースは再取得しない(中断・再実行に対応)。

使い方:
  python fetch_full_odds_cache.py [--limit 400]
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from boatrace_scraper import BoatraceScraper

CACHE_PATH = Path("full_odds_cache.json")


def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict):
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def get_training_data_max_date() -> str:
    df = pd.read_csv("dataset/training_data.csv", encoding="utf-8-sig", low_memory=False,
                      usecols=["race_date", "finish"])
    df = df[df["finish"].notna()]
    return df["race_date"].astype(str).max()


def get_target_races(limit: int, max_date: str) -> list[dict]:
    from supabase import create_client
    supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    all_data = []
    offset = 0
    while True:
        res = (
            supabase.table("prediction_records")
            .select("race_date,venue_code,race_no,actual")
            .not_.is_("hit", "null")
            .lte("race_date", max_date)
            .order("race_date", desc=True)
            .range(offset, offset + 999)
            .execute()
        )
        if not res.data:
            break
        all_data.extend(res.data)
        if len(res.data) < 1000 or len(all_data) >= limit:
            break
        offset += 1000
    # 重複除去(race_date,venue_code,race_no)して直近順に並べ、limit件に絞る
    seen = set()
    targets = []
    for r in all_data:
        key = (r["race_date"], r["venue_code"], r["race_no"])
        if key in seen:
            continue
        seen.add(key)
        targets.append(r)
    targets.sort(key=lambda r: r["race_date"], reverse=True)
    return targets[:limit]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--delay", type=float, default=1.2)
    args = parser.parse_args()

    max_date = get_training_data_max_date()
    print(f"training_data.csvの最終日付: {max_date}")

    targets = get_target_races(args.limit, max_date)
    print(f"対象レース数: {len(targets)}")

    cache = load_cache()
    print(f"キャッシュ済み: {len(cache)}件")

    todo = []
    for r in targets:
        key = f"{r['race_date']}_{r['venue_code']}_{r['race_no']}"
        if key not in cache:
            todo.append(r)
    print(f"新規取得が必要: {len(todo)}件")

    if not todo:
        print("すべてキャッシュ済みです。取得をスキップします。")
        return

    from datetime import datetime as dt
    sc = BoatraceScraper(delay=args.delay)
    if not sc.login():
        print("❌ ログイン失敗。.envのBOATRACE_*を確認してください")
        return
    print("✅ ログイン成功")

    n_ok, n_fail = 0, 0
    for i, r in enumerate(todo):
        key = f"{r['race_date']}_{r['venue_code']}_{r['race_no']}"
        race_date_obj = dt.strptime(r["race_date"], "%Y%m%d").date()
        try:
            odds_map = sc._get_sanren_tan(race_date_obj, r["venue_code"], r["race_no"])
        except Exception as e:
            print(f"[{i+1}/{len(todo)}] {key}: 取得エラー {e}")
            n_fail += 1
            continue
        if not odds_map or len(odds_map) < 100:
            print(f"[{i+1}/{len(todo)}] {key}: オッズ不足({len(odds_map) if odds_map else 0}件) スキップ")
            n_fail += 1
            continue
        cache[key] = odds_map
        n_ok += 1
        if (i + 1) % 10 == 0:
            save_cache(cache)
            print(f"[{i+1}/{len(todo)}] 進捗保存... 成功{n_ok}件 失敗{n_fail}件")

    save_cache(cache)
    print(f"\n完了。成功{n_ok}件 失敗{n_fail}件。キャッシュ合計{len(cache)}件 ({CACHE_PATH.resolve()})")


if __name__ == "__main__":
    main()
