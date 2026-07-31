"""
full_odds_cache.json の対象レース(280レース)について、公式結果ページ(raceresult)から
実際の3連単確定払戻を取得し、キャッシュ(official_payouts_cache.json)に保存する。

odds3tページの過去レースオッズが実際の確定払戻と大きく乖離する(平均6.9倍)ことが
判明したため、勝ち組み合わせの払戻だけは公式結果ページから正確な値を取得し直す。

使い方:
  python fetch_official_payouts.py
"""
import json
from pathlib import Path
from datetime import datetime as dt

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from boatrace_scraper import BoatraceScraper

ODDS_CACHE_PATH = Path("full_odds_cache.json")
OUT_CACHE_PATH = Path("official_payouts_cache.json")


def load_cache(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_cache(path: Path, cache: dict):
    path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def main():
    odds_cache = load_cache(ODDS_CACHE_PATH)
    targets = list(odds_cache.keys())
    print(f"対象レース数: {len(targets)}")

    cache = load_cache(OUT_CACHE_PATH)
    print(f"キャッシュ済み: {len(cache)}件")

    todo = [k for k in targets if k not in cache]
    print(f"新規取得が必要: {len(todo)}件")
    if not todo:
        print("すべてキャッシュ済みです。")
        return

    sc = BoatraceScraper(delay=1.0)
    if not sc.login():
        print("❌ ログイン失敗")
        return
    print("✅ ログイン成功")

    n_ok, n_fail = 0, 0
    for i, key in enumerate(todo):
        race_date_str, venue_code, race_no = key.split("_")
        race_date_obj = dt.strptime(race_date_str, "%Y%m%d").date()
        try:
            result = sc.get_result(race_date_obj, venue_code, int(race_no))
        except Exception as e:
            print(f"[{i+1}/{len(todo)}] {key}: 取得エラー {e}")
            n_fail += 1
            continue
        if not result or not result.arrival or len(result.arrival) < 3:
            print(f"[{i+1}/{len(todo)}] {key}: 結果取得失敗")
            n_fail += 1
            continue
        actual = "-".join(str(x) for x in result.arrival[:3])
        payout = (result.payouts or {}).get(f"3連単_{actual}")
        cache[key] = {"actual": actual, "payout": payout}
        n_ok += 1
        if (i + 1) % 10 == 0:
            save_cache(OUT_CACHE_PATH, cache)
            print(f"[{i+1}/{len(todo)}] 進捗保存... 成功{n_ok}件 失敗{n_fail}件")

    save_cache(OUT_CACHE_PATH, cache)
    print(f"\n完了。成功{n_ok}件 失敗{n_fail}件。キャッシュ合計{len(cache)}件")


if __name__ == "__main__":
    main()
