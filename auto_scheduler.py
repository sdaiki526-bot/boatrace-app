"""
競艇 毎日自動バッチ

スケジュール:
  08:00 → 今日の出走表を一括取得 + 一括予想 + キャッシュ保存
  23:00 → 昨日の結果を一括取得 + 的中判定を自動更新

起動方法:
  python auto_scheduler.py

  # 今すぐテスト実行
  python auto_scheduler.py --run-morning   # 朝の処理をテスト
  python auto_scheduler.py --run-night     # 夜の処理をテスト
"""

import argparse
import json
import logging
import time
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

import schedule

import sys
sys.path.insert(0, str(Path(__file__).parent))

from boatrace_scraper import BoatraceScraper, VENUE_MAP
from predictor import BoatracePredictor
from crawler import get_holding_venues

# ─────────────────────────────────────────────
# ロギング
# ─────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "auto_scheduler.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

RECORD_FILE = Path("prediction_records.json")


# ─────────────────────────────────────────────
# 記録ユーティリティ
# ─────────────────────────────────────────────
def load_records():
    if RECORD_FILE.exists():
        return json.loads(RECORD_FILE.read_text(encoding="utf-8"))
    return []

def save_records(records):
    RECORD_FILE.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

def save_record(record):
    records = load_records()
    key = f"{record['race_date']}_{record['venue_code']}_{record['race_no']}"
    records = [r for r in records if f"{r['race_date']}_{r['venue_code']}_{r['race_no']}" != key]
    records.append(record)
    save_records(records)

def save_cache(today_racers, target_date):
    cache_file = Path(f"cache_racers_{target_date.strftime('%Y%m%d')}.json")
    serializable = {}
    for venue, races in today_racers.items():
        serializable[venue] = {}
        for rno, racers in races.items():
            serializable[venue][str(rno)] = [asdict(r) for r in racers]
    cache_file.write_text(json.dumps(serializable, ensure_ascii=False), encoding="utf-8")
    logger.info(f"キャッシュ保存: {cache_file}")


# ─────────────────────────────────────────────
# 朝の処理: 出走表取得 + 予想
# ─────────────────────────────────────────────
def morning_job():
    today = date.today()
    logger.info(f"=== 朝バッチ開始 {today} ===")

    sc = BoatraceScraper(delay=1.0)
    if not sc.login():
        logger.error("ログイン失敗。朝バッチを中断します")
        return

    # 開催会場を取得
    venues = get_holding_venues(sc, today)
    if not venues:
        logger.warning("本日の開催会場が見つかりませんでした")
        return

    logger.info(f"開催会場: {', '.join([VENUE_MAP[v] for v in venues])}")

    # 出走表を一括取得
    today_racers = {}
    for venue in venues:
        today_racers[venue] = {}
        for rno in range(1, 13):
            racers = sc.get_racelist(today, venue, rno)
            if racers:
                today_racers[venue][rno] = racers
                logger.info(f"取得: {VENUE_MAP[venue]} {rno}R {len(racers)}艇")

    # キャッシュ保存
    save_cache(today_racers, today)

    # 一括予想 + 記録
    predictor = BoatracePredictor()
    count = 0
    for venue, races in today_racers.items():
        for rno, racers in races.items():
            pred = predictor.predict(
                racers,
                race_date=today.strftime("%Y%m%d"),
                venue_name=VENUE_MAP[venue],
                race_no=rno,
            )
            record = {
                "race_date":   today.strftime("%Y%m%d"),
                "venue_code":  venue,
                "venue_name":  VENUE_MAP[venue],
                "race_no":     rno,
                "tansho":      pred.tansho,
                "sanren_tan":  pred.sanren_tan,
                "sanren_fuku": pred.sanren_fuku,
                "hit":         None,
                "actual":      "",
                "payout":      None,
            }
            save_record(record)
            count += 1

    total_races = sum(len(v) for v in today_racers.values())
    logger.info(f"✅ 朝バッチ完了: {len(venues)}会場 {total_races}レース取得 {count}レース予想・記録")


# ─────────────────────────────────────────────
# 夜の処理: 昨日の結果取得 + 的中判定
# ─────────────────────────────────────────────
def night_job():
    yesterday = date.today() - timedelta(days=1)
    logger.info(f"=== 夜バッチ開始 {yesterday} の結果取得 ===")

    sc = BoatraceScraper(delay=1.0)
    if not sc.login():
        logger.error("ログイン失敗。夜バッチを中断します")
        return

    records = load_records()
    date_str = yesterday.strftime("%Y%m%d")
    target_records = [r for r in records if r["race_date"] == date_str and r["hit"] is None]

    if not target_records:
        logger.info("昨日の未確認レースはありません")
        return

    logger.info(f"昨日の未確認レース: {len(target_records)}件")

    hit_count = 0
    miss_count = 0

    for r in target_records:
        result = sc.get_result(yesterday, r["venue_code"], r["race_no"])
        if not result or not result.arrival:
            continue

        if len(result.arrival) >= 3:
            actual = f"{result.arrival[0]}-{result.arrival[1]}-{result.arrival[2]}"
            r["actual"] = actual

            # 的中チェック
            hit = actual in r["sanren_tan"]
            r["hit"] = hit

            # 払戻金
            sanren_key = f"3連単_{actual}"
            r["payout"] = result.payouts.get(sanren_key)

            if hit:
                hit_count += 1
                logger.info(f"🎉 的中: {r['venue_name']} {r['race_no']}R {actual} ¥{r['payout']}")
            else:
                miss_count += 1
                logger.info(f"❌ ハズレ: {r['venue_name']} {r['race_no']}R 実際:{actual} 予想:{r['sanren_tan']}")

    save_records(records)

    total_checked = hit_count + miss_count
    hit_rate = hit_count / total_checked * 100 if total_checked > 0 else 0
    logger.info(f"✅ 夜バッチ完了: {total_checked}レース確認 的中{hit_count}件 ({hit_rate:.1f}%)")


# ─────────────────────────────────────────────
# スケジューラー起動
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="競艇 毎日自動バッチ")
    parser.add_argument("--run-morning", action="store_true", help="朝の処理を今すぐ実行")
    parser.add_argument("--run-night",   action="store_true", help="夜の処理を今すぐ実行")
    args = parser.parse_args()

    if args.run_morning:
        morning_job()
        return

    if args.run_night:
        night_job()
        return

    # スケジュール登録
    schedule.every().day.at("08:00").do(morning_job)
    schedule.every().day.at("23:00").do(night_job)

    logger.info("スケジューラー起動")
    logger.info("  08:00 → 出走表取得・一括予想・記録")
    logger.info("  23:00 → 昨日の結果取得・的中判定")
    logger.info("Ctrl+C で停止")

    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        logger.info("スケジューラー停止")


if __name__ == "__main__":
    main()