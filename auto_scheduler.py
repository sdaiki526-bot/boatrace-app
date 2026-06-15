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
import os
from supabase import create_client

import sys
sys.path.insert(0, str(Path(__file__).parent))

from boatrace_scraper import BoatraceScraper, VENUE_MAP, get_deadline_times
from predictor import BoatracePredictor, MLPredictor
from crawler import get_holding_venues

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None


def get_predictor():
    """LightGBMモデルがあればMLPredictor、無ければルールベースにフォールバック"""
    try:
        return MLPredictor(model_dir=Path(__file__).parent / "models")
    except FileNotFoundError as e:
        logger.warning(f"MLモデル未検出。ルールベース予想を使用します: {e}")
        return BoatracePredictor()


def _score_metrics(pred):
    """予想結果から1位スコアと1位-2位の差を計算する"""
    sorted_scores = sorted(pred.scores, key=lambda s: s.predicted_rank)
    if len(sorted_scores) < 2:
        return None, None
    top_score = sorted_scores[0].total_score
    score_gap = sorted_scores[0].total_score - sorted_scores[1].total_score
    return round(top_score, 3), round(score_gap, 3)

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
    # ローカルJSONにも保存（フォールバック）
    records = load_records()
    key = f"{record['race_date']}_{record['venue_code']}_{record['race_no']}"
    records = [r for r in records if f"{r['race_date']}_{r['venue_code']}_{r['race_no']}" != key]
    records.append(record)
    save_records(records)

    # Supabaseに保存
    if supabase:
        try:
            db_record = dict(record)
            db_record["sanren_tan"] = json.dumps(record["sanren_tan"], ensure_ascii=False)
            supabase.table("prediction_records").upsert(
                db_record, on_conflict="race_date,venue_code,race_no"
            ).execute()
        except Exception as e:
            logger.error(f"Supabase保存失敗: {e}")

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

    # 各会場の締切時刻を取得
    deadlines_by_venue = {}
    for venue in venues:
        try:
            deadlines_by_venue[venue] = get_deadline_times(sc, today, venue)
        except Exception as e:
            logger.error(f"締切時刻取得失敗 {VENUE_MAP[venue]}: {e}")
            deadlines_by_venue[venue] = {}

    # 出走表を一括取得
    today_racers = {}
    for venue in venues:
        today_racers[venue] = {}
        for rno in range(1, 13):
            racers = sc.get_racelist(today, venue, rno)
            if racers:
                today_racers[venue][rno] = racers
                logger.info(f"取得: {VENUE_MAP[venue]} {rno}R {len(racers)}艇")

    # キャッシュ保存（ローカル）
    save_cache(today_racers, today)

    # Supabaseにも保存
    if supabase:
        from dataclasses import asdict
        for venue, races in today_racers.items():
            for rno, racers in races.items():
                deadline_dt = deadlines_by_venue.get(venue, {}).get(rno)
                deadline_str = deadline_dt.strftime("%H:%M") if deadline_dt else None
                try:
                    supabase.table("today_racelist").upsert({
                        "race_date": today.strftime("%Y%m%d"),
                        "venue_code": venue,
                        "venue_name": VENUE_MAP[venue],
                        "race_no": rno,
                        "racers": json.dumps([asdict(r) for r in racers], ensure_ascii=False),
                        "deadline_time": deadline_str,
                    }, on_conflict="race_date,venue_code,race_no").execute()
                except Exception as e:
                    logger.error(f"出走表Supabase保存失敗: {e}")

    # 一括予想 + 記録
    predictor = get_predictor()
    count = 0
    for venue, races in today_racers.items():
        for rno, racers in races.items():
            pred = predictor.predict(
                racers,
                race_date=today.strftime("%Y%m%d"),
                venue_name=VENUE_MAP[venue],
                race_no=rno,
            )
            top_score, score_gap = _score_metrics(pred)
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
                "top_score":   top_score,
                "score_gap":   score_gap,
            }
            save_record(record)
            count += 1

    total_races = sum(len(v) for v in today_racers.values())
    logger.info(f"✅ 朝バッチ完了: {len(venues)}会場 {total_races}レース取得 {count}レース予想・記録")


# ─────────────────────────────────────────────
# 結果確認処理: 今日の確定済みレースを判定（12:00/15:00/18:00/23:00）
# ─────────────────────────────────────────────
def result_check_job():
    today = date.today()
    logger.info(f"=== 結果確認バッチ開始 {today} ===")

    sc = BoatraceScraper(delay=1.0)
    if not sc.login():
        logger.error("ログイン失敗。結果確認バッチを中断します")
        return

    _check_and_update_results(sc, today)


# ─────────────────────────────────────────────
# 夜の処理: 昨日の結果取得 + 的中判定（最終確認）
# ─────────────────────────────────────────────
def night_job():
    yesterday = date.today() - timedelta(days=1)
    logger.info(f"=== 夜バッチ開始 {yesterday} の結果取得 ===")

    sc = BoatraceScraper(delay=1.0)
    if not sc.login():
        logger.error("ログイン失敗。夜バッチを中断します")
        return

    _check_and_update_results(sc, yesterday)


def _check_and_update_results(sc, target_date):
    records = load_records()
    date_str = target_date.strftime("%Y%m%d")
    target_records = [r for r in records if r["race_date"] == date_str and r["hit"] is None]

    if not target_records:
        logger.info(f"{target_date} の未確認レースはありません")
        return

    logger.info(f"{target_date} の未確認レース: {len(target_records)}件")

    hit_count = 0
    miss_count = 0
    checked_records = []

    for r in target_records:
        result = sc.get_result(target_date, r["venue_code"], r["race_no"])
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

            checked_records.append(r)

    save_records(records)

    # Supabaseの該当レコードを更新
    if supabase:
        for r in checked_records:
            try:
                supabase.table("prediction_records").update({
                    "hit": r["hit"],
                    "actual": r["actual"],
                    "payout": r["payout"],
                }).eq("race_date", r["race_date"]).eq("venue_code", r["venue_code"]).eq("race_no", r["race_no"]).execute()
            except Exception as e:
                logger.error(f"Supabase更新失敗: {e}")

    total_checked = hit_count + miss_count
    hit_rate = hit_count / total_checked * 100 if total_checked > 0 else 0
    logger.info(f"✅ 結果確認完了({target_date}): {total_checked}レース確認 的中{hit_count}件 ({hit_rate:.1f}%)")


# ─────────────────────────────────────────────
# 展示タイム更新処理: 締切60分以内のレースを更新
# ─────────────────────────────────────────────
EXHIBITION_DONE_FILE = Path("exhibition_done.json")

def _load_exhibition_done():
    if EXHIBITION_DONE_FILE.exists():
        try:
            data = json.loads(EXHIBITION_DONE_FILE.read_text(encoding="utf-8"))
            today_str = date.today().strftime("%Y%m%d")
            # 今日以外のキーは捨てる（日付が変わったらリセット）
            return set(k for k in data if k.startswith(today_str))
        except Exception:
            return set()
    return set()

def _save_exhibition_done(done_set):
    EXHIBITION_DONE_FILE.write_text(json.dumps(list(done_set), ensure_ascii=False), encoding="utf-8")


def exhibition_job():
    """
    10分おきに実行。締切時刻が60分以内のレースについて
    展示タイムを取得し、予想を再計算してSupabaseを更新する。
    """
    from before_info_scraper import BeforeInfoScraper
    from datetime import datetime as _dt

    today = date.today()
    date_str = today.strftime("%Y%m%d")
    done = _load_exhibition_done()

    sc = BoatraceScraper(delay=1.0)
    if not sc.login():
        logger.error("ログイン失敗。展示タイム更新を中断します")
        return

    venues = get_holding_venues(sc, today)
    if not venues:
        return

    now = _dt.now()
    before_scraper = BeforeInfoScraper(delay=1.0)
    predictor = get_predictor()
    updated_count = 0

    for venue in venues:
        try:
            deadlines = get_deadline_times(sc, today, venue)
        except Exception as e:
            logger.error(f"締切時刻取得失敗 {VENUE_MAP.get(venue, venue)}: {e}")
            continue

        for rno, deadline in deadlines.items():
            key = f"{date_str}_{venue}_{rno}"
            if key in done:
                continue

            minutes_to_deadline = (deadline - now).total_seconds() / 60
            # 締切60分前〜締切後10分の間に処理対象とする
            if not (-10 <= minutes_to_deadline <= 60):
                continue

            # 展示タイム取得
            info = before_scraper.get_before_info(today, venue, rno)
            if not info or not info.exhibitions:
                continue

            exhibition_times = {
                e.lane: e.exhibition_time for e in info.exhibitions
                if e.exhibition_time is not None
            }
            if not exhibition_times:
                continue

            # 出走表（既存予想の元データ）を再取得して予想更新
            racers = sc.get_racelist(today, venue, rno)
            if not racers:
                continue

            pred = predictor.predict(
                racers,
                race_date=date_str,
                venue_name=VENUE_MAP[venue],
                race_no=rno,
                exhibition_times=exhibition_times,
            )

            top_score, score_gap = _score_metrics(pred)

            # オッズ取得（最有力3連単候補のオッズ）
            odds_value = None
            try:
                odds = sc.get_odds(today, venue, rno)
                if odds and odds.sanren_tan and pred.sanren_tan:
                    top_combo = pred.sanren_tan[0]
                    odds_value = odds.sanren_tan.get(top_combo)
            except Exception as e:
                logger.error(f"オッズ取得失敗 {VENUE_MAP[venue]} {rno}R: {e}")

            save_record({
                "race_date":   date_str,
                "venue_code":  venue,
                "venue_name":  VENUE_MAP[venue],
                "race_no":     rno,
                "tansho":      pred.tansho,
                "sanren_tan":  pred.sanren_tan,
                "sanren_fuku": pred.sanren_fuku,
                "hit":         None,
                "actual":      "",
                "payout":      None,
                "top_score":   top_score,
                "score_gap":   score_gap,
                "odds_value":  odds_value,
            })

            done.add(key)
            updated_count += 1
            logger.info(f"展示タイム反映: {VENUE_MAP[venue]} {rno}R → {pred.sanren_tan}")

    _save_exhibition_done(done)
    if updated_count:
        logger.info(f"✅ 展示タイム更新完了: {updated_count}レース")


# ─────────────────────────────────────────────
# 週次再学習: データセット再構築 + モデル再学習
# ─────────────────────────────────────────────
def retrain_job():
    import subprocess
    import os as _os

    base_dir = Path(__file__).parent
    logger.info("=== 週次再学習バッチ開始 ===")

    env = _os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        logger.info("build_dataset.py 実行中...")
        result = subprocess.run(
            ["python", str(base_dir / "build_dataset.py")],
            cwd=str(base_dir), capture_output=True, text=True,
            encoding="utf-8", errors="replace", env=env, timeout=3600,
        )
        if result.returncode != 0:
            logger.error(f"build_dataset.py 失敗: {result.stderr[-2000:]}")
            return
        logger.info("build_dataset.py 完了")
    except Exception as e:
        logger.error(f"build_dataset.py 実行エラー: {e}")
        return

    try:
        logger.info("train_model.py 実行中...")
        result = subprocess.run(
            ["python", str(base_dir / "train_model.py")],
            cwd=str(base_dir), capture_output=True, text=True,
            encoding="utf-8", errors="replace", env=env, timeout=3600,
        )
        if result.returncode != 0:
            logger.error(f"train_model.py 失敗: {result.stderr[-2000:]}")
            return
        logger.info("train_model.py 完了")
        # train_model.pyの出力からAUC等を抜粋してログに残す
        auc_lines = []
        for line in result.stdout.splitlines():
            if "AUC" in line or "保存完了" in line:
                logger.info(f"  {line.strip()}")
                if "AUC" in line:
                    auc_lines.append(line.strip())
    except Exception as e:
        logger.error(f"train_model.py 実行エラー: {e}")
        return

    # モデルをGitへ自動コミット・プッシュ
    try:
        commit_msg = "weekly model retrain"
        if auc_lines:
            commit_msg += " (" + ", ".join(auc_lines) + ")"

        subprocess.run(["git", "add", "models/"], cwd=str(base_dir),
                        capture_output=True, text=True, encoding="utf-8", errors="replace")

        diff_check = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=str(base_dir))
        if diff_check.returncode == 0:
            logger.info("モデルに変更なし。Gitコミットはスキップします")
        else:
            commit_result = subprocess.run(
                ["git", "commit", "-m", commit_msg], cwd=str(base_dir),
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            if commit_result.returncode != 0:
                logger.error(f"git commit 失敗: {commit_result.stderr.strip()}")
            else:
                push_result = subprocess.run(
                    ["git", "push"], cwd=str(base_dir),
                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
                )
                if push_result.returncode != 0:
                    logger.error(f"git push 失敗: {push_result.stderr.strip()}")
                else:
                    logger.info("✅ モデルをGitにコミット・プッシュしました")
    except Exception as e:
        logger.error(f"Git自動コミット失敗: {e}")

    logger.info("✅ 週次再学習バッチ完了")


# ─────────────────────────────────────────────
# スケジューラー起動
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="競艇 毎日自動バッチ")
    parser.add_argument("--run-morning", action="store_true", help="朝の処理を今すぐ実行")
    parser.add_argument("--run-night",   action="store_true", help="夜の処理（昨日分の最終確認）を今すぐ実行")
    parser.add_argument("--run-result-check", action="store_true", help="結果確認（今日分）を今すぐ実行")
    parser.add_argument("--run-exhibition", action="store_true", help="展示タイム更新を今すぐ実行")
    parser.add_argument("--run-retrain", action="store_true", help="モデル再学習を今すぐ実行")
    args = parser.parse_args()

    if args.run_morning:
        morning_job()
        return

    if args.run_night:
        night_job()
        return

    if args.run_result_check:
        result_check_job()
        return

    if args.run_exhibition:
        exhibition_job()
        return

    if args.run_retrain:
        retrain_job()
        return

    # スケジュール登録
    schedule.every().day.at("08:00").do(morning_job)
    schedule.every().day.at("12:00").do(result_check_job)
    schedule.every().day.at("15:00").do(result_check_job)
    schedule.every().day.at("18:00").do(result_check_job)
    schedule.every().day.at("23:00").do(result_check_job)
    schedule.every().day.at("23:30").do(night_job)
    schedule.every(10).minutes.do(exhibition_job)
    schedule.every().monday.at("05:00").do(retrain_job)

    logger.info("スケジューラー起動")
    logger.info("  08:00 → 出走表取得・一括予想・記録")
    logger.info("  10分おき → 締切60分以内のレースの展示タイムを反映")
    logger.info("  12:00/15:00/18:00/23:00 → 今日の確定済みレースの的中判定")
    logger.info("  23:30 → 昨日分の最終確認")
    logger.info("  月曜 05:00 → データセット再構築・モデル週次再学習・Git自動push")
    logger.info("Ctrl+C で停止")

    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        logger.info("スケジューラー停止")


if __name__ == "__main__":
    main()