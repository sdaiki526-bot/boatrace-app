"""
競艇 自動スケジューラー

毎日指定時刻に：
  1. 指定会場の全レース出走表・オッズを取得
  2. 予想ロジックを実行
  3. 結果をJSONとテキストで保存
  4. （オプション）Slack / LINE Notify / メール通知

設定ファイル: scheduler_config.json

実行方法:
  python scheduler.py                # フォアグラウンドで常時起動
  python scheduler.py --run-now      # 今すぐ1回実行してテスト
  python scheduler.py --config my_config.json  # 設定ファイル指定
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import date, datetime
from pathlib import Path

import schedule
import requests as http_requests

# ── 自モジュール
sys.path.insert(0, str(Path(__file__).parent))
from boatrace_scraper import BoatraceScraper, VENUE_MAP
from predictor import BoatracePredictor, PredictorConfig

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
        logging.FileHandler(LOG_DIR / "scheduler.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# デフォルト設定
# ─────────────────────────────────────────────
DEFAULT_CONFIG = {
    "schedules": [
        {
            "time": "08:00",          # 毎日この時刻に実行
            "venues": ["01", "12"],   # 取得する会場コード（複数可）
            "max_race": 12,           # 取得レース数
        }
    ],
    "output_dir": "output",           # 保存ディレクトリ
    "request_delay": 1.5,            # リクエスト間隔（秒）
    "notify": {
        "slack_webhook_url": "",      # Slack Incoming Webhook URL（空なら無効）
        "line_token": "",             # LINE Notify トークン（空なら無効）
    },
    "predictor": {
        "w_course": 35.0,
        "w_win_rate": 20.0,
        "w_local_rate": 15.0,
        "w_rank": 10.0,
        "w_motor": 10.0,
        "w_boat": 5.0,
        "w_start": 5.0,
        "penalty_per_flying": 3.0,
        "penalty_per_late": 1.5,
    }
}

CONFIG_PATH = Path("scheduler_config.json")


# ─────────────────────────────────────────────
# 設定ファイル管理
# ─────────────────────────────────────────────
def load_config(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        logger.info(f"設定ファイル読み込み: {path}")
        return cfg
    else:
        logger.info(f"設定ファイルが見つかりません。デフォルト設定で {path} を作成します")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=2)
        return DEFAULT_CONFIG


# ─────────────────────────────────────────────
# 通知
# ─────────────────────────────────────────────
def notify_slack(webhook_url: str, text: str):
    if not webhook_url:
        return
    try:
        http_requests.post(webhook_url, json={"text": text}, timeout=10)
        logger.info("Slack通知送信完了")
    except Exception as e:
        logger.error(f"Slack通知失敗: {e}")


def notify_line(token: str, message: str):
    if not token:
        return
    try:
        http_requests.post(
            "https://notify-api.line.me/api/notify",
            headers={"Authorization": f"Bearer {token}"},
            data={"message": message},
            timeout=10,
        )
        logger.info("LINE Notify送信完了")
    except Exception as e:
        logger.error(f"LINE通知失敗: {e}")


# ─────────────────────────────────────────────
# メインジョブ
# ─────────────────────────────────────────────
def run_job(schedule_cfg: dict, global_cfg: dict):
    today = date.today()
    date_str = today.strftime("%Y%m%d")
    venues = schedule_cfg.get("venues", ["01"])
    max_race = schedule_cfg.get("max_race", 12)

    output_dir = Path(global_cfg.get("output_dir", "output")) / date_str
    output_dir.mkdir(parents=True, exist_ok=True)

    delay = global_cfg.get("request_delay", 1.5)
    notify_cfg = global_cfg.get("notify", {})

    predictor_cfg = PredictorConfig(**{
        k: v for k, v in global_cfg.get("predictor", {}).items()
        if hasattr(PredictorConfig, k)
    })
    predictor = BoatracePredictor(predictor_cfg)
    scraper = BoatraceScraper(delay=delay)

    all_summaries = []

    for venue_code in venues:
        vc = venue_code.zfill(2)
        vname = VENUE_MAP.get(vc, vc)
        logger.info(f"▶ {vname}（{vc}） データ取得開始")

        try:
            races = scraper.get_all_races(today, vc, max_race=max_race)
        except Exception as e:
            logger.error(f"{vname} 取得エラー: {e}")
            continue

        # 生データ保存
        raw_path = output_dir / f"{vname}_{date_str}_raw.json"
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(races, f, ensure_ascii=False, indent=2)
        logger.info(f"生データ保存: {raw_path}")

        # 予想
        predictions = predictor.predict_all_races(races)

        # テキストサマリー生成
        txt_lines = [f"【{vname} {date_str} 全レース予想】\n"]
        for pred in predictions:
            txt_lines.append(pred.summary())
            txt_lines.append("")

        summary_text = "\n".join(txt_lines)
        all_summaries.append(summary_text)

        # テキスト保存
        txt_path = output_dir / f"{vname}_{date_str}_predictions.txt"
        txt_path.write_text(summary_text, encoding="utf-8")
        logger.info(f"予想テキスト保存: {txt_path}")

        # JSON保存
        json_path = output_dir / f"{vname}_{date_str}_predictions.json"
        BoatracePredictor.save_predictions(predictions, str(json_path))

    # 通知
    if all_summaries:
        notify_text = f"🚤 競艇予想 {date_str}\n" + "\n\n".join(
            s[:500] for s in all_summaries  # 通知は先頭500文字
        )
        notify_slack(notify_cfg.get("slack_webhook_url", ""), notify_text)
        notify_line(notify_cfg.get("line_token", ""), notify_text)

    logger.info(f"✅ ジョブ完了: {date_str} 処理会場={venues}")


# ─────────────────────────────────────────────
# スケジューラー起動
# ─────────────────────────────────────────────
def setup_schedules(config: dict):
    for sched in config.get("schedules", []):
        run_time = sched.get("time", "08:00")
        schedule.every().day.at(run_time).do(run_job, sched, config)
        logger.info(f"スケジュール登録: 毎日 {run_time} に実行 (会場: {sched.get('venues')})")


def run_scheduler(config: dict):
    setup_schedules(config)
    logger.info("スケジューラー起動。Ctrl+C で停止。")
    try:
        while True:
            schedule.run_pending()
            time.sleep(30)  # 30秒ごとにチェック
    except KeyboardInterrupt:
        logger.info("スケジューラー停止")


# ─────────────────────────────────────────────
# エントリーポイント
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="競艇 自動スケジューラー")
    parser.add_argument(
        "--config", default=str(CONFIG_PATH),
        help="設定ファイルパス（デフォルト: scheduler_config.json）"
    )
    parser.add_argument(
        "--run-now", action="store_true",
        help="スケジューラーを起動せず、今すぐ1回だけ実行する（テスト用）"
    )
    parser.add_argument(
        "--venue", nargs="+",
        help="--run-now 時に取得する会場コード（例: 01 12）"
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))

    if args.run_now:
        # テスト実行
        sched_cfg = config["schedules"][0].copy()
        if args.venue:
            sched_cfg["venues"] = args.venue
        logger.info("▶ テスト実行開始")
        run_job(sched_cfg, config)
    else:
        run_scheduler(config)


if __name__ == "__main__":
    main()
