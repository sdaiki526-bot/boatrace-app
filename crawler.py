"""
競艇 過去データ一括取得クローラー（高速版）

改善点:
  - 開催スケジュールを先に確認し、開催のない会場はスキップ
  - 中断・再開対応
  - エラー自動リトライ

使い方:
  python crawler.py --start 20250522 --end 20260522
  python crawler.py --start 20250522 --end 20260522 --venues 01 12
"""

import argparse
import json
import logging
import time
from datetime import date, timedelta, datetime
from pathlib import Path

from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from boatrace_scraper import BoatraceScraper, VENUE_MAP

# ─────────────────────────────────────────────
# ロギング
# ─────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.WARNING,   # コンソールはWARNING以上のみ表示
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "crawler.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

PROGRESS_FILE = Path("crawler_progress.json")
DEFAULT_OUTPUT_DIR = Path("data")
MAX_RETRY = 3
RETRY_WAIT = 5


# ─────────────────────────────────────────────
# 進捗管理
# ─────────────────────────────────────────────
class ProgressTracker:
    def __init__(self, path: Path = PROGRESS_FILE):
        self.path = path
        self.done: set[str] = set()
        self.errors: dict[str, int] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.done = set(data.get("done", []))
            self.errors = data.get("errors", {})
            print(f"  📂 進捗ファイル読み込み: 取得済み {len(self.done):,} 件")

    def save(self):
        self.path.write_text(
            json.dumps({"done": sorted(self.done), "errors": self.errors},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def is_done(self, key: str) -> bool:
        return key in self.done

    def mark_done(self, key: str):
        self.done.add(key)
        self.errors.pop(key, None)

    def mark_error(self, key: str):
        self.errors[key] = self.errors.get(key, 0) + 1


# ─────────────────────────────────────────────
# 開催スケジュール取得
# ─────────────────────────────────────────────
def get_holding_venues(scraper: BoatraceScraper, d: date) -> list[str]:
    """
    指定日に開催している会場コードのリストを返す。
    公式の「本日のレース」ページから開催会場を取得する。
    """
    from bs4 import BeautifulSoup
    date_str = d.strftime("%Y%m%d")
    url = f"https://www.boatrace.jp/owpc/pc/race/index?hd={date_str}"
    try:
        resp = scraper.session.get(url, timeout=15)
        time.sleep(scraper.delay)
        soup = BeautifulSoup(resp.text, "html.parser")

        venues = []
        # 開催会場リンクからjcd（会場コード）を抽出
        for a in soup.select("a[href*='jcd=']"):
            href = a.get("href", "")
            for part in href.split("&"):
                if part.startswith("jcd="):
                    jcd = part.split("=")[1].zfill(2)
                    if jcd not in venues and jcd in VENUE_MAP:
                        venues.append(jcd)

        # 重複除去して返す
        return sorted(set(venues))
    except Exception as e:
        logger.warning(f"開催スケジュール取得失敗 {date_str}: {e}")
        return []


# ─────────────────────────────────────────────
# クローラー本体
# ─────────────────────────────────────────────
class BulkCrawler:
    def __init__(self, output_dir: Path = DEFAULT_OUTPUT_DIR, delay: float = 1.5, overwrite: bool = False):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.overwrite = overwrite
        self.scraper = BoatraceScraper(delay=delay)
        self.progress = ProgressTracker()

    def _key(self, date_str, venue, race_no):
        return f"{date_str}_{venue}_{race_no:02d}"

    def _output_path(self, date_str, venue):
        month_dir = self.output_dir / date_str[:6]
        month_dir.mkdir(parents=True, exist_ok=True)
        return month_dir / f"{date_str}_{VENUE_MAP.get(venue, venue)}.json"

    def _load_existing(self, path):
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _fetch_race(self, d, venue, race_no):
        from dataclasses import asdict
        from concurrent.futures import ThreadPoolExecutor, as_completed

        for attempt in range(1, MAX_RETRY + 1):
            try:
                # 出走表と結果を並列取得
                with ThreadPoolExecutor(max_workers=2) as ex:
                    f_racers = ex.submit(self.scraper.get_racelist, d, venue, race_no)
                    f_result = ex.submit(self.scraper.get_result,   d, venue, race_no)
                    racers = f_racers.result(timeout=20)
                    result = f_result.result(timeout=20)
                return {
                    "race_no": race_no,
                    "racers": [asdict(r) for r in racers],
                    "result": asdict(result) if result else None,
                }
            except Exception as e:
                logger.warning(f"取得失敗 ({attempt}/{MAX_RETRY}): {e}")
                if attempt < MAX_RETRY:
                    time.sleep(RETRY_WAIT * attempt)
        return None

    def _save_group(self, date_str, venue, races):
        path = self._output_path(date_str, venue)
        existing = self._load_existing(path)
        existing.update({str(k): v for k, v in races.items()})
        path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")

    def crawl(self, start: date, end: date, venues: list[str], max_race: int = 12):
        # ログイン
        self.scraper.login()

        dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]

        print(f"\n{'='*55}")
        print(f"  競艇データ 一括取得クローラー（高速版）")
        print(f"{'='*55}")
        print(f"  期間  : {start} 〜 {end}  ({len(dates)}日間)")
        print(f"  会場  : 各日の開催会場を自動検出")
        print(f"  取得済: {len(self.progress.done):,} 件（スキップ）")
        print(f"{'='*55}\n")

        total_fetched = 0
        total_skipped = 0

        for d in tqdm(dates, desc="日付", ncols=70, unit="日"):
            date_str = d.strftime("%Y%m%d")

            # 開催会場を自動検出（指定会場でフィルタ）
            holding = get_holding_venues(self.scraper, d)
            if venues:
                holding = [v for v in holding if v in venues]

            if not holding:
                continue

            for venue in holding:
                vname = VENUE_MAP.get(venue, venue)
                group_data = {}

                for race_no in range(1, max_race + 1):
                    key = self._key(date_str, venue, race_no)

                    if not self.overwrite and self.progress.is_done(key):
                        total_skipped += 1
                        continue

                    race_data = self._fetch_race(d, venue, race_no)
                    if race_data:
                        group_data[race_no] = race_data
                        self.progress.mark_done(key)
                        total_fetched += 1
                    else:
                        self.progress.mark_error(key)

                if group_data:
                    self._save_group(date_str, venue, group_data)

            # 1日ごとに進捗保存
            self.progress.save()

        self.progress.save()
        print(f"\n{'='*55}")
        print(f"  ✅ 完了!")
        print(f"  新規取得: {total_fetched:,} 件")
        print(f"  スキップ: {total_skipped:,} 件（取得済み）")
        print(f"  エラー  : {len(self.progress.errors)} 件")
        print(f"  保存先  : {self.output_dir.resolve()}")
        print(f"{'='*55}\n")


# ─────────────────────────────────────────────
# エントリーポイント
# ─────────────────────────────────────────────
def parse_date(s):
    return datetime.strptime(s, "%Y%m%d").date()

def main():
    parser = argparse.ArgumentParser(description="競艇 過去データ一括取得クローラー（高速版）")
    parser.add_argument("--start", required=True, help="取得開始日 YYYYMMDD")
    parser.add_argument("--end",   required=True, help="取得終了日 YYYYMMDD")
    parser.add_argument("--venues", nargs="+", default=[], help="会場コード（省略時=開催全会場）")
    parser.add_argument("--max-race", type=int, default=12)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    crawler = BulkCrawler(
        output_dir=Path(args.output_dir),
        delay=args.delay,
        overwrite=args.overwrite,
    )
    crawler.crawl(
        start=parse_date(args.start),
        end=parse_date(args.end),
        venues=[v.zfill(2) for v in args.venues],
        max_race=args.max_race,
    )

if __name__ == "__main__":
    main()