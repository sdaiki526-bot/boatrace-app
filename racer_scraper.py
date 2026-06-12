"""
選手成績スクレイパー

boatrace.jp のレーサー検索ページから選手情報・成績を取得する。
ログイン不要。

取得データ:
  - 基本情報（名前・登録番号・体重・支部・級別など）
  - 期別成績（勝率・2連対率・3連対率・平均ST・F数・L数）
  - コース別成績

使い方:
  # 単体テスト
  python racer_scraper.py --toban 4096

  # data/フォルダから選手番号を自動抽出して一括取得
  python racer_scraper.py --from-data

  # 出力先指定
  python racer_scraper.py --from-data --out dataset/racer_stats.json
"""

import argparse
import json
import logging
import time
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://www.boatrace.jp/owpc/pc/data/racersearch"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


@dataclass
class RacerStats:
    toban: str
    name: str = ""
    kana: str = ""
    branch: str = ""
    birth_date: str = ""
    height: Optional[int] = None
    weight: Optional[int] = None
    rank: str = ""
    period: str = ""
    win_rate: Optional[float] = None
    rate_2: Optional[float] = None
    rate_3: Optional[float] = None
    avg_st: Optional[float] = None
    flying: int = 0
    late: int = 0
    race_count: Optional[int] = None
    course_win: dict = field(default_factory=dict)


class RacerScraper:
    def __init__(self, delay: float = 1.0):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def _get(self, url):
        try:
            resp = self.session.get(url, timeout=12)
            resp.raise_for_status()
            time.sleep(self.delay)
            return BeautifulSoup(resp.text, "html.parser")
        except Exception as e:
            logger.error(f"取得失敗 {url}: {e}")
            return None

    @staticmethod
    def _sf(text):
        try:
            return float(str(text).strip().replace("%","").replace("回","").replace(",",""))
        except:
            return None

    @staticmethod
    def _si(text):
        try:
            return int(str(text).strip().replace("cm","").replace("kg","").replace("回","").replace("期",""))
        except:
            return None

    def get_racer(self, toban: str) -> Optional[RacerStats]:
        toban = str(toban).zfill(4)
        stats = RacerStats(toban=toban)

        # 期別成績ページ（基本情報+成績）
        soup = self._get(f"{BASE_URL}/season?toban={toban}")
        if soup is None or "データがありません" in soup.text:
            logger.warning(f"選手データなし: {toban}")
            return None

        # 基本情報
        dl = soup.select_one("dl.list3")
        if dl:
            keys = [dt.text.strip() for dt in dl.select("dt")]
            vals = [dd.text.strip() for dd in dl.select("dd")]
            items = dict(zip(keys, vals))
            name_el = soup.select_one("p.racer1_bodyName")
            kana_el = soup.select_one("p.racer1_bodyKana")
            stats.name       = name_el.text.strip() if name_el else ""
            stats.kana       = kana_el.text.strip() if kana_el else ""
            stats.branch     = items.get("支部", "")
            stats.birth_date = items.get("生年月日", "")
            stats.rank       = items.get("級別", "").replace("級", "")
            stats.period     = items.get("登録期", "")
            stats.height     = self._si(items.get("身長", ""))
            stats.weight     = self._si(items.get("体重", ""))

        # 期別成績テーブル
        tbl = soup.select_one("table.is-w832")
        if tbl:
            label_map = {}
            for row in tbl.select("tr"):
                cells = row.select("th, td")
                for i in range(0, len(cells) - 1, 2):
                    label_map[cells[i].text.strip()] = cells[i+1].text.strip() if i+1 < len(cells) else ""
            stats.win_rate   = self._sf(label_map.get("勝率", ""))
            stats.rate_2     = self._sf(label_map.get("2連対率", ""))
            stats.rate_3     = self._sf(label_map.get("3連対率", ""))
            stats.avg_st     = self._sf(label_map.get("平均スタートタイミング", ""))
            stats.flying     = self._si(label_map.get("フライング回数", "0")) or 0
            stats.late       = self._si(label_map.get("出遅れ回数（選手責任）", "0")) or 0
            stats.race_count = self._si(label_map.get("出走回数", ""))

        # コース別成績
        soup2 = self._get(f"{BASE_URL}/course?toban={toban}")
        if soup2 and "データがありません" not in soup2.text:
            for tbl2 in soup2.select("table"):
                for row in tbl2.select("tr"):
                    cells = row.select("td")
                    if len(cells) >= 2 and cells[0].text.strip().isdigit():
                        stats.course_win[cells[0].text.strip()] = self._sf(cells[1].text)

        logger.info(f"取得完了: {toban} {stats.name} {stats.rank} 勝率={stats.win_rate}")
        return stats

    def get_racers_bulk(self, tobans: list, out_path: str = None) -> dict:
        results = {}
        for i, toban in enumerate(tobans):
            logger.info(f"[{i+1}/{len(tobans)}] {toban}")
            stats = self.get_racer(toban)
            if stats:
                results[toban] = asdict(stats)
            # 100件ごとに中間保存
            if out_path and (i+1) % 100 == 0:
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                Path(out_path).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
                logger.info(f"中間保存: {i+1}件")

        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\n✅ 保存完了: {out_path} ({len(results)}件)")
        return results


def extract_tobans_from_data(data_dir: str = "data") -> list:
    import glob
    tobans = set()
    for jf in glob.glob(f"{data_dir}/**/*.json", recursive=True):
        try:
            data = json.loads(Path(jf).read_text(encoding="utf-8"))
            for race in data.values():
                for racer in race.get("racers", []):
                    no = str(racer.get("racer_no", "")).strip()
                    if no and no.isdigit():
                        tobans.add(no.zfill(4))
        except:
            pass
    return sorted(tobans)


def main():
    parser = argparse.ArgumentParser(description="選手成績スクレイパー")
    parser.add_argument("--toban", help="選手番号（単体テスト用）")
    parser.add_argument("--toban-list", help="選手番号リストファイル（1行1番号）")
    parser.add_argument("--from-data", action="store_true", help="data/フォルダから自動抽出")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="dataset/racer_stats.json")
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()

    scraper = RacerScraper(delay=args.delay)

    if args.toban:
        stats = scraper.get_racer(args.toban)
        if stats:
            print(json.dumps(asdict(stats), ensure_ascii=False, indent=2))
    elif args.toban_list:
        tobans = Path(args.toban_list).read_text(encoding="utf-8").strip().splitlines()
        scraper.get_racers_bulk(tobans, args.out)
    elif args.from_data:
        print(f"data/フォルダから選手番号を抽出中...")
        tobans = extract_tobans_from_data(args.data_dir)
        print(f"  抽出選手数: {len(tobans)}名")
        scraper.get_racers_bulk(tobans, args.out)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()