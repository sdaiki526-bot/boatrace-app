"""
公式予想（コンピューター予想）スクレイパー

boatrace.jp の「直前情報」ではなく「コンピューター予想」ページから、
各艇に付与された印（◎○▲△）を取得する。ログイン不要。

使い方:
  python official_prediction_scraper.py --venue 04 --race 1
  python official_prediction_scraper.py --venue 04 --race 1 --date 20260729
"""

import argparse
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://www.boatrace.jp/owpc/pc/race/pcexpect"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# icon_mark1_{n}.png の n → 印の対応（1位予想=◎ ... 4位予想=△）
MARK_MAP = {1: "◎", 2: "○", 3: "▲", 4: "△"}


@dataclass
class OfficialPrediction:
    race_date: str
    venue_code: str
    race_no: int
    marks: dict = field(default_factory=dict)  # {lane(int): "◎"/"○"/"▲"/"△"}

    @property
    def honmei(self) -> Optional[int]:
        """本命（◎）の艇番。無ければNone"""
        for lane, mark in self.marks.items():
            if mark == "◎":
                return lane
        return None


class OfficialPredictionScraper:
    def __init__(self, delay: float = 1.0):
        self.delay = delay
        self.session = requests.Session()

    def get_official_prediction(
        self, race_date: date, venue_code: str, race_no: int
    ) -> Optional[OfficialPrediction]:
        vc = venue_code.zfill(2)
        params = {"rno": race_no, "jcd": vc, "hd": race_date.strftime("%Y%m%d")}
        try:
            resp = self.session.get(BASE_URL, params=params, headers=HEADERS, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            logger.error(f"取得失敗: {e}")
            return None

        if "データがありません" in resp.text:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        marks: dict = {}
        for img in soup.select("img[src*='icon_mark1_']"):
            src = img.get("src", "")
            try:
                n = int(src.rsplit("_", 1)[-1].split(".")[0])
            except (ValueError, IndexError):
                continue
            mark = MARK_MAP.get(n)
            if not mark:
                continue
            td = img.find_parent("td")
            if not td:
                continue
            # 印セルのすぐ後ろの<td>に艇番が入っている（例: class="is-boatColor3">3）
            lane_td = td.find_next_sibling("td")
            if not lane_td:
                continue
            lane_text = lane_td.get_text(strip=True)
            try:
                lane = int(lane_text)
            except ValueError:
                continue
            marks[lane] = mark

        if not marks:
            return None

        return OfficialPrediction(
            race_date=race_date.strftime("%Y%m%d"),
            venue_code=vc,
            race_no=race_no,
            marks=marks,
        )


def main():
    parser = argparse.ArgumentParser(description="公式コンピューター予想の印を取得")
    parser.add_argument("--venue", required=True, help="会場コード 例: 04")
    parser.add_argument("--race", type=int, required=True, help="レース番号 1〜12")
    parser.add_argument("--date", help="日付 YYYYMMDD（省略時=今日）")
    args = parser.parse_args()

    if args.date:
        from datetime import datetime
        race_date = datetime.strptime(args.date, "%Y%m%d").date()
    else:
        race_date = date.today()

    scraper = OfficialPredictionScraper()
    pred = scraper.get_official_prediction(race_date, args.venue, args.race)
    if pred is None:
        print("取得できませんでした")
        return
    print(f"{pred.race_date} 会場{pred.venue_code} {pred.race_no}R")
    for lane in range(1, 7):
        print(f"  {lane}号艇: {pred.marks.get(lane, '－')}")
    print(f"本命(◎): {pred.honmei}号艇")


if __name__ == "__main__":
    main()
