"""
直前情報スクレイパー

レース直前に公開される以下のデータを取得する（ログイン不要）:
  - 展示タイム・チルト・部品交換
  - スタート展示（コース・並び・ST）
  - 気象情報（気温・風速・風向・水温・波高）

※ 展示タイムはレース約1時間前から公開される。
  公開前はNullで返る。

使い方:
  python before_info_scraper.py --venue 01 --race 1
  python before_info_scraper.py --venue 01 --race 1 --date 20260605
"""

import argparse
import json
import logging
import re
import time
from dataclasses import dataclass, asdict, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://www.boatrace.jp/owpc/pc/race/beforeinfo"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


# ─────────────────────────────────────────────
# データクラス
# ─────────────────────────────────────────────
@dataclass
class ExhibitionInfo:
    """1艇分の直前情報"""
    lane: int
    name: str = ""
    racer_no: str = ""
    weight: Optional[float] = None          # 体重
    adj_weight: Optional[float] = None      # 調整重量
    exhibition_time: Optional[float] = None # 展示タイム
    tilt: Optional[float] = None            # チルト
    propeller: str = ""                     # プロペラ（新=交換）
    parts_changed: list = field(default_factory=list)  # 部品交換

@dataclass
class StartExhibition:
    """スタート展示1艇分"""
    course: int
    boat_no: int
    st: Optional[float] = None

@dataclass
class WeatherInfo:
    """気象情報"""
    temperature: Optional[float] = None    # 気温
    weather: str = ""                       # 天候
    wind_speed: Optional[float] = None     # 風速
    wind_direction: str = ""               # 風向
    water_temp: Optional[float] = None     # 水温
    wave_height: Optional[float] = None    # 波高

@dataclass
class BeforeInfo:
    """直前情報まとめ"""
    race_date: str
    venue_code: str
    race_no: int
    exhibitions: list = field(default_factory=list)    # 展示タイム（6艇）
    start_exhibition: list = field(default_factory=list)  # スタート展示
    weather: Optional[WeatherInfo] = None


# ─────────────────────────────────────────────
# スクレイパー
# ─────────────────────────────────────────────
class BeforeInfoScraper:
    def __init__(self, delay: float = 1.0):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def _get(self, url: str, params: dict = None) -> Optional[BeautifulSoup]:
        try:
            resp = self.session.get(url, params=params, timeout=12)
            resp.raise_for_status()
            time.sleep(self.delay)
            return BeautifulSoup(resp.text, "html.parser")
        except Exception as e:
            logger.error(f"取得失敗: {e}")
            return None

    @staticmethod
    def _sf(text: str) -> Optional[float]:
        try:
            return float(str(text).strip().replace("m", "").replace("℃", "")
                         .replace("°C", "").replace("cm", "").replace("kg", ""))
        except:
            return None

    def get_before_info(self, race_date: date, venue_code: str, race_no: int) -> Optional[BeforeInfo]:
        vc = venue_code.zfill(2)
        params = {
            "hd":  race_date.strftime("%Y%m%d"),
            "jcd": vc,
            "rno": race_no,
        }
        soup = self._get(BASE_URL, params)
        if soup is None or "データがありません" in soup.text:
            return None

        info = BeforeInfo(
            race_date=race_date.strftime("%Y%m%d"),
            venue_code=vc,
            race_no=race_no,
        )

        # ── 展示タイム（table.is-w748）──────────
        main_table = soup.select_one("table.is-w748")
        if main_table:
            for tbody in main_table.select("tbody"):
                rows = tbody.select("tr")
                if not rows:
                    continue

                # 枠番
                lane_el = tbody.select_one("td.is-fs14")
                if not lane_el:
                    continue
                lane_text = lane_el.text.strip()
                try:
                    lane = int(lane_el.text.strip())
                except:
                    continue

                # 選手名・登録番号
                name_el = tbody.select_one("td.is-fs18 a")
                name = name_el.text.strip() if name_el else ""
                racer_no = ""
                if name_el:
                    href = name_el.get("href", "")
                    m = re.search(r'toban=(\d+)', href)
                    if m:
                        racer_no = m.group(1)

                # 体重・調整重量（最初の2行目のtd）
                weight, adj_weight = None, None
                weight_tds = [td for td in rows[0].select("td") if td.text.strip() not in ["", lane_text, name]]
                for td in weight_tds:
                    v = self._sf(td.text)
                    if v and 40 <= v <= 80:
                        weight = v
                        break
                if len(rows) > 1:
                    for td in rows[1].select("td"):
                        v = self._sf(td.text)
                        if v and -5 <= v <= 5:
                            adj_weight = v
                            break

                # 展示タイム（colgroup width=61pxの列）
                exhibition_time = None
                all_tds = tbody.select("td")
                for td in all_tds:
                    txt = td.text.strip()
                    v = self._sf(txt)
                    if v and 6.0 <= v <= 8.0:  # 展示タイムの範囲
                        exhibition_time = v
                        break

                # チルト
                tilt = None
                for td in all_tds:
                    txt = td.text.strip()
                    if txt in ["-3.0", "-1.5", "0", "0.5", "1.0", "1.5", "2.0", "3.0"]:
                        try:
                            tilt = float(txt)
                        except:
                            pass
                        break

                # 部品交換
                parts = [span.text.strip() for span in tbody.select("span.label4")]

                info.exhibitions.append(ExhibitionInfo(
                    lane=lane, name=name, racer_no=racer_no,
                    weight=weight, adj_weight=adj_weight,
                    exhibition_time=exhibition_time,
                    tilt=tilt, parts_changed=parts,
                ))

        # ── スタート展示（table.is-w238）──────────
        start_table = soup.select_one("table.is-w238")
        if start_table:
            rows = start_table.select("tbody tr")
            for course_idx, row in enumerate(rows, start=1):
                cells = row.select("td")
                if not cells:
                    continue
                # 1つのtdに「艇番\n\nST」が入っている（行の位置=進入コース）
                text = cells[0].text.strip()
                parts = [p for p in text.split("\n") if p.strip()]
                if len(parts) < 2:
                    continue
                try:
                    boat_no = int(parts[0].strip())
                except ValueError:
                    continue
                st_text = parts[-1].strip()
                # F(フライング)/L(出遅れ)の記号を除いてST数値を取る
                st_clean = st_text.replace("F", "").replace("L", "").strip()
                st_val = self._sf(st_clean)
                info.start_exhibition.append(StartExhibition(
                    course=course_idx,   # 行の位置 = 進入コース
                    boat_no=boat_no,     # セルの数字 = 艇番
                    st=st_val,
                ))

        # ── 気象情報 ────────────────────────────
        weather = WeatherInfo()
        w_div = soup.select_one("div.weather1")
        if w_div:
            for unit in w_div.select("div.weather1_bodyUnit"):
                title_el = unit.select_one("span.weather1_bodyUnitLabelTitle")
                data_el  = unit.select_one("span.weather1_bodyUnitLabelData")
                if not title_el:
                    continue
                title = title_el.text.strip()
                data  = data_el.text.strip() if data_el else ""
                if title == "気温":   weather.temperature = self._sf(data)
                elif title == "風速": weather.wind_speed  = self._sf(data)
                elif title == "水温": weather.water_temp  = self._sf(data)
                elif title == "波高": weather.wave_height = self._sf(data)

            # 風向（CSSクラスから取得）
            wind_dir_el = w_div.select_one("p.weather1_bodyUnitImage[class*='is-wind']")
            if wind_dir_el:
                for cls in wind_dir_el.get("class", []):
                    if cls.startswith("is-wind") and cls != "is-wind":
                        weather.wind_direction = cls.replace("is-wind", "")

            # 天候（CSSクラスから取得）
            weather_el = w_div.select_one("p.weather1_bodyUnitImage[class*='is-weather']")
            if weather_el:
                for cls in weather_el.get("class", []):
                    if cls.startswith("is-weather") and cls != "is-weather":
                        weather_codes = {
                            "1": "晴", "2": "曇", "3": "雨", "4": "雪",
                            "5": "霧", "6": "霙", "7": "晴曇",
                        }
                        num = cls.replace("is-weather", "")
                        weather.weather = weather_codes.get(num, num)

        info.weather = weather
        logger.info(f"取得完了: {vc} {race_no}R 展示タイム={[e.exhibition_time for e in info.exhibitions]}")
        return info


# ─────────────────────────────────────────────
# エントリーポイント
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="直前情報スクレイパー")
    parser.add_argument("--venue", required=True, help="会場コード 例: 01")
    parser.add_argument("--race",  type=int, required=True, help="レース番号")
    parser.add_argument("--date",  help="日付 YYYYMMDD（省略時=今日）")
    parser.add_argument("--out",   help="出力JSONパス")
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()

    d = datetime.strptime(args.date, "%Y%m%d").date() if args.date else date.today()
    scraper = BeforeInfoScraper(delay=args.delay)
    info = scraper.get_before_info(d, args.venue, args.race)

    if info is None:
        print("データが取得できませんでした")
        return

    print(f"\n▼ {args.venue} {args.race}R 直前情報 ({d})")
    print(f"\n【展示タイム】")
    for e in info.exhibitions:
        print(f"  {e.lane}枠 {e.name} ST={e.exhibition_time} チルト={e.tilt} 部品={e.parts_changed}")
    print(f"\n【スタート展示】")
    for s in info.start_exhibition:
        print(f"  {s.course}コース 艇{s.boat_no} ST={s.st}")
    if info.weather:
        w = info.weather
        print(f"\n【気象】気温={w.temperature} 天候={w.weather} 風速={w.wind_speed} 水温={w.water_temp} 波高={w.wave_height}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(asdict(info), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n保存: {args.out}")


if __name__ == "__main__":
    main()