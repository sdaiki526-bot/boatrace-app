"""
競艇（BOAT RACE）スクレイピングツール
対象: www.boatrace.jp（テレボートログイン対応）
"""

import time
import json
import logging
import os
import re
from datetime import datetime, date
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_URL    = "https://www.boatrace.jp/owpc/pc/race"
LOGIN_URL   = "https://www.boatrace.jp/owpc/pc/login"
LOGIN_CHECK = "https://www.boatrace.jp/owpc/pc/teleboat/mypage"

VENUE_MAP = {
    "01": "桐生", "02": "戸田", "03": "江戸川", "04": "平和島",
    "05": "多摩川", "06": "浜名湖", "07": "蒲郡", "08": "常滑",
    "09": "津",   "10": "三国", "11": "びわこ", "12": "住之江",
    "13": "尼崎", "14": "鳴門", "15": "丸亀", "16": "児島",
    "17": "宮島", "18": "徳山", "19": "下関", "20": "若松",
    "21": "芦屋", "22": "福岡", "23": "唐津", "24": "大村",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en;q=0.9",
    "Referer": "https://www.boatrace.jp/",
}


@dataclass
class RacerInfo:
    lane: int
    racer_no: str
    name: str
    branch: str
    age: Optional[int]
    weight: Optional[float]
    rank: str
    flying_count: int = 0
    late_count: int = 0
    avg_start_time: Optional[float] = None
    win_rate_all: Optional[float] = None
    win_rate_2: Optional[float] = None
    win_rate_3: Optional[float] = None
    local_win_rate: Optional[float] = None
    local_win_rate_2: Optional[float] = None
    local_win_rate_3: Optional[float] = None
    motor_no: Optional[str] = None
    motor_2rate: Optional[float] = None
    boat_no: Optional[str] = None
    boat_2rate: Optional[float] = None


@dataclass
class OddsInfo:
    race_date: str
    venue_code: str
    venue_name: str
    race_no: int
    tansho: dict = field(default_factory=dict)
    fukusho: dict = field(default_factory=dict)
    niren_tan: dict = field(default_factory=dict)
    niren_fuku: dict = field(default_factory=dict)
    sanren_tan: dict = field(default_factory=dict)
    sanren_fuku: dict = field(default_factory=dict)


@dataclass
class RaceResult:
    race_date: str
    venue_code: str
    venue_name: str
    race_no: int
    arrival: list = field(default_factory=list)
    payouts: dict = field(default_factory=dict)
    start_times: dict = field(default_factory=dict)


class BoatraceScraper:
    def __init__(self, delay: float = 1.5):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._logged_in = False

    def login(self, kanyusya_no=None, ansyo_no=None, password=None):
        kanyusya_no = kanyusya_no or os.getenv("BOATRACE_KANYUSYA_NO", "")
        ansyo_no    = ansyo_no    or os.getenv("BOATRACE_ANSYO_NO",    "")
        password    = password    or os.getenv("BOATRACE_PASSWORD",    "")

        if not all([kanyusya_no, ansyo_no, password]):
            logger.error(".env に BOATRACE_KANYUSYA_NO / BOATRACE_ANSYO_NO / BOATRACE_PASSWORD を設定してください")
            return False

        try:
            resp = self.session.get(LOGIN_URL, timeout=15)
            soup = BeautifulSoup(resp.text, "html.parser")
            view_state = ""
            vs_el = soup.find("input", {"id": "j_id1:javax.faces.ViewState:0"})
            if vs_el:
                view_state = vs_el.get("value", "stateless")
        except Exception as e:
            logger.error(f"ログインページ取得失敗: {e}")
            return False

        payload = {
            "TENT010_TENTPC010PRForm": "TENT010_TENTPC010PRForm",
            "in_KanyusyaNo": kanyusya_no,
            "in_AnsyoNo":    ansyo_no,
            "in_PassWord":   password,
            "in_AuthAfterUrl": "",
            "javax.faces.ViewState": view_state,
            "TENTP017A_2": "ログインする",
        }

        try:
            resp = self.session.post(
                LOGIN_URL, data=payload,
                headers={"Referer": LOGIN_URL},
                timeout=15, allow_redirects=True,
            )
            time.sleep(self.delay)
        except Exception as e:
            logger.error(f"ログインPOST失敗: {e}")
            return False

        if self._check_login():
            logger.info("✅ ログイン成功")
            self._logged_in = True
            return True
        else:
            logger.error("❌ ログイン失敗。加入者番号・暗証番号・パスワードを確認してください")
            return False

    def _check_login(self):
        try:
            resp = self.session.get(LOGIN_CHECK, timeout=15)
            return "/owpc/pc/login" not in resp.url and "マイページ" in resp.text
        except Exception:
            return False

    def ensure_login(self):
        if not self._logged_in:
            if not self.login():
                raise RuntimeError("ログインできませんでした。.env の設定を確認してください")

    def _get(self, url, params=None):
        try:
            logger.info(f"GET {url} params={params}")
            resp = self.session.get(url, params=params, timeout=12)
            resp.raise_for_status()
            time.sleep(self.delay)
            if "/owpc/pc/login" in resp.url:
                logger.warning("セッション切れ。再ログインします")
                self._logged_in = False
                self.ensure_login()
                resp = self.session.get(url, params=params, timeout=12)
                time.sleep(self.delay)
            return BeautifulSoup(resp.text, "html.parser")
        except requests.RequestException as e:
            logger.error(f"リクエスト失敗: {e}")
            return None

    @staticmethod
    def _safe_float(text):
        try:
            return float(str(text).strip().replace(",", ""))
        except (ValueError, AttributeError):
            return None

    @staticmethod
    def _safe_int(text):
        try:
            return int(str(text).strip().replace(",", ""))
        except (ValueError, AttributeError):
            return None

    @staticmethod
    def _fmt_date(d):
        return d.strftime("%Y%m%d")

    def get_racelist(self, race_date, venue_code, race_no):
        self.ensure_login()
        url = f"{BASE_URL}/racelist"
        params = {
            "hd":  self._fmt_date(race_date),
            "jcd": venue_code.zfill(2),
            "rno": race_no,
        }
        soup = self._get(url, params)
        if soup is None:
            return []
        if "データがありません" in soup.text:
            logger.warning(f"出走表データなし: {venue_code} {race_no}R {race_date}")
            return []
        return self._parse_racelist(soup)

    def _parse_racelist(self, soup):
        racers = []
        tables = soup.select("table")
        target = max(tables, key=lambda t: len(t.select("tr")), default=None)
        if not target:
            return []

        for row in target.select("tr"):
            cells = row.select("td")
            if len(cells) < 20:
                continue

            def cell(i):
                return cells[i].text.strip() if len(cells) > i else ""

            # 枠番（全角→半角）
            lane_text = cell(0)
            for z, h in zip("１２３４５６", "123456"):
                lane_text = lane_text.replace(z, h)
            lane = self._safe_int(lane_text)
            if not lane or lane not in range(1, 7):
                continue

            # セル[2]: 登録番号・級別・選手名・支部・年齢・体重
            cell2 = cells[2]
            racer_no = ""
            m = re.search(r'\b(\d{4})\b', cell2.text)
            if m:
                racer_no = m.group(1)

            rank = ""
            rank_el = cell2.select_one("span")
            if rank_el:
                rank = rank_el.text.strip()

            name = ""
            name_el = cell2.select_one("a")
            if name_el:
                name = name_el.text.strip()

            branch, age, weight = "", None, None
            full_text = cell2.text
            age_m = re.search(r'(\d+)歳', full_text)
            weight_m = re.search(r'([\d.]+)kg', full_text)
            branch_m = re.search(r'([^\s/]+)/[^\s/]+\n', full_text)
            if age_m:    age    = int(age_m.group(1))
            if weight_m: weight = float(weight_m.group(1))
            if branch_m: branch = branch_m.group(1)

            # セル[3]: F数・L数・平均ST
            flying, late, avg_st = 0, 0, None
            c3 = cell(3)
            f_m = re.search(r'F(\d+)', c3)
            l_m = re.search(r'L(\d+)', c3)
            if f_m: flying = int(f_m.group(1))
            if l_m: late   = int(l_m.group(1))
            for s in [x.strip() for x in c3.split("\n") if x.strip()]:
                v = self._safe_float(s)
                if v is not None and 0 <= v <= 1.0:
                    avg_st = v
                    break

            c4 = [x.strip() for x in cell(4).split("\n") if x.strip()]
            win_rate_all = self._safe_float(c4[0]) if len(c4) > 0 else None
            win_rate_2   = self._safe_float(c4[1]) if len(c4) > 1 else None
            win_rate_3   = self._safe_float(c4[2]) if len(c4) > 2 else None

            c5 = [x.strip() for x in cell(5).split("\n") if x.strip()]
            local_win = self._safe_float(c5[0]) if len(c5) > 0 else None
            local_2   = self._safe_float(c5[1]) if len(c5) > 1 else None
            local_3   = self._safe_float(c5[2]) if len(c5) > 2 else None

            c6 = [x.strip() for x in cell(6).split("\n") if x.strip()]
            motor_no = c6[0] if len(c6) > 0 else None
            motor_2r = self._safe_float(c6[1]) if len(c6) > 1 else None

            c7 = [x.strip() for x in cell(7).split("\n") if x.strip()]
            boat_no = c7[0] if len(c7) > 0 else None
            boat_2r = self._safe_float(c7[1]) if len(c7) > 1 else None

            racers.append(RacerInfo(
                lane=lane, racer_no=racer_no, name=name,
                branch=branch, age=age, weight=weight, rank=rank,
                flying_count=flying, late_count=late, avg_start_time=avg_st,
                win_rate_all=win_rate_all, win_rate_2=win_rate_2, win_rate_3=win_rate_3,
                local_win_rate=local_win, local_win_rate_2=local_2, local_win_rate_3=local_3,
                motor_no=motor_no, motor_2rate=motor_2r,
                boat_no=boat_no, boat_2rate=boat_2r,
            ))

        return racers

    def get_odds(self, race_date, venue_code, race_no):
        vc = venue_code.zfill(2)
        odds = OddsInfo(
            race_date=self._fmt_date(race_date),
            venue_code=vc, venue_name=VENUE_MAP.get(vc, vc),
            race_no=race_no,
        )
        odds.tansho, odds.fukusho = self._get_tansho_fukusho(race_date, vc, race_no)
        odds.niren_tan, odds.niren_fuku = self._get_niren(race_date, vc, race_no)
        odds.sanren_tan  = self._get_sanren_tan(race_date, vc, race_no)
        odds.sanren_fuku = self._get_sanren_fuku(race_date, vc, race_no)
        return odds

    def _get_tansho_fukusho(self, race_date, vc, rno):
        soup = self._get(f"{BASE_URL}/oddstf", {"hd": self._fmt_date(race_date), "jcd": vc, "rno": rno})
        tansho, fukusho = {}, {}
        if not soup: return tansho, fukusho
        for tbl in soup.select("table.is-w246"):
            cap = tbl.select_one("caption")
            cap_text = cap.text.strip() if cap else ""
            for row in tbl.select("tbody tr"):
                cells = row.select("td")
                if len(cells) < 2: continue
                key = cells[0].text.strip()
                val = self._safe_float(cells[-1].text)
                if "単勝" in cap_text: tansho[key] = val
                elif "複勝" in cap_text: fukusho[key] = val
        return tansho, fukusho

    def _get_niren(self, race_date, vc, rno):
        soup = self._get(f"{BASE_URL}/odds2tf", {"hd": self._fmt_date(race_date), "jcd": vc, "rno": rno})
        tan, fuku = {}, {}
        if not soup: return tan, fuku
        for row in soup.select("table.is-w495 tbody tr"):
            cells = row.select("td")
            if len(cells) < 2: continue
            if cells[0].text.strip(): tan[cells[0].text.strip()] = self._safe_float(cells[1].text)
            if len(cells) > 3 and cells[2].text.strip(): fuku[cells[2].text.strip()] = self._safe_float(cells[3].text)
        return tan, fuku

    def _get_sanren_tan(self, race_date, vc, rno):
        """
        3連単オッズページをパースする。

        テーブルは6つの列ブロックに分かれており、各ブロックのヘッダ（1〜6）が
        「3着艇番号」を表す。各ブロック内の各行は
        [1着艇番号(rowspanあり), 2着艇番号, オッズ] の3セル組で構成される。

        例: ブロック1(3着=1)の最初の行が [2, 3, 69.7] なら
            "2-3-1" のオッズが 69.7
        """
        soup = self._get(f"{BASE_URL}/odds3t", {"hd": self._fmt_date(race_date), "jcd": vc, "rno": rno})
        result = {}
        if not soup:
            return result

        # is-w748が無い場合に対応: div.table1内の素のtableを探す
        table = soup.select_one("table.is-w748")
        if not table:
            # 締切時刻テーブルを除外し、3連単オッズテーブル（theadにis-boatColorを含む）を探す
            for cand in soup.select(".table1 table"):
                if cand.select_one("th.is-boatColor1"):
                    table = cand
                    break
        if not table:
            return result

        rows = table.select("tbody tr")
        if not rows:
            return result

        n_blocks = 6
        # 各ブロックの「現在の1着艇番号」（rowspanで複数行に渡るため保持）
        current_first = [None] * n_blocks

        for row in rows:
            cells = row.select("td")
            cell_idx = 0
            for block in range(n_blocks):
                third = block + 1  # 3着艇番号 (列ブロック = 3着)
                if cell_idx >= len(cells):
                    break

                cell = cells[cell_idx]
                # rowspan付きセル = 新しい1着艇番号
                if cell.get("rowspan"):
                    current_first[block] = cell.text.strip()
                    cell_idx += 1
                    if cell_idx >= len(cells):
                        break
                    cell = cells[cell_idx]

                first = current_first[block]
                second = cell.text.strip()
                cell_idx += 1

                if cell_idx >= len(cells):
                    break
                odds_cell = cells[cell_idx]
                cell_idx += 1

                if first and second and first.isdigit() and second.isdigit():
                    key = f"{first}-{second}-{third}"
                    result[key] = self._safe_float(odds_cell.text)

        return result

    def _get_sanren_fuku(self, race_date, vc, rno):
        soup = self._get(f"{BASE_URL}/odds3f", {"hd": self._fmt_date(race_date), "jcd": vc, "rno": rno})
        result = {}
        if not soup: return result
        for row in soup.select("table.is-w748 tbody tr"):
            cells = row.select("td")
            for i in range(0, len(cells) - 1, 2):
                k = cells[i].text.strip()
                if k: result[k] = self._safe_float(cells[i+1].text)
        return result

    def get_result(self, race_date, venue_code, race_no):
        vc = venue_code.zfill(2)
        soup = self._get(f"{BASE_URL}/raceresult", {"hd": self._fmt_date(race_date), "jcd": vc, "rno": race_no})
        if not soup: return None
        return self._parse_result(soup, race_date, vc, race_no)

    def _parse_result(self, soup, race_date, vc, rno):
        result = RaceResult(
            race_date=self._fmt_date(race_date),
            venue_code=vc, venue_name=VENUE_MAP.get(vc, vc), race_no=rno,
        )
        tables_495 = soup.select("table.is-w495")
        if tables_495:
            for tbody in tables_495[0].select("tbody"):
                cells = tbody.select("td")
                if len(cells) < 2: continue
                boat_el = tbody.select_one("td.is-fBold")
                if boat_el:
                    b = self._safe_int(boat_el.text)
                    if b and b in range(1, 7):
                        result.arrival.append(b)
        for div in soup.select("div.table1_boatImage1"):
            lane_el = div.select_one("span.table1_boatImage1Number")
            time_el = div.select_one("span.table1_boatImage1TimeInner")
            if lane_el and time_el:
                lane = self._safe_int(lane_el.text)
                time_text = time_el.text.strip().split()[0] if time_el.text.strip() else ""
                if lane and time_text:
                    result.start_times[str(lane)] = time_text
        if len(tables_495) >= 3:
            current_bet_type = ""
            for tbody in tables_495[2].select("tbody"):
                for row in tbody.select("tr"):
                    cells = row.select("td")
                    if not cells: continue
                    if cells[0].get("rowspan"):
                        current_bet_type = cells[0].text.strip()
                        if len(cells) >= 3:
                            combo = cells[1].text.strip()
                            payout_text = cells[2].text.replace("¥","").replace(",","").strip()
                            payout = self._safe_int(payout_text)
                            if combo and payout:
                                result.payouts[f"{current_bet_type}_{combo}"] = payout
        return result

    def get_all_races(self, race_date, venue_code, max_race=12):
        vc = venue_code.zfill(2)
        races = []
        for rno in range(1, max_race + 1):
            logger.info(f"=== {VENUE_MAP.get(vc,vc)} {rno}R ===")
            racers = self.get_racelist(race_date, vc, rno)
            odds   = self.get_odds(race_date, vc, rno)
            result = self.get_result(race_date, vc, rno)
            races.append({
                "race_no": rno,
                "racers":  [asdict(r) for r in racers],
                "odds":    asdict(odds),
                "result":  asdict(result) if result else None,
            })
        return races

    def save_racers_csv(self, racers, path):
        pd.DataFrame([asdict(r) for r in racers]).to_csv(path, index=False, encoding="utf-8-sig")
        logger.info(f"出走表 → {path}")

    def save_odds_json(self, odds, path):
        Path(path).write_text(json.dumps(asdict(odds), ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"オッズ → {path}")

    def save_result_json(self, result, path):
        Path(path).write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"結果 → {path}")

    def save_all_json(self, races, path):
        Path(path).write_text(json.dumps(races, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"全レース → {path}")


# ─────────────────────────────────────────────
# 締切時刻取得（展示タイム取得タイミング判定用）
# ─────────────────────────────────────────────
def get_deadline_times(scraper: "BoatraceScraper", race_date, venue_code: str) -> dict:
    """
    指定会場の各レースの締切予定時刻を取得する。
    raceindexページの各レース行（1R〜12R）から締切時刻を取得する。
    戻り値: {race_no: datetime} の辞書（取得できなければ空辞書）
    """
    import re as _re
    from datetime import datetime as _dt
    vc = venue_code.zfill(2)
    soup = scraper._get(f"{BASE_URL}/raceindex", {"hd": scraper._fmt_date(race_date), "jcd": vc})
    if soup is None:
        return {}

    result = {}
    tables = soup.select("table")
    if not tables:
        return result

    # 最も行数が多いテーブルがレース一覧テーブル
    target = max(tables, key=lambda t: len(t.select("tbody tr")), default=None)
    if not target:
        return result

    for tbody in target.select("tbody"):
        row = tbody.select_one("tr")
        if not row:
            continue
        cells = row.select("td")
        if len(cells) < 2:
            continue

        # 1列目: "1R" のようなレース番号リンク
        race_link = cells[0].select_one("a")
        race_text = race_link.text.strip() if race_link else cells[0].text.strip()
        m = _re.match(r'(\d+)R', race_text)
        if not m:
            continue
        rno = int(m.group(1))

        # 2列目: "15:25" のような締切時刻
        time_text = cells[1].text.strip()
        tm = _re.match(r'(\d{1,2}):(\d{2})', time_text)
        if not tm:
            continue

        hh, mm = int(tm.group(1)), int(tm.group(2))
        dt = _dt.combine(race_date, _dt.min.time()).replace(hour=hh, minute=mm)
        result[rno] = dt

    return result