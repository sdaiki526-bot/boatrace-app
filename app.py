"""
競艇予想ツール - Streamlit Web アプリ（ライト・スポーティーデザイン）
起動方法: streamlit run app.py
"""

import streamlit as st
import json
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, datetime, timezone, timedelta as _td
from pathlib import Path
import sys

JST = timezone(_td(hours=9))

def today_jst():
    return datetime.now(JST).date()

sys.path.insert(0, str(Path(__file__).parent))

from boatrace_scraper import BoatraceScraper, VENUE_MAP
from predictor import BoatracePredictor, MLPredictor
from dashboard import render_dashboard


def _score_metrics(pred):
    """予想結果から1位スコアと1位-2位の差を計算する"""
    sorted_scores = sorted(pred.scores, key=lambda s: s.predicted_rank)
    if len(sorted_scores) < 2:
        return None, None
    top_score = sorted_scores[0].total_score
    score_gap = sorted_scores[0].total_score - sorted_scores[1].total_score
    return round(top_score, 3), round(score_gap, 3)


from before_info_scraper import BeforeInfoScraper

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from supabase import create_client

def get_supabase():
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
    except Exception as e:
        st.session_state["_supabase_error"] = f"Secrets読めない: {e}"
        return None
    try:
        return create_client(url, key)
    except Exception as e:
        st.session_state["_supabase_error"] = f"接続失敗: {e}"
        return None

# セッションに接続を保持し、失敗したら毎回再接続を試みる
supabase = get_supabase()

# ─────────────────────────────────────────────
# ページ設定
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="競艇予想ツール",
    page_icon="🚤",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* カジノ・ナイター風テーマ（黒×金） */
    .stApp { background-color: #0a0f1c; color: #e2e8f0; font-size: 16px; }
    .main .block-container { padding: 1.5rem 2rem; max-width: 1400px; }
    p, span, div, label, li { font-size: 1rem; }
    h1 { font-size: 1.8rem !important; color: #f5c542 !important; }
    h2 { font-size: 1.5rem !important; color: #f5c542 !important; }
    h3 { font-size: 1.3rem !important; color: #f5c542 !important; }
    h4 { font-size: 1.1rem !important; color: #f1f5f9 !important; }

    .stSelectbox > div > div { background: #1a2540 !important; border-color: #2d3a52 !important; }
    .stSelectbox > div > div > div { color: #f1f5f9 !important; }
    .stSelectbox svg { color: #f5c542 !important; }
    .stSelectbox label, .stMultiSelect label { color: #8b9bb4 !important; }

    /* ヘッダー */
    .header-box {
        background: linear-gradient(135deg, #0f1729 0%, #1a2540 60%, #0f1729 100%);
        border: 1px solid #b8860b;
        border-radius: 14px;
        padding: 1.2rem 1.8rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1.2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        position: relative;
        overflow: hidden;
    }
    .header-box::after {
        content: "";
        position: absolute;
        right: -40px; top: -40px;
        width: 160px; height: 160px;
        background: rgba(245,197,66,0.06);
        border-radius: 50%;
    }
    .header-icon { flex-shrink: 0; }
    .header-title { font-size: 1.7rem; font-weight: 800; color: #f5c542; margin: 0; letter-spacing: 0.02em; }
    .header-date { color: #8b9bb4; font-size: 0.85rem; margin: 0.2rem 0 0; font-weight: 600; }
    .header-meta { margin-left: auto; text-align: right; z-index: 1; }
    .header-meta-value { font-size: 1.5rem; font-weight: 800; color: #f5c542; line-height: 1; }
    .header-meta-label { font-size: 0.75rem; color: #8b9bb4; font-weight: 600; margin-top: 0.2rem; }

    /* カード */
    .stat-card {
        background: linear-gradient(135deg, #1a2540, #0f1729);
        border: 1px solid #2d3a52;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .stat-value { font-size: 1.8rem; font-weight: 800; color: #f5c542; }
    .stat-label { font-size: 0.78rem; color: #8b9bb4; margin-top: 0.2rem; }

    /* 買い目ボックス */
    .buy-box {
        background: #1a2540;
        border: 1px solid #b8860b;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .buy-title { font-size: 1rem; font-weight: 700; color: #f5c542; margin-bottom: 0.8rem; }
    .buy-combo {
        background: #0f1729;
        border: 1px solid #b8860b;
        border-radius: 6px;
        padding: 0.4rem 0.9rem;
        margin: 0.3rem 0.2rem;
        font-size: 1rem;
        font-weight: 700;
        color: #f5c542;
        display: inline-block;
        letter-spacing: 0.05em;
    }

    /* ボタン */
    .stButton > button {
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.95rem;
        border: 1px solid #2d3a52 !important;
        background: #1a2540 !important;
        color: #f1f5f9 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #b8860b, #f5c542) !important;
        border-color: #f5c542 !important;
        color: #0f1729 !important;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* 出走表行 */
    .racer-row {
        display: flex;
        align-items: center;
        padding: 0.7rem 1rem;
        background: #1a2540;
        border: 1px solid #2d3a52;
        border-radius: 8px;
        margin: 4px 0;
        gap: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }
    .lane-badge {
        width: 30px; height: 30px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-weight: 800; font-size: 0.85rem; flex-shrink: 0;
    }
    .lane-1 { background: #eab308; color: #000; }
    .lane-2 { background: #2563eb; color: #fff; }
    .lane-3 { background: #dc2626; color: #fff; }
    .lane-4 { background: #6b7280; color: #fff; }
    .lane-5 { background: #f97316; color: #fff; }
    .lane-6 { background: #16a34a; color: #fff; }

    /* 的中・ハズレ */
    .hit-badge { background: #064e3b; border: 1px solid #10b981; color: #6ee7b7; border-radius: 6px; padding: 2px 10px; font-size: 0.82rem; font-weight: 700; }
    .miss-badge { background: #7f1d1d; border: 1px solid #ef4444; color: #fca5a5; border-radius: 6px; padding: 2px 10px; font-size: 0.82rem; font-weight: 700; }
    .pending-badge { background: #1a2540; border: 1px solid #2d3a52; color: #8b9bb4; border-radius: 6px; padding: 2px 10px; font-size: 0.82rem; font-weight: 700; }

    /* 記録カード */
    .record-card {
        background: #1a2540;
        border: 1px solid #2d3a52;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        margin: 4px 0;
        display: flex;
        align-items: center;
        gap: 1rem;
        flex-wrap: wrap;
    }
    hr { border-color: #2d3a52 !important; }

    /* Expander */
    [data-testid="stExpander"] {
        background: #1a2540;
        border: 1px solid #2d3a52 !important;
        border-radius: 8px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }
    [data-testid="stExpander"] summary { color: #f1f5f9 !important; font-weight: 600 !important; }
    [data-testid="stExpander"] summary span { color: #f1f5f9 !important; }

    /* メトリクス */
    [data-testid="stMetric"] { background: #1a2540; border: 1px solid #2d3a52; border-radius: 8px; padding: 0.8rem; }
    [data-testid="stMetricLabel"] { color: #8b9bb4 !important; }
    [data-testid="stMetricValue"] { color: #f5c542 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ユーティリティ
# ─────────────────────────────────────────────
RECORD_FILE = Path("prediction_records.json")
CACHE_FILE = Path(f"cache_racers_{today_jst().strftime('%Y%m%d')}.json")

def load_records():
    # Supabaseから取得（優先）。1000件のデフォルト制限を超えるためページネーションする
    if supabase:
        try:
            all_data = []
            page_size = 1000
            offset = 0
            while True:
                res = (
                    supabase.table("prediction_records")
                    .select("*")
                    .order("race_date", desc=True)
                    .range(offset, offset + page_size - 1)
                    .execute()
                )
                if not res.data:
                    break
                all_data.extend(res.data)
                if len(res.data) < page_size:
                    break
                offset += page_size

            records = []
            for row in all_data:
                sanren_tan = row.get("sanren_tan")
                if isinstance(sanren_tan, str):
                    sanren_tan = json.loads(sanren_tan)
                records.append({
                    "race_date":   row["race_date"],
                    "venue_code":  row["venue_code"],
                    "venue_name":  row["venue_name"],
                    "race_no":     row["race_no"],
                    "tansho":      row.get("tansho"),
                    "sanren_tan":  sanren_tan or [],
                    "sanren_fuku": row.get("sanren_fuku"),
                    "hit":         row.get("hit"),
                    "actual":      row.get("actual") or "",
                    "payout":      row.get("payout"),
                    "top_score":   row.get("top_score"),
                    "score_gap":   row.get("score_gap"),
                    "odds_value":  row.get("odds_value"),
                    "weather":     row.get("weather"),
                    "wind_speed":  row.get("wind_speed"),
                    "wave_height": row.get("wave_height"),
                    "water_temp":  row.get("water_temp"),
                })
            return records
        except Exception as e:
            st.warning(f"Supabase読み込み失敗: {e}")

    # フォールバック: ローカルJSON
    if RECORD_FILE.exists():
        return json.loads(RECORD_FILE.read_text(encoding="utf-8"))
    return []

def save_record(record):
    # Supabaseに保存
    if supabase:
        try:
            db_record = dict(record)
            db_record["sanren_tan"] = json.dumps(record["sanren_tan"], ensure_ascii=False)
            supabase.table("prediction_records").upsert(
                db_record, on_conflict="race_date,venue_code,race_no"
            ).execute()
            return
        except Exception as e:
            st.warning(f"Supabase保存失敗: {e}")

    # フォールバック: ローカルJSON
    records = load_records()
    key = f"{record['race_date']}_{record['venue_code']}_{record['race_no']}"
    records = [r for r in records if f"{r['race_date']}_{r['venue_code']}_{r['race_no']}" != key]
    records.append(record)
    RECORD_FILE.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

def load_cache():
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    return {}

def save_cache(today_racers):
    from dataclasses import asdict
    s = {}
    for venue, races in today_racers.items():
        s[venue] = {}
        for rno, racers in races.items():
            s[venue][str(rno)] = [asdict(r) for r in racers]
    CACHE_FILE.write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")

def restore_cache(data):
    from boatrace_scraper import RacerInfo
    r = {}
    for venue, races in data.items():
        r[venue] = {}
        for rno, racers in races.items():
            r[venue][int(rno)] = [RacerInfo(**x) for x in racers]
    return r

def check_hit(sanren_tan_combos, arrival):
    if len(arrival) < 3:
        return False, ""
    actual = f"{arrival[0]}-{arrival[1]}-{arrival[2]}"
    return actual in sanren_tan_combos, actual

def get_scraper():
    if not st.session_state.logged_in:
        sc = BoatraceScraper(delay=1.0)
        if sc.login():
            st.session_state.scraper = sc
            st.session_state.logged_in = True
        else:
            st.error("ログイン失敗。.envファイルを確認してください")
            return None
    return st.session_state.scraper

def save_fetch_history(fetch_type, race_date_str, venue_code, venue_name, race_no):
    if not supabase:
        return
    try:
        supabase.table("fetch_history").insert({
            "fetch_type":  fetch_type,
            "race_date":   race_date_str,
            "venue_code":  venue_code,
            "venue_name":  venue_name,
            "race_no":     race_no,
        }).execute()
    except Exception as e:
        st.warning(f"履歴保存失敗: {e}")

def load_fetch_history(fetch_type, limit=10):
    if not supabase:
        return []
    try:
        res = (
            supabase.table("fetch_history")
            .select("*")
            .eq("fetch_type", fetch_type)
            .order("fetched_at", desc=True)
            .limit(limit)
            .execute()
        )
        return res.data
    except Exception as e:
        st.warning(f"履歴読み込み失敗: {e}")
        return []

def load_bet_records():
    """実戦記録（実際に買ったレース）を読む"""
    if not supabase:
        return []
    try:
        res = supabase.table("bet_records").select("*").order("race_date", desc=True).execute()
        return res.data or []
    except Exception as e:
        st.warning(f"実戦記録の読み込み失敗: {e}")
        return []


def save_bet_record(record):
    """実戦記録を1件保存する"""
    if not supabase:
        return False
    try:
        db = dict(record)
        if isinstance(db.get("sanren_tan"), list):
            db["sanren_tan"] = json.dumps(db["sanren_tan"], ensure_ascii=False)
        supabase.table("bet_records").upsert(
            db, on_conflict="race_date,venue_code,race_no"
        ).execute()
        return True
    except Exception as e:
        st.warning(f"実戦記録の保存失敗: {e}")
        return False
# ─────────────────────────────────────────────
# セッション初期化
# ─────────────────────────────────────────────
for key, val in {
    "scraper": None, "logged_in": False, "prediction": None,
    "before_info": None, "last_venue": None, "last_race": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

def load_today_racelist_from_supabase():
    if not supabase:
        return {}
    try:
        date_str = today_jst().strftime("%Y%m%d")
        res = supabase.table("today_racelist").select("*").eq("race_date", date_str).execute()
        if not res.data:
            return {}
        from boatrace_scraper import RacerInfo
        result = {}
        error_count = 0
        for row in res.data:
            venue = row["venue_code"]
            rno = row["race_no"]
            racers_data = row["racers"]
            if isinstance(racers_data, str):
                racers_data = json.loads(racers_data)
            if venue not in result:
                result[venue] = {}
            try:
                result[venue][rno] = [RacerInfo(**r) for r in racers_data]
            except Exception as e2:
                error_count += 1
        return result
    except Exception as e:
        st.warning(f"出走表Supabase読み込み失敗: {e}")
        return {}

def load_deadline_times_from_supabase():
    """{(venue_code, race_no): "HH:MM"} の辞書を返す"""
    if not supabase:
        return {}
    try:
        date_str = today_jst().strftime("%Y%m%d")
        res = supabase.table("today_racelist").select("venue_code,venue_name,race_no,deadline_time").eq("race_date", date_str).execute()
        result = {}
        for row in res.data or []:
            if row.get("deadline_time"):
                result[(row["venue_code"], row["race_no"])] = {
                    "deadline_time": row["deadline_time"],
                    "venue_name": row["venue_name"],
                }
        return result
    except Exception as e:
        return {}

today_str_check = today_jst().strftime("%Y%m%d")

# 日付が変わったらセッションをリセット
if st.session_state.get("cache_date") != today_str_check:
    st.session_state.cache_date = today_str_check
    st.session_state.today_racers = {}
    st.session_state.deadline_times = {}
    st.session_state.selected_race = None

if "today_racers" not in st.session_state or not st.session_state.today_racers:
    cached = load_cache()
    if cached:
        st.session_state.today_racers = restore_cache(cached)
    else:
        st.session_state.today_racers = load_today_racelist_from_supabase()

if "deadline_times" not in st.session_state or not st.session_state.deadline_times:
    st.session_state.deadline_times = load_deadline_times_from_supabase()

if "selected_race" not in st.session_state:
    st.session_state.selected_race = None

# ─────────────────────────────────────────────
# ヘッダー
# ─────────────────────────────────────────────
_venue_count = len(st.session_state.today_racers) if st.session_state.today_racers else 0
_race_count = sum(len(v) for v in st.session_state.today_racers.values()) if st.session_state.today_racers else 0

_meta_html = ""
if _venue_count:
    _meta_html = (
        f'<div class="header-meta">'
        f'<div class="header-meta-value">{_venue_count}会場</div>'
        f'<div class="header-meta-label">{_race_count}レース取得済み</div>'
        f'</div>'
    )

_BOAT_SVG = (
    '<svg class="header-icon" width="64" height="56" viewBox="0 0 64 56" xmlns="http://www.w3.org/2000/svg">'
    '<ellipse cx="14" cy="46" rx="13" ry="5" fill="#ffffff" opacity="0.18"/>'
    '<ellipse cx="30" cy="50" rx="16" ry="4" fill="#ffffff" opacity="0.12"/>'
    '<path d="M2 40 C8 34, 14 34, 18 38 C22 42, 28 42, 32 38" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" opacity="0.55"/>'
    '<path d="M6 46 C12 41, 18 41, 22 45 C26 49, 32 49, 36 45" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" opacity="0.35"/>'
    '<path d="M16 38 L52 30 C56 29, 59 31, 59 35 L59 38 C59 41, 56 43, 52 42 L18 42 Z" fill="#fbbf24"/>'
    '<path d="M18 42 L52 42 C56 43, 56 47, 52 47 L24 47 C20 47, 17 45, 18 42 Z" fill="#f97316"/>'
    '<rect x="30" y="18" width="4" height="13" rx="1" fill="#1f2937"/>'
    '<path d="M34 19 L50 26 L34 28 Z" fill="#ef4444"/>'
    '<circle cx="46" cy="34" r="2.2" fill="#1f2937"/>'
    '<path d="M58 36 C62 35, 64 33, 63 30 C61 33, 58 33, 56 35 Z" fill="#bfdbfe" opacity="0.85"/>'
    '<path d="M60 33 C64 31, 66 28, 64 25 C62 29, 59 30, 57 32 Z" fill="#bfdbfe" opacity="0.6"/>'
    '</svg>'
)

header_html = (
    f'<div class="header-box">'
    f'{_BOAT_SVG}'
    f'<div>'
    f'<p class="header-title">競艇予想ツール</p>'
    f'<p class="header-date">📅 {today_jst().strftime("%Y年%m月%d日")}</p>'
    f'</div>'
    f'{_meta_html}'
    f'</div>'
)
st.markdown(header_html, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# サイドバーナビゲーション
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* サイドバー - ブルーグラデーション（ライトテーマに映える） */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a8a 0%, #1d4ed8 100%);
        border-right: none;
    }
    [data-testid="stSidebar"] > div { padding-top: 1.5rem; }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] .stRadio > div { gap: 0.3rem; }
    [data-testid="stSidebar"] .stRadio label {
        background: rgba(255,255,255,0.08);
        border: none;
        border-radius: 10px;
        padding: 0.65rem 1rem;
        color: #e2e8f0 !important;
        font-weight: 600;
        font-size: 0.95rem;
        cursor: pointer;
        transition: all 0.15s;
        width: 100%;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255,255,255,0.18);
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stRadio label p,
    [data-testid="stSidebar"] .stRadio label span { color: inherit !important; }
    [data-testid="stSidebar"] .stRadio [aria-checked="true"] + label,
    [data-testid="stSidebar"] .stRadio input:checked + label {
        background: rgba(255,255,255,0.25);
        color: #fff !important;
    }
    [data-testid="stSidebar"] .stRadio input[type="radio"] { display: none; }
    [data-testid="stSidebar"] .stRadio > label > div:first-child { display: none; }
    [data-testid="stSidebar"] .stCaption { color: #bfdbfe !important; font-size: 0.78rem; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.2) !important; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(
        "<div style='display:flex;align-items:center;gap:0.7rem;margin-bottom:1.5rem'>"
        "<svg width='32' height='28' viewBox='0 0 64 56' xmlns='http://www.w3.org/2000/svg'>"
        "<path d='M16 38 L52 30 C56 29, 59 31, 59 35 L59 38 C59 41, 56 43, 52 42 L18 42 Z' fill='#fbbf24'/>"
        "<path d='M18 42 L52 42 C56 43, 56 47, 52 47 L24 47 C20 47, 17 45, 18 42 Z' fill='#f97316'/>"
        "<path d='M34 19 L50 26 L34 28 Z' fill='#ef4444'/>"
        "<rect x='30' y='18' width='4' height='13' rx='1' fill='#1f2937'/>"
        "</svg>"
        "<span style='font-size:1.2rem;font-weight:800;color:#f1f5f9;letter-spacing:0.02em'>競艇予想ツール</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    menu = st.radio(
        "",
        ["🏠 ホーム", "🔥 ピックアップ", "🎯 予想", "📊 直前情報", "📊 成績"],
        label_visibility="collapsed",
    )

    # 「成績」内のサブタブ
    if menu == "📊 成績":
        sub = st.radio(
            "成績メニュー",
            ["📋 結果確認", "📈 成績記録", "💎 高配当殿堂"],
            label_visibility="collapsed",
        )
        page = sub
    else:
        page = menu
    
    st.markdown("---")
    st.caption(f"📅 {today_jst().strftime('%Y年%m月%d日')}")
    if _venue_count:
        st.caption(f"🏟 {_venue_count}会場 {_race_count}レース取得済み")

# ─────────────────────────────────────────────
# ホーム（会場グリッドダッシュボード）
# ─────────────────────────────────────────────
if page == "🏠 ホーム":
    render_dashboard(supabase, today_jst().strftime("%Y%m%d"))

# ─────────────────────────────────────────────
# ピックアップ
# ─────────────────────────────────────────────
PICKUP_TOP_SCORE_MIN = 35.0
PICKUP_SCORE_GAP_MIN = 15.0

if page == "🔥 ピックアップ":
    records = load_records()
    today_str = today_jst().strftime("%Y%m%d")
    today_records = [r for r in records if r["race_date"] == today_str]

    # ─────────────────────────────────────────────
    # 💰 狙い目レース（回収率重視）
    # 検証: gap<=15 または top_score<=30 のレースに絞ると回収率約117%（全体は約56%）
    # 理由: モデルが迷う/確信度が低い荒れそうなレースほど高配当で期待値が出る
    # ─────────────────────────────────────────────
    VALUE_GAP_MAX = 15.0
    VALUE_SCORE_MAX = 30.0

    value_pickups = [
        r for r in today_records
        if r.get("top_score") is not None and r.get("score_gap") is not None
        and (r["score_gap"] <= VALUE_GAP_MAX or r["top_score"] <= VALUE_SCORE_MAX)
    ]

    st.markdown("### 💰 狙い目レース（回収率重視）")
    st.markdown(
        "<p style='color:#64748b;font-size:0.85rem;margin-bottom:1rem'>"
        "モデルの確信度が低め（1-2位差15以下 または 確信度30以下）の荒れそうなレース。"
        "過去データではこの条件に絞ると回収率が大きく改善しました。</p>",
        unsafe_allow_html=True,
    )
    if not value_pickups:
        st.info("本日、狙い目条件を満たすレースはまだありません。")
    else:
        from collections import defaultdict
        v_groups = defaultdict(list)
        for r in sorted(value_pickups, key=lambda r: r["score_gap"]):
            v_groups[r["venue_name"]].append(r)
        for venue_name, races in sorted(v_groups.items()):
            with st.expander(f"🏟 {venue_name}　({len(races)}件)", expanded=False):
                for r in races:
                    dl_info = st.session_state.deadline_times.get((r["venue_code"], r["race_no"]))
                    time_label = dl_info["deadline_time"] if dl_info else "--:--"
                    odds_val = r.get("odds_value")
                    odds_text = f"<span style='color:#0891b2;font-weight:700'>オッズ {odds_val:.1f}倍</span>" if odds_val else ""
                    card_html = (
                        "<div style='background:#ecfeff;border:2px solid #06b6d4;border-radius:10px;"
                        "padding:0.8rem 1.2rem;margin:0.3rem 0;display:flex;align-items:center;"
                        "gap:1rem;flex-wrap:wrap'>"
                        f"<span style='color:#155e75;font-size:0.85rem'>⏰ {time_label}</span>"
                        f"<span style='font-size:1rem;font-weight:800;color:#0891b2'>💰 {r['race_no']}R</span>"
                        f"<span style='color:#1d4ed8;font-size:0.9rem'>3連単 <strong>{' / '.join(r['sanren_tan'])}</strong></span>"
                        f"{odds_text}"
                        f"<span style='margin-left:auto;color:#475569;font-size:0.82rem'>"
                        f"確信度 <strong style='color:#0891b2'>{r['top_score']:.1f}</strong>"
                        f" / 差 <strong style='color:#0891b2'>{r['score_gap']:.1f}</strong>pt</span>"
                        "</div>"
                    )
                    st.markdown(card_html, unsafe_allow_html=True)

    st.markdown("---")

    pickups = [
        r for r in today_records
        if r.get("top_score") is not None and r.get("score_gap") is not None
        and r["top_score"] >= PICKUP_TOP_SCORE_MIN
        and r["score_gap"] >= PICKUP_SCORE_GAP_MIN
    ]

    # 穴狙いピックアップ（単勝1以外 かつ top_score >= 25）
    ana_pickups = [
        r for r in today_records
        if r.get("top_score") is not None and r.get("score_gap") is not None
        and r["top_score"] >= 25.0
        and r["score_gap"] >= 10.0
        and str(r.get("tansho", "1")) != "1"
    ]

    # 高配当狙いピックアップ（odds_value >= 20 かつ top_score >= 25）
    high_odds_pickups = [
        r for r in today_records
        if r.get("odds_value") is not None and r.get("top_score") is not None
        and r["odds_value"] >= 20.0
        and r["top_score"] >= 25.0
    ]

    # 穴狙いセクション
    st.markdown("### 🎯 穴狙い（単勝1以外）")
    st.markdown(
        "<p style='color:#64748b;font-size:0.85rem;margin-bottom:1rem'>"
        "モデルが1枠以外を本命予想したレース（確信度25以上・差10以上）</p>",
        unsafe_allow_html=True,
    )
    if not ana_pickups:
        st.info("本日、穴狙い条件を満たすレースはまだありません。")
    else:
        from collections import defaultdict
        ana_groups = defaultdict(list)
        for r in sorted(ana_pickups, key=lambda r: r["top_score"], reverse=True):
            ana_groups[r["venue_name"]].append(r)
        for venue_name, races in sorted(ana_groups.items()):
            with st.expander(f"🏟 {venue_name}　({len(races)}件)", expanded=False):
                for r in races:
                    dl_info = st.session_state.deadline_times.get((r["venue_code"], r["race_no"]))
                    time_label = dl_info["deadline_time"] if dl_info else "--:--"
                    odds_val = r.get("odds_value")
                    odds_text = f"<span style='color:#059669;font-weight:700'>オッズ {odds_val:.1f}倍</span>" if odds_val else ""
                    card_html = (
                        "<div style='background:#f0fdf4;border:2px solid #22c55e;border-radius:10px;"
                        "padding:0.8rem 1.2rem;margin:0.3rem 0;display:flex;align-items:center;"
                        "gap:1rem;flex-wrap:wrap'>"
                        f"<span style='color:#475569;font-size:0.85rem'>⏰ {time_label}</span>"
                        f"<span style='font-size:1rem;font-weight:800;color:#16a34a'>🎯 {r['race_no']}R</span>"
                        f"<span style='background:#dcfce7;border-radius:6px;padding:2px 10px;font-weight:800;color:#15803d;font-size:1rem'>{r['tansho']}枠</span>"
                        f"<span style='color:#1d4ed8;font-size:0.9rem'>3連単 <strong>{' / '.join(r['sanren_tan'])}</strong></span>"
                        f"{odds_text}"
                        f"<span style='margin-left:auto;color:#475569;font-size:0.82rem'>"
                        f"確信度 <strong style='color:#16a34a'>{r['top_score']:.1f}</strong>"
                        f" / 差 <strong style='color:#16a34a'>{r['score_gap']:.1f}</strong>pt</span>"
                        "</div>"
                    )
                    st.markdown(card_html, unsafe_allow_html=True)

    st.markdown("---")

    # 高配当狙いセクション
    st.markdown("### 💎 高配当狙い（オッズ20倍以上）")
    st.markdown(
        "<p style='color:#64748b;font-size:0.85rem;margin-bottom:1rem'>"
        "確信度25以上 かつ オッズ20倍以上の穴狙いレース</p>",
        unsafe_allow_html=True,
    )
    if not high_odds_pickups:
        st.info("本日、条件を満たす高配当レースはまだありません。")
    else:
        from collections import defaultdict
        ho_groups = defaultdict(list)
        for r in sorted(high_odds_pickups, key=lambda r: r.get("odds_value", 0), reverse=True):
            ho_groups[r["venue_name"]].append(r)
        for venue_name, races in sorted(ho_groups.items()):
            with st.expander(f"🏟 {venue_name}　({len(races)}件)", expanded=False):
                for r in races:
                    dl_info = st.session_state.deadline_times.get((r["venue_code"], r["race_no"]))
                    time_label = dl_info["deadline_time"] if dl_info else "--:--"
                    odds_val = r.get("odds_value", 0)
                    card_html = (
                        "<div style='background:#fdf4ff;border:2px solid #a855f7;border-radius:10px;"
                        "padding:0.8rem 1.2rem;margin:0.3rem 0;display:flex;align-items:center;"
                        "gap:1rem;flex-wrap:wrap'>"
                        f"<span style='color:#6b21a8;font-size:0.85rem'>⏰ {time_label}</span>"
                        f"<span style='font-size:1rem;font-weight:800;color:#a855f7'>💎 {r['race_no']}R</span>"
                        f"<span style='color:#1d4ed8;font-size:0.9rem'>単勝 <strong>{r['tansho']}</strong></span>"
                        f"<span style='color:#1d4ed8;font-size:0.9rem'>3連単 <strong>{' / '.join(r['sanren_tan'])}</strong></span>"
                        f"<span style='color:#7e22ce;font-weight:700'>オッズ {odds_val:.1f}倍</span>"
                        f"<span style='margin-left:auto;color:#64748b;font-size:0.82rem'>"
                        f"確信度 <strong style='color:#a855f7'>{r['top_score']:.1f}</strong>pt</span>"
                        "</div>"
                    )
                    st.markdown(card_html, unsafe_allow_html=True)

    st.markdown("---")

    # 通常ピックアップセクション
    st.markdown("### 🔥 通常ピックアップ")
    st.markdown(
        f"<p style='color:#64748b;font-size:0.85rem;margin-bottom:1rem'>"
        f"1位確信度 {PICKUP_TOP_SCORE_MIN:.0f}%以上 かつ 2位との差 {PICKUP_SCORE_GAP_MIN:.0f}pt以上</p>",
        unsafe_allow_html=True,
    )

    if not today_records:
        st.info("本日の予想データがまだありません。08:00のバッチを待つか、予想タブで取得してください。")
    elif not pickups:
        st.info("本日、条件を満たすレースはまだありません。")
    else:
        # 会場ごとにグループ化（race_no順）
        from collections import defaultdict
        venue_groups = defaultdict(list)
        for r in sorted(pickups, key=lambda r: r["race_no"]):
            venue_groups[r["venue_name"]].append(r)

        for venue_name, races in sorted(venue_groups.items()):
            with st.expander(f"🏟 {venue_name}　({len(races)}件)", expanded=False):
                for r in races:
                    dl_info = st.session_state.deadline_times.get((r["venue_code"], r["race_no"]))
                    time_label = dl_info["deadline_time"] if dl_info else "--:--"
                    odds_val = r.get("odds_value")
                    odds_text = f"<span style='color:#475569;font-size:0.9rem'>オッズ <strong style='color:#059669'>{odds_val:.1f}倍</strong></span>" if odds_val else ""
                    value_badge = ""
                    if odds_val and odds_val >= 5.0:
                        value_badge = "<span style='background:#d1fae5;border:1px solid #10b981;color:#065f46;border-radius:6px;padding:2px 8px;font-size:0.75rem;font-weight:700'>💎 妙味</span>"
                    weather_text2 = ""
                    if r.get("wind_speed") or r.get("wave_height"):
                        parts = []
                        if r.get("wind_speed"): parts.append(f"風{r['wind_speed']}m")
                        if r.get("wave_height"): parts.append(f"波{r['wave_height']}cm")
                        weather_text2 = f"<span style='color:#475569;font-size:0.82rem'>{'　'.join(parts)}</span>"
                    card_html = (
                        "<div style='background:#fffbeb;border:2px solid #f59e0b;border-radius:10px;"
                        "padding:0.8rem 1.2rem;margin:0.3rem 0;display:flex;align-items:center;"
                        "gap:1rem;flex-wrap:wrap'>"
                        f"<span style='color:#92400e;font-size:0.85rem'>⏰ {time_label}</span>"
                        f"<span style='font-size:1rem;font-weight:800;color:#d97706'>🔥 {r['race_no']}R</span>"
                        f"<span style='color:#1d4ed8;font-size:0.9rem'>単勝 <strong>{r['tansho']}</strong></span>"
                        f"<span style='color:#1d4ed8;font-size:0.9rem'>3連単 <strong>{' / '.join(r['sanren_tan'])}</strong></span>"
                        f"{odds_text}{value_badge}{weather_text2}"
                        f"<span style='margin-left:auto;color:#475569;font-size:0.82rem'>"
                        f"確信度 <strong style='color:#d97706'>{r['top_score']:.1f}</strong>"
                        f" / 差 <strong style='color:#d97706'>{r['score_gap']:.1f}</strong>pt</span>"
                        "</div>"
                    )
                    st.markdown(card_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# タブ1: 予想
# ─────────────────────────────────────────────
if page == "🎯 予想":
    # 出走表データが無ければ手動取得ボタンを表示（バッチが08:00に自動取得）
    if not st.session_state.today_racers:
        if st.button("📥  出走表を取得（通常はバッチが自動取得します）", type="secondary", use_container_width=True):
            sc = get_scraper()
            if sc:
                from crawler import get_holding_venues
                from boatrace_scraper import get_deadline_times
                with st.spinner("開催会場を確認中..."):
                    venues = get_holding_venues(sc, today_jst())
                if not venues:
                    st.warning("本日の開催会場が見つかりませんでした")
                else:
                    # 締切時刻取得
                    deadlines_by_venue = {}
                    for venue in venues:
                        try:
                            deadlines_by_venue[venue] = get_deadline_times(sc, today_jst(), venue)
                        except Exception:
                            deadlines_by_venue[venue] = {}

                    progress = st.progress(0, text="取得中...")
                    today_racers = {}
                    total = len(venues) * 12
                    count = 0
                    for venue in venues:
                        today_racers[venue] = {}
                        for rno in range(1, 13):
                            racers = sc.get_racelist(today_jst(), venue, rno)
                            if racers:
                                today_racers[venue][rno] = racers
                            count += 1
                            progress.progress(count / total, text=f"{VENUE_MAP[venue]} {rno}R 取得中...")
                    st.session_state.today_racers = today_racers
                    save_cache(today_racers)

                    # Supabaseにも保存
                    if supabase:
                        from dataclasses import asdict
                        for venue, races in today_racers.items():
                            for rno, racers in races.items():
                                deadline_dt = deadlines_by_venue.get(venue, {}).get(rno)
                                deadline_str = deadline_dt.strftime("%H:%M") if deadline_dt else None
                                try:
                                    supabase.table("today_racelist").upsert({
                                        "race_date": today_jst().strftime("%Y%m%d"),
                                        "venue_code": venue,
                                        "venue_name": VENUE_MAP[venue],
                                        "race_no": rno,
                                        "racers": json.dumps([asdict(r) for r in racers], ensure_ascii=False),
                                        "deadline_time": deadline_str,
                                    }, on_conflict="race_date,venue_code,race_no").execute()
                                except Exception as e:
                                    st.warning(f"出走表Supabase保存失敗: {e}")

                    st.session_state.deadline_times = load_deadline_times_from_supabase()
                    total_races = sum(len(v) for v in today_racers.values())
                    st.success(f"✅  {len(venues)}会場 {total_races}レース取得完了")
                    st.rerun()

    if st.session_state.today_racers:
        # 締切時刻順のレース一覧
        from datetime import datetime as _dt
        now_time = _dt.now(JST).strftime("%H:%M")

        deadline_times = st.session_state.deadline_times
        upcoming_list = []
        past_list = []
        for venue, races in st.session_state.today_racers.items():
            for rno in races.keys():
                info = deadline_times.get((venue, rno))
                deadline_str = info["deadline_time"] if info else None
                is_past = deadline_str is not None and deadline_str < now_time
                if is_past:
                    past_list.append((deadline_str, venue, rno))
                else:
                    upcoming_list.append((deadline_str, venue, rno))

        upcoming_sorted = sorted(
            [r for r in upcoming_list if r[0]], key=lambda r: r[0]
        ) + sorted([r for r in upcoming_list if not r[0]], key=lambda r: (r[1], r[2]))

        past_sorted = sorted(
            [r for r in past_list if r[0]], key=lambda r: r[0], reverse=True
        ) + sorted([r for r in past_list if not r[0]], key=lambda r: (r[1], r[2]))

        st.markdown("#### 本日のレース一覧（出走時刻順）")

        def _show_race_detail(venue, rno):
            """選択レースの出走表・予想ボタン・予想結果をインライン表示"""
            racers = st.session_state.today_racers[venue][rno]
            st.markdown(
                "<div style='background:#f8fafc;border:1px solid #1d4ed8;border-radius:10px;"
                "padding:1rem 1.2rem;margin:0.3rem 0 0.8rem'>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**{VENUE_MAP[venue]} {rno}R 出走表**")
            for r in racers:
                st.markdown(f"""
                <div class="racer-row" style="margin:2px 0;padding:0.5rem 0.8rem">
                    <div class="lane-badge lane-{r.lane}">{r.lane}</div>
                    <div style="flex:1">
                        <span style="font-weight:700;color:#1e293b">{r.name}</span>
                        <span style="margin-left:8px;font-size:0.8rem;color:#475569">{r.rank}</span>
                    </div>
                    <div style="text-align:right">
                        <span style="font-size:0.85rem;color:#475569">勝率 </span>
                        <span style="font-weight:700;color:#3b82f6">{r.win_rate_all or '-'}</span>
                    </div>
                    <div style="text-align:right;min-width:80px">
                        <span style="font-size:0.85rem;color:#475569">モーター </span>
                        <span style="font-weight:600;color:#475569">{r.motor_2rate or '-'}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            if st.button("🎯 このレースを予想する", type="primary",
                         use_container_width=True, key=f"predict_{venue}_{rno}"):
                predictor = get_predictor()
                pred = predictor.predict(racers, race_date=today_jst().strftime("%Y%m%d"),
                                          venue_name=VENUE_MAP[venue], race_no=rno)
                st.session_state.prediction = pred
                st.session_state.last_venue = venue
                st.session_state.last_race = rno
                top_score, score_gap = _score_metrics(pred)
                save_record({
                    "race_date": today_jst().strftime("%Y%m%d"),
                    "venue_code": venue, "venue_name": VENUE_MAP[venue], "race_no": rno,
                    "tansho": pred.tansho, "sanren_tan": pred.sanren_tan,
                    "sanren_fuku": pred.sanren_fuku, "hit": None, "actual": "", "payout": None,
                    "top_score": top_score, "score_gap": score_gap,
                })

            # 予想結果もインライン表示
            pred = st.session_state.get("prediction")
            if pred and st.session_state.get("last_venue") == venue and st.session_state.get("last_race") == rno:
                sorted_scores = sorted(pred.scores, key=lambda s: s.predicted_rank)
                medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣"]
                st.markdown("---")
                st.markdown("### 🏆 予想結果")

                fig = go.Figure(go.Bar(
                    x=[s.total_score for s in sorted_scores],
                    y=[f"{s.lane}枠 {s.name}" for s in sorted_scores],
                    orientation='h',
                    marker_color=['#f59e0b','#94a3b8','#b45309','#475569','#475569','#475569'],
                    text=[f"{s.total_score:.1f}" for s in sorted_scores],
                    textposition='outside',
                ))
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#94a3b8', height=220,
                    margin=dict(l=10, r=60, t=10, b=10),
                    xaxis=dict(gridcolor='#1e293b', showgrid=True),
                    yaxis=dict(autorange='reversed'),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"""
                    <div class="buy-box">
                        <p class="buy-title">💰 推奨買い目</p>
                        <p style="color:#475569;font-size:0.85rem;margin:0">単勝</p>
                        <div class="buy-combo">{pred.tansho}</div>
                        <p style="color:#475569;font-size:0.85rem;margin:0.8rem 0 0">3連単</p>
                        {''.join([f'<div class="buy-combo">{c}</div>' for c in pred.sanren_tan])}
                        <p style="color:#475569;font-size:0.85rem;margin:0.8rem 0 0">3連複</p>
                        <div class="buy-combo">{pred.sanren_fuku}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_b:
                    st.markdown(f"""
                    <div class="buy-box">
                        <p class="buy-title">📊 予想順位</p>
                        {''.join([f'<p style="margin:0.4rem 0;color:#1e293b">{medals[s.predicted_rank-1]} {s.lane}枠 {s.name} <span style="color:#3b82f6;font-weight:700">{s.total_score:.1f}pt</span></p>' for s in sorted_scores])}
                    </div>
                    """, unsafe_allow_html=True)

        if upcoming_sorted:
            for deadline_str, venue, rno in upcoming_sorted:
                time_label = deadline_str if deadline_str else "--:--"
                is_selected = st.session_state.selected_race == (venue, rno)
                label = f"⏰ {time_label}　{VENUE_MAP[venue]} {rno}R"
                if st.button(label, key=f"race_select_{venue}_{rno}",
                              type="primary" if is_selected else "secondary",
                              use_container_width=True):
                    if is_selected:
                        st.session_state.selected_race = None
                        st.session_state.prediction = None
                    else:
                        st.session_state.selected_race = (venue, rno)
                        st.session_state.prediction = None
                    st.rerun()
                if is_selected:
                    _show_race_detail(venue, rno)
        elif past_sorted:
            st.info("本日のレースは全て終了しました。")
            with st.expander(f"終了済みレース ({len(past_sorted)}件)", expanded=False):
                for deadline_str, venue, rno in past_sorted:
                    time_label = deadline_str if deadline_str else "--:--"
                    is_selected = st.session_state.selected_race == (venue, rno)
                    label = f"✅ {time_label}　{VENUE_MAP[venue]} {rno}R"
                    if st.button(label, key=f"race_select_{venue}_{rno}",
                                  type="primary" if is_selected else "secondary",
                                  use_container_width=True):
                        if is_selected:
                            st.session_state.selected_race = None
                            st.session_state.prediction = None
                        else:
                            st.session_state.selected_race = (venue, rno)
                            st.session_state.prediction = None
                        st.rerun()
                    if is_selected:
                        _show_race_detail(venue, rno)
        else:
            st.info("本日の出走表データがまだありません。08:00のバッチを待つか、上のボタンで取得してください。")

    else:
        st.info("本日の出走表データがまだありません。08:00のバッチを待つか、上のボタンで取得してください。")

# ─────────────────────────────────────────────
# タブ2: 直前情報
# ─────────────────────────────────────────────
if page == "📊 直前情報":
    col1, col2 = st.columns(2)
    with col1:
        venue_options2 = {v: k for k, v in VENUE_MAP.items()}
        vn2 = st.selectbox("会場", list(venue_options2.keys()), key="venue2",
                            index=list(venue_options2.keys()).index("住之江"))
        vc2 = venue_options2[vn2]
    with col2:
        rno2 = st.selectbox("レース", list(range(1, 13)), key="race2", format_func=lambda x: f"{x}R")

    if st.button("📥  直前情報を取得", type="primary", key="before_btn", use_container_width=True):
        before_scraper = BeforeInfoScraper(delay=1.0)
        with st.spinner("取得中..."):
            info = before_scraper.get_before_info(today_jst(), vc2, rno2)
            if info:
                st.session_state.before_info = info
                save_fetch_history("before_info", today_jst().strftime("%Y%m%d"), vc2, VENUE_MAP[vc2], rno2)
                st.success("✅  取得しました")
            else:
                st.warning("直前情報がまだ公開されていません")

    if st.session_state.before_info:
        info = st.session_state.before_info

        # 気象情報
        if info.weather:
            w = info.weather
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("🌡 気温", f"{w.temperature}℃" if w.temperature else "-")
            with c2:
                st.metric("💨 風速", f"{w.wind_speed}m" if w.wind_speed else "-")
            with c3:
                st.metric("🌊 波高", f"{w.wave_height}cm" if w.wave_height else "-")
            with c4:
                st.metric("💧 水温", f"{w.water_temp}℃" if w.water_temp else "-")

        # 展示タイム（バーチャート）
        if info.exhibitions:
            st.markdown("#### 展示タイム")
            times = [(e.lane, e.name or f"{e.lane}枠", e.exhibition_time) for e in info.exhibitions if e.exhibition_time]
            if times:
                fig2 = go.Figure(go.Bar(
                    x=[f"{t[0]}枠 {t[1]}" for t in times],
                    y=[t[2] for t in times],
                    marker_color=['#f59e0b' if t[2] == min(t[2] for t in times) else '#1d4ed8' for t in times],
                    text=[str(t[2]) for t in times],
                    textposition='outside',
                ))
                fig2.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#94a3b8', height=200,
                    margin=dict(l=10, r=10, t=10, b=10),
                    yaxis=dict(gridcolor='#1e293b', range=[6.5, 7.5]),
                    showlegend=False,
                )
                st.plotly_chart(fig2, use_container_width=True)
                st.caption("🟡 最速タイム")

        # スタート展示
        if info.start_exhibition:
            st.markdown("#### スタート展示ST")
            for s in info.start_exhibition:
                st.markdown(f"""
                <div class="racer-row">
                    <div class="lane-badge lane-{s.boat_no}">{s.boat_no}</div>
                    <span style="color:#475569">{s.course}コース</span>
                    <span style="font-weight:700;color:#3b82f6;margin-left:auto">ST: {s.st or '-'}</span>
                </div>
                """, unsafe_allow_html=True)

    # 取得履歴
    st.markdown("---")
    st.markdown("#### 📜 取得履歴")
    history = load_fetch_history("before_info", limit=10)
    if not history:
        st.markdown("<span style='color:#475569;font-size:0.85rem'>まだ取得履歴がありません</span>", unsafe_allow_html=True)
    else:
        for h in history:
            ds = h["race_date"]
            formatted = f"{ds[:4]}/{ds[4:6]}/{ds[6:]}"
            fetched_at = h.get("fetched_at", "")
            try:
                from datetime import datetime as _dt2, timezone, timedelta as _td2
                _jst = timezone(_td2(hours=9))
                _dt_utc = _dt2.fromisoformat(fetched_at.replace("Z", "+00:00"))
                time_str = _dt_utc.astimezone(_jst).strftime("%H:%M")
            except Exception:
                time_str = fetched_at[11:16] if len(fetched_at) >= 16 else ""
            race_date_obj = date(int(ds[:4]), int(ds[4:6]), int(ds[6:]))
            with st.expander(f"📊 {formatted} {time_str}　{h['venue_name']} {h['race_no']}R"):
                sc2 = get_scraper()
                if sc2:
                    before_scraper2 = BeforeInfoScraper(delay=1.0)
                    with st.spinner("取得中..."):
                        info2 = before_scraper2.get_before_info(race_date_obj, h["venue_code"], h["race_no"])
                    if info2:
                        if info2.weather:
                            w = info2.weather
                            c1, c2, c3, c4 = st.columns(4)
                            with c1: st.metric("🌡 気温", f"{w.temperature}℃" if w.temperature else "-")
                            with c2: st.metric("💨 風速", f"{w.wind_speed}m" if w.wind_speed else "-")
                            with c3: st.metric("🌊 波高", f"{w.wave_height}cm" if w.wave_height else "-")
                            with c4: st.metric("💧 水温", f"{w.water_temp}℃" if w.water_temp else "-")
                        if info2.exhibitions:
                            st.markdown("**展示タイム**")
                            for e in info2.exhibitions:
                                if e.exhibition_time:
                                    st.markdown(f"<div class='racer-row'><div class='lane-badge lane-{e.lane}'>{e.lane}</div><span style='color:#1e293b'>{e.name or ''}</span><span style='margin-left:auto;font-weight:700;color:#3b82f6'>{e.exhibition_time}</span></div>", unsafe_allow_html=True)
                    else:
                        st.caption("データが取得できませんでした")

# ─────────────────────────────────────────────
# タブ3: 結果確認
# ─────────────────────────────────────────────
if page == "📋 結果確認":
    col1, col2 = st.columns(2)
    with col1:
        venue_options3 = {v: k for k, v in VENUE_MAP.items()}
        vn3 = st.selectbox("会場", list(venue_options3.keys()), key="venue3",
                            index=list(venue_options3.keys()).index("住之江"))
        vc3 = venue_options3[vn3]
    with col2:
        rno3 = st.selectbox("レース", list(range(1, 13)), key="race3", format_func=lambda x: f"{x}R")

    if st.button("📥  結果を取得して記録を更新", type="primary", key="result_btn", use_container_width=True):
        sc = get_scraper()
        if sc:
            with st.spinner("取得中..."):
                result = sc.get_result(today_jst(), vc3, rno3)
                if result and result.arrival:
                    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣"]
                    col_r1, col_r2 = st.columns(2)
                    with col_r1:
                        st.markdown("#### 着順")
                        for i, boat in enumerate(result.arrival):
                            st.markdown(f"<p style='margin:0.3rem 0;color:#1e293b'>{medals[i]} <strong>{boat}号艇</strong></p>", unsafe_allow_html=True)
                    with col_r2:
                        if result.payouts:
                            st.markdown("#### 払戻金")
                            for key, val in result.payouts.items():
                                if val:
                                    st.markdown(f"<p style='margin:0.2rem 0;color:#475569'>{key}: <strong style='color:#3b82f6'>¥{val:,}</strong></p>", unsafe_allow_html=True)

                    # 記録更新
                    records = load_records()
                    date_str = today_jst().strftime("%Y%m%d")
                    save_fetch_history("result", date_str, vc3, VENUE_MAP[vc3], rno3)
                    for r in records:
                        if r["race_date"] == date_str and r["venue_code"] == vc3 and r["race_no"] == rno3:
                            hit, actual = check_hit(r["sanren_tan"], result.arrival)
                            r["hit"] = hit
                            r["actual"] = actual
                            # 払戻は的中時のみ。外れは0（自分の買い目が当たった場合だけ配当を記録）
                            r["payout"] = result.payouts.get(f"3連単_{actual}", 0) if hit else 0
                            if hit:
                                st.success(f"🎉 的中！{actual}  ¥{r['payout']:,}" if r['payout'] else "🎉 的中！")
                            else:
                                st.error(f"❌ ハズレ  実際: {actual}  予想: {' / '.join(r['sanren_tan'])}")

                            # Supabase更新
                            if supabase:
                                try:
                                    supabase.table("prediction_records").update({
                                        "hit": r["hit"], "actual": r["actual"], "payout": r["payout"],
                                    }).eq("race_date", r["race_date"]).eq("venue_code", r["venue_code"]).eq("race_no", r["race_no"]).execute()
                                except Exception as e:
                                    st.warning(f"Supabase更新失敗: {e}")
                            else:
                                RECORD_FILE.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
                else:
                    st.warning("結果がまだ出ていません")

    # 取得履歴
    st.markdown("---")
    st.markdown("#### 📜 取得履歴")
    history3 = load_fetch_history("result", limit=10)
    if not history3:
        st.markdown("<span style='color:#475569;font-size:0.85rem'>まだ取得履歴がありません</span>", unsafe_allow_html=True)
    else:
        all_records = load_records()
        for h in history3:
            ds = h["race_date"]
            formatted = f"{ds[:4]}/{ds[4:6]}/{ds[6:]}"
            fetched_at = h.get("fetched_at", "")
            try:
                from datetime import datetime as _dt2, timezone, timedelta as _td2
                _jst = timezone(_td2(hours=9))
                _dt_utc = _dt2.fromisoformat(fetched_at.replace("Z", "+00:00"))
                time_str = _dt_utc.astimezone(_jst).strftime("%H:%M")
            except Exception:
                time_str = fetched_at[11:16] if len(fetched_at) >= 16 else ""
            with st.expander(f"📋 {formatted} {time_str}　{h['venue_name']} {h['race_no']}R"):
                matched = [r for r in all_records
                           if r["race_date"] == h["race_date"]
                           and r["venue_code"] == h["venue_code"]
                           and r["race_no"] == h["race_no"]]
                if matched:
                    r = matched[0]
                    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣"]
                    if r.get("actual"):
                        parts = r["actual"].split("-")
                        for i, p in enumerate(parts[:3]):
                            st.markdown(f"<p style='margin:0.2rem 0;color:#1e293b'>{medals[i]} <strong>{p}号艇</strong></p>", unsafe_allow_html=True)
                    if r.get("payout"):
                        st.markdown(f"<p style='color:#3b82f6;font-weight:700;margin-top:0.5rem'>払戻: ¥{r['payout']:,}</p>", unsafe_allow_html=True)
                    if r.get("hit") is True:
                        st.success("🎉 的中！")
                    elif r.get("hit") is False:
                        st.error(f"❌ ハズレ　予想: {' / '.join(r['sanren_tan'])}")
                    else:
                        st.caption("未確認")
                else:
                    st.caption("予想記録が見つかりません")

# ─────────────────────────────────────────────
# タブ4: 成績記録
# ─────────────────────────────────────────────
if page == "📈 成績記録":
    records = load_records()

    if not records:
        st.info("まだ予想記録がありません。予想タブで予想すると自動で記録されます。")
    else:
        # ─────────────────────────────────────────────
        # 💰 狙い目レースの成績（回収率検証）
        # ─────────────────────────────────────────────
        VALUE_GAP_MAX = 15.0
        VALUE_SCORE_MAX = 30.0
        value_checked = [
            r for r in records
            if r.get("hit") is not None
            and r.get("top_score") is not None and r.get("score_gap") is not None
            and (r["score_gap"] <= VALUE_GAP_MAX or r["top_score"] <= VALUE_SCORE_MAX)
        ]
        if value_checked:
            v_hits = [r for r in value_checked if r["hit"]]
            v_payout = sum(r.get("payout") or 0 for r in v_hits)
            v_cost = len(value_checked) * 300
            v_roi = v_payout / v_cost * 100 if v_cost else 0
            v_hit_rate = len(v_hits) / len(value_checked) * 100 if value_checked else 0

            # 全体の回収率（比較用）
            all_checked = [r for r in records if r.get("hit") is not None]
            all_hits = [r for r in all_checked if r["hit"]]
            all_payout = sum(r.get("payout") or 0 for r in all_hits)
            all_cost = len(all_checked) * 300
            all_roi = all_payout / all_cost * 100 if all_cost else 0

            roi_color = "#4ade80" if v_roi >= 100 else ("#f5c542" if v_roi >= 75 else "#ef4444")

            st.markdown("### 💰 狙い目レースの成績")
            st.markdown(
                "<p style='color:#8b9bb4;font-size:0.85rem;margin-bottom:1rem'>"
                "確信度が低め（1-2位差15以下 または 確信度30以下）の荒れそうなレースだけを買った場合の成績。"
                "全体より回収率が高ければ、狙い目フィルタが機能している証拠です。</p>",
                unsafe_allow_html=True,
            )
            vc1, vc2, vc3, vc4 = st.columns(4)
            with vc1:
                st.markdown(f'<div class="stat-card"><div class="stat-value">{len(value_checked)}</div><div class="stat-label">狙い目レース数</div></div>', unsafe_allow_html=True)
            with vc2:
                st.markdown(f'<div class="stat-card"><div class="stat-value">{len(v_hits)}</div><div class="stat-label">的中数（{v_hit_rate:.1f}%）</div></div>', unsafe_allow_html=True)
            with vc3:
                st.markdown(f'<div class="stat-card"><div class="stat-value" style="color:{roi_color}">{v_roi:.1f}%</div><div class="stat-label">狙い目の回収率</div></div>', unsafe_allow_html=True)
            with vc4:
                st.markdown(f'<div class="stat-card"><div class="stat-value" style="color:#8b9bb4">{all_roi:.1f}%</div><div class="stat-label">全体の回収率</div></div>', unsafe_allow_html=True)

            st.markdown(
                "<p style='color:#64748b;font-size:0.78rem;margin-top:0.5rem'>"
                "※ まだサンプルが少ないため、数本の高配当で数字が大きく動きます。継続して検証が必要です。</p>",
                unsafe_allow_html=True,
            )
            st.markdown("---")
        checked = [r for r in records if r["hit"] is not None]
        hits = [r for r in checked if r["hit"]]
        hit_rate = len(hits) / len(checked) * 100 if checked else 0
        total_payout = sum(r["payout"] or 0 for r in hits)
        total_cost = len(checked) * 300  # 1レース300円（3連単3点）
        profit = total_payout - total_cost
        profit_color = "#10b981" if profit >= 0 else "#ef4444"
        profit_sign = "+" if profit >= 0 else ""

        # サマリーカード
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            st.markdown(f'<div class="stat-card"><div class="stat-value">{len(records)}</div><div class="stat-label">総予想数</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-card"><div class="stat-value">{len(hits)}</div><div class="stat-label">的中数</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="stat-card"><div class="stat-value">{hit_rate:.1f}%</div><div class="stat-label">的中率</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="stat-card"><div class="stat-value">¥{total_payout:,}</div><div class="stat-label">総払戻金</div></div>', unsafe_allow_html=True)
        with c5:
            st.markdown(f'<div class="stat-card"><div class="stat-value">¥{total_cost:,}</div><div class="stat-label">総投資額</div></div>', unsafe_allow_html=True)
        with c6:
            st.markdown(f'<div class="stat-card"><div class="stat-value" style="color:{profit_color}">{profit_sign}¥{profit:,}</div><div class="stat-label">収支</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 的中率推移グラフ
        if len(checked) >= 2:
            cumulative_hits = []
            cum = 0
            for i, r in enumerate(checked):
                if r["hit"]:
                    cum += 1
                cumulative_hits.append(cum / (i + 1) * 100)

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                y=cumulative_hits,
                mode='lines+markers',
                line=dict(color='#3b82f6', width=2),
                marker=dict(color='#60a5fa', size=6),
                fill='tozeroy',
                fillcolor='rgba(59,130,246,0.1)',
            ))
            fig3.add_hline(y=hit_rate, line_dash="dash", line_color="#f59e0b",
                           annotation_text=f"平均 {hit_rate:.1f}%")
            fig3.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='#94a3b8', height=200,
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis=dict(gridcolor='#1e293b', title="的中率(%)"),
                xaxis=dict(gridcolor='#1e293b', title="レース数"),
                showlegend=False,
            )
            st.markdown("#### 的中率推移")
            st.plotly_chart(fig3, use_container_width=True)

        # 会場別・レース番号別 集計
        if len(checked) >= 5:
            st.markdown("#### 会場別・レース別 的中率")
            col_v, col_r = st.columns(2)

            with col_v:
                venue_stats = {}
                for r in checked:
                    vn = r["venue_name"]
                    venue_stats.setdefault(vn, {"hit": 0, "total": 0})
                    venue_stats[vn]["total"] += 1
                    if r["hit"]:
                        venue_stats[vn]["hit"] += 1

                venue_names = list(venue_stats.keys())
                venue_rates = [venue_stats[v]["hit"] / venue_stats[v]["total"] * 100 for v in venue_names]
                venue_totals = [venue_stats[v]["total"] for v in venue_names]

                order = sorted(range(len(venue_names)), key=lambda i: venue_rates[i], reverse=True)
                venue_names = [venue_names[i] for i in order]
                venue_rates = [venue_rates[i] for i in order]
                venue_totals = [venue_totals[i] for i in order]

                fig_venue = go.Figure(go.Bar(
                    x=venue_rates,
                    y=[f"{n} ({t}件)" for n, t in zip(venue_names, venue_totals)],
                    orientation='h',
                    marker_color='#3b82f6',
                    text=[f"{r:.1f}%" for r in venue_rates],
                    textposition='outside',
                ))
                fig_venue.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#94a3b8', height=max(200, 30 * len(venue_names)),
                    margin=dict(l=10, r=40, t=30, b=10),
                    xaxis=dict(gridcolor='#1e293b', title="的中率(%)", range=[0, 100]),
                    yaxis=dict(autorange='reversed'),
                    showlegend=False,
                    title=dict(text="会場別", font=dict(size=12)),
                )
                st.plotly_chart(fig_venue, use_container_width=True)

            with col_r:
                race_stats = {}
                for r in checked:
                    rn = r["race_no"]
                    race_stats.setdefault(rn, {"hit": 0, "total": 0})
                    race_stats[rn]["total"] += 1
                    if r["hit"]:
                        race_stats[rn]["hit"] += 1

                race_nos = sorted(race_stats.keys())
                race_rates = [race_stats[rn]["hit"] / race_stats[rn]["total"] * 100 for rn in race_nos]
                race_totals = [race_stats[rn]["total"] for rn in race_nos]

                fig_race = go.Figure(go.Bar(
                    x=[f"{rn}R" for rn in race_nos],
                    y=race_rates,
                    marker_color='#60a5fa',
                    text=[f"{r:.0f}%" for r in race_rates],
                    textposition='outside',
                ))
                fig_race.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#94a3b8', height=240,
                    margin=dict(l=10, r=10, t=30, b=10),
                    yaxis=dict(gridcolor='#1e293b', title="的中率(%)", range=[0, 100]),
                    showlegend=False,
                    title=dict(text="レース番号別", font=dict(size=12)),
                )
                st.plotly_chart(fig_race, use_container_width=True)

        # 記録一覧
        st.markdown("#### 記録一覧")
        for r in sorted(records, reverse=True, key=lambda x: (x["race_date"], x["venue_code"], x["race_no"])):
            if r["hit"] is True:
                badge = "<span style='background:#065f46;border:1px solid #10b981;color:#6ee7b7;border-radius:6px;padding:2px 10px;font-size:0.82rem;font-weight:700'>🎉 的中</span>"
            elif r["hit"] is False:
                badge = "<span style='background:#7f1d1d;border:1px solid #ef4444;color:#fca5a5;border-radius:6px;padding:2px 10px;font-size:0.82rem;font-weight:700'>❌ ハズレ</span>"
            else:
                badge = "<span style='background:#292524;border:1px solid #57534e;color:#d6d3d1;border-radius:6px;padding:2px 10px;font-size:0.82rem;font-weight:700'>⏳ 未確認</span>"

            ds = r["race_date"]
            formatted = f"{ds[:4]}/{ds[4:6]}/{ds[6:]}"
            actual_text = f"実際: <strong>{r['actual']}</strong>" if r["actual"] else ""
            payout_text = f"払戻: <strong style='color:#3b82f6'>¥{r['payout']:,}</strong>" if r.get("payout") else ""

            extra1 = f"<span style='color:#475569;font-size:0.85rem'>{actual_text}</span>" if actual_text else ""
            extra2 = f"<span style='font-size:0.85rem'>{payout_text}</span>" if payout_text else ""

            border_color = "#f59e0b" if r["hit"] is True else "#1e3a8a"
            border_width = "2px" if r["hit"] is True else "1px"

            card_html = (
                f"<div style='background:#f8fafc;border:{border_width} solid {border_color};border-radius:10px;"
                "padding:0.8rem 1rem;margin:0.4rem 0;display:flex;align-items:center;"
                "gap:1rem;flex-wrap:wrap'>"
                f"<span style='color:#475569;font-size:0.85rem'>{formatted}</span>"
                f"<span style='font-weight:700;color:#1e293b'>{r['venue_name']} {r['race_no']}R</span>"
                f"<span style='color:#3b82f6;font-size:0.85rem'>{' / '.join(r['sanren_tan'])}</span>"
                f"{extra1}{extra2}"
                f"<span style='margin-left:auto'>{badge}</span>"
                "</div>"
            )
            st.markdown(card_html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑  記録をリセット", type="secondary"):
            if supabase:
                try:
                    supabase.table("prediction_records").delete().neq("id", 0).execute()
                except Exception as e:
                    st.warning(f"Supabase削除失敗: {e}")
            RECORD_FILE.unlink(missing_ok=True)
            st.success("リセットしました")
            st.rerun()

            # ─────────────────────────────────────────────
# タブ5: 高配当殿堂
# ─────────────────────────────────────────────
if page == "💎 高配当殿堂":
    st.markdown("### 💎 高配当殿堂")
    st.markdown(
        "<p style='color:#64748b;font-size:0.85rem;margin-bottom:1rem'>"
        "払戻1万円以上の的中レース。狙い目が大きく当たった記録です。</p>",
        unsafe_allow_html=True,
    )

    records = load_records()
    big_hits = [
        r for r in records
        if r.get("hit") is True and r.get("payout") and r["payout"] >= 10000
    ]
    big_hits.sort(key=lambda r: r["payout"], reverse=True)

    if not big_hits:
        st.info("まだ1万円以上の的中はありません。")
    else:
        # サマリー
        total = sum(r["payout"] for r in big_hits)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="stat-card"><div class="stat-value">{len(big_hits)}</div><div class="stat-label">殿堂入り</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-card"><div class="stat-value">¥{big_hits[0]["payout"]:,}</div><div class="stat-label">最高配当</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="stat-card"><div class="stat-value">¥{total:,}</div><div class="stat-label">合計払戻</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        for rank, r in enumerate(big_hits, start=1):
            p = r["payout"]
            # 配当帯で色とバッジを変える
            if p >= 50000:
                bg, border, badge = "#fef3c7", "#f59e0b", "👑 超高配当"
            elif p >= 30000:
                bg, border, badge = "#fae8ff", "#a855f7", "💎 高配当"
            else:
                bg, border, badge = "#ecfeff", "#06b6d4", "✨ 万舟"
            ds = r["race_date"]
            formatted = f"{ds[:4]}/{ds[4:6]}/{ds[6:]}"
            card_html = (
                f"<div style='background:{bg};border:2px solid {border};border-radius:10px;"
                "padding:0.9rem 1.2rem;margin:0.4rem 0;display:flex;align-items:center;"
                "gap:1rem;flex-wrap:wrap'>"
                f"<span style='font-size:1.2rem;font-weight:800;color:{border};min-width:36px'>#{rank}</span>"
                f"<span style='background:{border};color:#fff;border-radius:6px;padding:2px 10px;font-size:0.78rem;font-weight:700'>{badge}</span>"
                f"<span style='color:#475569;font-size:0.85rem'>{formatted}</span>"
                f"<span style='font-weight:700;color:#1e293b'>{r['venue_name']} {r['race_no']}R</span>"
                f"<span style='color:#1d4ed8;font-size:0.9rem'>{r.get('actual','')}</span>"
                f"<span style='margin-left:auto;font-size:1.3rem;font-weight:800;color:{border}'>¥{p:,}</span>"
                "</div>"
            )
            st.markdown(card_html, unsafe_allow_html=True)