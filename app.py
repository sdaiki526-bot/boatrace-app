"""
競艇予想ツール - Streamlit Web アプリ（ダーク・スポーティーデザイン）
起動方法: streamlit run app.py
"""

import streamlit as st
import json
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import date
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from boatrace_scraper import BoatraceScraper, VENUE_MAP
from predictor import BoatracePredictor, MLPredictor


@st.cache_resource
def get_predictor():
    """LightGBMモデルがあればMLPredictor、無ければルールベースにフォールバック"""
    try:
        return MLPredictor(model_dir=Path(__file__).parent / "models")
    except FileNotFoundError:
        return BoatracePredictor()
from before_info_scraper import BeforeInfoScraper

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from supabase import create_client

def get_supabase():
    url = os.getenv("SUPABASE_URL") or st.secrets.get("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY") or st.secrets.get("SUPABASE_KEY", "")
    if url and key:
        return create_client(url, key)
    return None

supabase = get_supabase()

# ─────────────────────────────────────────────
# ページ設定
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="競艇予想ツール",
    page_icon="🚤",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    /* ネイビー・スポーティーテーマ */
    .stApp { background-color: #111827; }
    .main .block-container { padding: 1.5rem 2rem; max-width: 1400px; }

    /* Streamlitデフォルト要素を統一 */
    .stSelectbox > div > div { background: #374151 !important; border-color: #4b5563 !important; }
    .stSelectbox > div > div > div { color: #f3f4f6 !important; }
    .stSelectbox svg { color: #9ca3af !important; }
    .stSelectbox label, .stMultiSelect label { color: #9ca3af !important; }

    /* ヘッダー */
    .header-box {
        background: #1f2937;
        border-left: 4px solid #3b82f6;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.5rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .header-title { font-size: 1.6rem; font-weight: 800; color: #f3f4f6; margin: 0; }
    .header-date { color: #6b7280; font-size: 0.85rem; margin: 0; }

    /* カード */
    .stat-card {
        background: #1f2937;
        border: 1px solid #374151;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stat-value { font-size: 1.8rem; font-weight: 800; color: #60a5fa; }
    .stat-label { font-size: 0.78rem; color: #6b7280; margin-top: 0.2rem; }

    /* 買い目ボックス */
    .buy-box {
        background: #1f2937;
        border: 1px solid #3b82f6;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 0.8rem 0;
    }
    .buy-title { font-size: 1rem; font-weight: 700; color: #93c5fd; margin-bottom: 0.8rem; }
    .buy-combo {
        background: #1e3a8a;
        border-radius: 6px;
        padding: 0.4rem 0.9rem;
        margin: 0.3rem 0.2rem;
        font-size: 1rem;
        font-weight: 700;
        color: #bfdbfe;
        display: inline-block;
        letter-spacing: 0.05em;
    }

    /* ボタン */
    .stButton > button {
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.95rem;
        border: 1px solid #374151 !important;
        background: #1f2937 !important;
        color: #f3f4f6 !important;
    }
    .stButton > button[kind="primary"] {
        background: #1d4ed8 !important;
        border-color: #1d4ed8 !important;
        color: #fff !important;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* タブ */
    .stTabs [data-baseweb="tab-list"] {
        background: #1f2937;
        border-radius: 8px;
        padding: 3px;
        gap: 2px;
        border: 1px solid #374151;
        width: 100%;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        color: #9ca3af !important;
        font-weight: 600;
        flex: 1;
        justify-content: center;
    }
    .stTabs [aria-selected="true"] { background: #1d4ed8 !important; color: #fff !important; }

    /* 出走表行 */
    .racer-row {
        display: flex;
        align-items: center;
        padding: 0.7rem 1rem;
        background: #1f2937;
        border-radius: 8px;
        margin: 4px 0;
        gap: 1rem;
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
    .hit-badge { background: #065f46; border: 1px solid #10b981; color: #6ee7b7; border-radius: 6px; padding: 2px 10px; font-size: 0.82rem; font-weight: 700; }
    .miss-badge { background: #7f1d1d; border: 1px solid #ef4444; color: #fca5a5; border-radius: 6px; padding: 2px 10px; font-size: 0.82rem; font-weight: 700; }
    .pending-badge { background: #292524; border: 1px solid #57534e; color: #d6d3d1; border-radius: 6px; padding: 2px 10px; font-size: 0.82rem; font-weight: 700; }

    /* 記録カード */
    .record-card {
        background: #1f2937;
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        margin: 4px 0;
        display: flex;
        align-items: center;
        gap: 1rem;
        flex-wrap: wrap;
    }
    hr { border-color: #374151 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ユーティリティ
# ─────────────────────────────────────────────
RECORD_FILE = Path("prediction_records.json")
CACHE_FILE = Path(f"cache_racers_{date.today().strftime('%Y%m%d')}.json")

def load_records():
    # Supabaseから取得（優先）
    if supabase:
        try:
            res = supabase.table("prediction_records").select("*").order("race_date", desc=True).execute()
            records = []
            for row in res.data:
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
        date_str = date.today().strftime("%Y%m%d")
        res = supabase.table("today_racelist").select("*").eq("race_date", date_str).execute()
        if not res.data:
            return {}
        from boatrace_scraper import RacerInfo
        result = {}
        for row in res.data:
            venue = row["venue_code"]
            rno = row["race_no"]
            racers_data = row["racers"]
            if isinstance(racers_data, str):
                racers_data = json.loads(racers_data)
            if venue not in result:
                result[venue] = {}
            result[venue][rno] = [RacerInfo(**r) for r in racers_data]
        return result
    except Exception as e:
        st.warning(f"出走表Supabase読み込み失敗: {e}")
        return {}

if "today_racers" not in st.session_state:
    cached = load_cache()
    if cached:
        st.session_state.today_racers = restore_cache(cached)
    else:
        st.session_state.today_racers = load_today_racelist_from_supabase()

# ─────────────────────────────────────────────
# ヘッダー
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="header-box">
    <span style="font-size:2.5rem">🚤</span>
    <div>
        <p class="header-title">競艇予想ツール</p>
        <p class="header-date">📅 {date.today().strftime('%Y年%m月%d日')}</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# タブ
# ─────────────────────────────────────────────
tab0, tab1, tab2, tab3, tab4 = st.tabs(["🔥  ピックアップ", "🎯  予想", "📊  直前情報", "📋  結果確認", "📈  成績記録"])

# ─────────────────────────────────────────────
# タブ0: ピックアップ
# ─────────────────────────────────────────────
PICKUP_TOP_SCORE_MIN = 25.0
PICKUP_SCORE_GAP_MIN = 10.0

with tab0:
    records = load_records()
    today_str = date.today().strftime("%Y%m%d")
    today_records = [r for r in records if r["race_date"] == today_str]

    pickups = [
        r for r in today_records
        if r.get("top_score") is not None and r.get("score_gap") is not None
        and r["top_score"] >= PICKUP_TOP_SCORE_MIN
        and r["score_gap"] >= PICKUP_SCORE_GAP_MIN
    ]

    st.markdown(
        f"<p style='color:#9ca3af;font-size:0.85rem;margin-bottom:1rem'>"
        f"1位確信度 {PICKUP_TOP_SCORE_MIN:.0f}%以上 かつ 2位との差 {PICKUP_SCORE_GAP_MIN:.0f}pt以上 のレースを表示"
        f"</p>",
        unsafe_allow_html=True,
    )

    if not today_records:
        st.info("本日の予想データがまだありません。08:00のバッチを待つか、予想タブで取得してください。")
    elif not pickups:
        st.info("本日、条件を満たすレースはまだありません。")
    else:
        pickups_sorted = sorted(pickups, key=lambda r: r["top_score"], reverse=True)
        for r in pickups_sorted:
            card_html = (
                "<div style='background:#1f2937;border:1px solid #f59e0b;border-radius:10px;"
                "padding:1rem 1.2rem;margin:0.5rem 0;display:flex;align-items:center;"
                "gap:1.2rem;flex-wrap:wrap'>"
                f"<span style='font-size:1.1rem;font-weight:800;color:#fbbf24'>🔥 {r['venue_name']} {r['race_no']}R</span>"
                f"<span style='color:#93c5fd;font-size:0.95rem'>単勝 <strong>{r['tansho']}</strong></span>"
                f"<span style='color:#93c5fd;font-size:0.95rem'>3連単 <strong>{' / '.join(r['sanren_tan'])}</strong></span>"
                f"<span style='margin-left:auto;color:#9ca3af;font-size:0.85rem'>"
                f"1位確信度 <strong style='color:#fbbf24'>{r['top_score']:.1f}</strong>"
                f" / 差 <strong style='color:#fbbf24'>{r['score_gap']:.1f}</strong>pt</span>"
                "</div>"
            )
            st.markdown(card_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# タブ1: 予想
# ─────────────────────────────────────────────
with tab1:
    # 出走表データが無ければ手動取得ボタンを表示（バッチが08:00に自動取得）
    if not st.session_state.today_racers:
        if st.button("📥  出走表を取得（通常はバッチが自動取得します）", type="secondary", use_container_width=True):
            sc = get_scraper()
            if sc:
                from crawler import get_holding_venues
                with st.spinner("開催会場を確認中..."):
                    venues = get_holding_venues(sc, date.today())
                if not venues:
                    st.warning("本日の開催会場が見つかりませんでした")
                else:
                    progress = st.progress(0, text="取得中...")
                    today_racers = {}
                    total = len(venues) * 12
                    count = 0
                    for venue in venues:
                        today_racers[venue] = {}
                        for rno in range(1, 13):
                            racers = sc.get_racelist(date.today(), venue, rno)
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
                                try:
                                    supabase.table("today_racelist").upsert({
                                        "race_date": date.today().strftime("%Y%m%d"),
                                        "venue_code": venue,
                                        "venue_name": VENUE_MAP[venue],
                                        "race_no": rno,
                                        "racers": json.dumps([asdict(r) for r in racers], ensure_ascii=False),
                                    }, on_conflict="race_date,venue_code,race_no").execute()
                                except Exception as e:
                                    st.warning(f"出走表Supabase保存失敗: {e}")

                    total_races = sum(len(v) for v in today_racers.values())
                    st.success(f"✅  {len(venues)}会場 {total_races}レース取得完了")
                    st.rerun()

    if st.session_state.today_racers:
        col_v, col_r = st.columns(2)
        with col_v:
            available_venues = list(st.session_state.today_racers.keys())
            venue_labels = [f"{VENUE_MAP[v]}（{v}）" for v in available_venues]
            sel_label = st.selectbox("会場", venue_labels)
            sel_venue = available_venues[venue_labels.index(sel_label)]
        with col_r:
            available_races = sorted(st.session_state.today_racers[sel_venue].keys())
            sel_race = st.selectbox("レース", available_races, format_func=lambda x: f"{x}R")

        racers = st.session_state.today_racers[sel_venue][sel_race]

        # 出走表（カラー枠番付き）
        st.markdown("#### 出走表")
        for r in racers:
            st.markdown(f"""
            <div class="racer-row">
                <div class="lane-badge lane-{r.lane}">{r.lane}</div>
                <div style="flex:1">
                    <span style="font-weight:700;color:#e0e6ff">{r.name}</span>
                    <span style="margin-left:8px;font-size:0.8rem;color:#64748b">{r.rank}</span>
                </div>
                <div style="text-align:right">
                    <span style="font-size:0.85rem;color:#64748b">勝率 </span>
                    <span style="font-weight:700;color:#3b82f6">{r.win_rate_all or '-'}</span>
                </div>
                <div style="text-align:right;min-width:80px">
                    <span style="font-size:0.85rem;color:#64748b">モーター </span>
                    <span style="font-weight:600;color:#94a3b8">{r.motor_2rate or '-'}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🎯  このレースを予想する", type="primary", use_container_width=True):
            predictor = get_predictor()
            pred = predictor.predict(racers, race_date=date.today().strftime("%Y%m%d"),
                                      venue_name=VENUE_MAP[sel_venue], race_no=sel_race)
            st.session_state.prediction = pred
            st.session_state.last_venue = sel_venue
            st.session_state.last_race = sel_race
            save_record({
                "race_date": date.today().strftime("%Y%m%d"),
                "venue_code": sel_venue, "venue_name": VENUE_MAP[sel_venue], "race_no": sel_race,
                "tansho": pred.tansho, "sanren_tan": pred.sanren_tan,
                "sanren_fuku": pred.sanren_fuku, "hit": None, "actual": "", "payout": None,
            })
            st.success("✅  予想を記録しました")

    else:
        st.info("本日の出走表データがまだありません。08:00のバッチを待つか、上のボタンで取得してください。")

    # 予想結果表示
    if st.session_state.prediction:
        pred = st.session_state.prediction
        st.markdown("---")
        st.markdown("### 🏆  予想結果")

        sorted_scores = sorted(pred.scores, key=lambda s: s.predicted_rank)
        medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣"]

        # スコアバーチャート
        fig = go.Figure(go.Bar(
            x=[s.total_score for s in sorted_scores],
            y=[f"{s.lane}枠 {s.name}" for s in sorted_scores],
            orientation='h',
            marker_color=['#f59e0b','#94a3b8','#b45309','#475569','#475569','#475569'],
            text=[f"{s.total_score:.1f}" for s in sorted_scores],
            textposition='outside',
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#94a3b8',
            height=220,
            margin=dict(l=10, r=60, t=10, b=10),
            xaxis=dict(gridcolor='#1e293b', showgrid=True),
            yaxis=dict(autorange='reversed'),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # 買い目
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div class="buy-box">
                <p class="buy-title">💰 推奨買い目</p>
                <p style="color:#94a3b8;font-size:0.85rem;margin:0">単勝</p>
                <div class="buy-combo">{pred.tansho}</div>
                <p style="color:#94a3b8;font-size:0.85rem;margin:0.8rem 0 0">3連単</p>
                {''.join([f'<div class="buy-combo">{c}</div>' for c in pred.sanren_tan])}
                <p style="color:#94a3b8;font-size:0.85rem;margin:0.8rem 0 0">3連複</p>
                <div class="buy-combo">{pred.sanren_fuku}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div class="buy-box">
                <p class="buy-title">📊 予想順位</p>
                {''.join([f'<p style="margin:0.4rem 0;color:#e0e6ff">{medals[s.predicted_rank-1]} {s.lane}枠 {s.name} <span style="color:#3b82f6;font-weight:700">{s.total_score:.1f}pt</span></p>' for s in sorted_scores])}
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# タブ2: 直前情報
# ─────────────────────────────────────────────
with tab2:
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
            info = before_scraper.get_before_info(date.today(), vc2, rno2)
            if info:
                st.session_state.before_info = info
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
                    <span style="color:#94a3b8">{s.course}コース</span>
                    <span style="font-weight:700;color:#3b82f6;margin-left:auto">ST: {s.st or '-'}</span>
                </div>
                """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# タブ3: 結果確認
# ─────────────────────────────────────────────
with tab3:
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
                result = sc.get_result(date.today(), vc3, rno3)
                if result and result.arrival:
                    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣"]
                    col_r1, col_r2 = st.columns(2)
                    with col_r1:
                        st.markdown("#### 着順")
                        for i, boat in enumerate(result.arrival):
                            st.markdown(f"<p style='margin:0.3rem 0;color:#e0e6ff'>{medals[i]} <strong>{boat}号艇</strong></p>", unsafe_allow_html=True)
                    with col_r2:
                        if result.payouts:
                            st.markdown("#### 払戻金")
                            for key, val in result.payouts.items():
                                if val:
                                    st.markdown(f"<p style='margin:0.2rem 0;color:#94a3b8'>{key}: <strong style='color:#3b82f6'>¥{val:,}</strong></p>", unsafe_allow_html=True)

                    # 記録更新
                    records = load_records()
                    date_str = date.today().strftime("%Y%m%d")
                    for r in records:
                        if r["race_date"] == date_str and r["venue_code"] == vc3 and r["race_no"] == rno3:
                            hit, actual = check_hit(r["sanren_tan"], result.arrival)
                            r["hit"] = hit
                            r["actual"] = actual
                            r["payout"] = result.payouts.get(f"3連単_{actual}")
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

# ─────────────────────────────────────────────
# タブ4: 成績記録
# ─────────────────────────────────────────────
with tab4:
    records = load_records()

    if not records:
        st.info("まだ予想記録がありません。予想タブで予想すると自動で記録されます。")
    else:
        checked = [r for r in records if r["hit"] is not None]
        hits = [r for r in checked if r["hit"]]
        hit_rate = len(hits) / len(checked) * 100 if checked else 0
        total_payout = sum(r["payout"] or 0 for r in hits)
        total_cost = len(checked) * 100  # 仮に1レース100円

        # サマリーカード
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="stat-card"><div class="stat-value">{len(records)}</div><div class="stat-label">総予想数</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-card"><div class="stat-value">{len(hits)}</div><div class="stat-label">的中数</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="stat-card"><div class="stat-value">{hit_rate:.1f}%</div><div class="stat-label">的中率</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="stat-card"><div class="stat-value">¥{total_payout:,}</div><div class="stat-label">総払戻金</div></div>', unsafe_allow_html=True)

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

            extra1 = f"<span style='color:#94a3b8;font-size:0.85rem'>{actual_text}</span>" if actual_text else ""
            extra2 = f"<span style='font-size:0.85rem'>{payout_text}</span>" if payout_text else ""

            card_html = (
                "<div style='background:#0d1b3e;border:1px solid #1e3a8a;border-radius:10px;"
                "padding:0.8rem 1rem;margin:0.4rem 0;display:flex;align-items:center;"
                "gap:1rem;flex-wrap:wrap'>"
                f"<span style='color:#64748b;font-size:0.85rem'>{formatted}</span>"
                f"<span style='font-weight:700;color:#e0e6ff'>{r['venue_name']} {r['race_no']}R</span>"
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