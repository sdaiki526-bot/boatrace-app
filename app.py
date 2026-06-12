"""
競艇予想ツール - Streamlit Web アプリ
起動方法: streamlit run app.py
"""

import streamlit as st
import json
from datetime import date
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from boatrace_scraper import BoatraceScraper, VENUE_MAP
from predictor import BoatracePredictor
from before_info_scraper import BeforeInfoScraper

# ─────────────────────────────────────────────
# ページ設定
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="競艇予想ツール",
    page_icon="🚤",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .main { padding: 0.5rem; }
    .stButton > button {
        width: 100%;
        height: 3rem;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 10px;
    }
    .buy-box {
        background: #1a1a2e;
        color: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 予想記録の保存・読み込み
# ─────────────────────────────────────────────
RECORD_FILE = Path("prediction_records.json")

def load_records():
    if RECORD_FILE.exists():
        return json.loads(RECORD_FILE.read_text(encoding="utf-8"))
    return []

def save_record(record):
    records = load_records()
    # 同じレースの記録があれば上書き
    key = f"{record['race_date']}_{record['venue_code']}_{record['race_no']}"
    records = [r for r in records if f"{r['race_date']}_{r['venue_code']}_{r['race_no']}" != key]
    records.append(record)
    RECORD_FILE.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

def check_hit(sanren_tan_combos, arrival):
    """3連単が的中しているか確認"""
    if len(arrival) < 3:
        return False, ""
    actual = f"{arrival[0]}-{arrival[1]}-{arrival[2]}"
    for combo in sanren_tan_combos:
        if combo == actual:
            return True, combo
    return False, actual

# ─────────────────────────────────────────────
# 出走表キャッシュ（日付単位でファイル保存）
# ─────────────────────────────────────────────
CACHE_FILE = Path(f"cache_racers_{date.today().strftime('%Y%m%d')}.json")

def load_cache():
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    return {}

def save_cache(today_racers):
    from dataclasses import asdict
    serializable = {}
    for venue, races in today_racers.items():
        serializable[venue] = {}
        for rno, racers in races.items():
            serializable[venue][str(rno)] = [asdict(r) for r in racers]
    CACHE_FILE.write_text(json.dumps(serializable, ensure_ascii=False), encoding="utf-8")

def restore_cache(data):
    from boatrace_scraper import RacerInfo
    restored = {}
    for venue, races in data.items():
        restored[venue] = {}
        for rno, racers in races.items():
            restored[venue][int(rno)] = [RacerInfo(**r) for r in racers]
    return restored

# ─────────────────────────────────────────────
# セッション初期化
# ─────────────────────────────────────────────
if "scraper" not in st.session_state:
    st.session_state.scraper = None
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "today_racers" not in st.session_state:
    cached = load_cache()
    if cached:
        st.session_state.today_racers = restore_cache(cached)
    else:
        st.session_state.today_racers = {}
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "before_info" not in st.session_state:
    st.session_state.before_info = None
if "last_venue" not in st.session_state:
    st.session_state.last_venue = None
if "last_race" not in st.session_state:
    st.session_state.last_race = None

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
# ヘッダー
# ─────────────────────────────────────────────
st.title("🚤 競艇予想ツール")
st.caption(f"📅 {date.today().strftime('%Y年%m月%d日')}")

# ─────────────────────────────────────────────
# タブ
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🎯 予想", "📊 直前情報", "📋 結果確認", "📈 成績記録"])

# ─────────────────────────────────────────────
# タブ1: 予想
# ─────────────────────────────────────────────
with tab1:
    if st.button("📥 今日の出走表を一括取得", type="primary"):
        sc = get_scraper()
        if sc:
            from crawler import get_holding_venues
            with st.spinner("開催会場を確認中..."):
                venues = get_holding_venues(sc, date.today())
            if not venues:
                st.warning("本日の開催会場が見つかりませんでした")
            else:
                st.info(f"開催会場: {', '.join([VENUE_MAP[v] for v in venues])}")
                progress = st.progress(0)
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
                        progress.progress(count / total)
                st.session_state.today_racers = today_racers
                save_cache(today_racers)
                total_races = sum(len(v) for v in today_racers.values())
                st.success(f"✅ {len(venues)}会場 {total_races}レース分を取得しました")

    # ── 一括予想ボタン ──────────────────────
    if st.session_state.today_racers:
        if st.button("🎯 今日の全レースを一括予想", type="secondary"):
            predictor = BoatracePredictor()
            count = 0
            for venue, races in st.session_state.today_racers.items():
                for rno, racers in races.items():
                    pred = predictor.predict(
                        racers,
                        race_date=date.today().strftime("%Y%m%d"),
                        venue_name=VENUE_MAP[venue],
                        race_no=rno,
                    )
                    record = {
                        "race_date":   date.today().strftime("%Y%m%d"),
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
            st.success(f"✅ {count}レース分の予想を記録しました！成績記録タブで確認できます")

    st.divider()
    st.subheader("レース選択")

    if st.session_state.today_racers:
        available_venues = list(st.session_state.today_racers.keys())
        venue_labels = [f"{VENUE_MAP[v]}（{v}）" for v in available_venues]
        selected_venue_label = st.selectbox("会場", venue_labels)
        selected_venue = available_venues[venue_labels.index(selected_venue_label)]
        available_races = sorted(st.session_state.today_racers[selected_venue].keys())
        selected_race = st.selectbox("レース", available_races, format_func=lambda x: f"{x}R")
        racers = st.session_state.today_racers[selected_venue][selected_race]

        st.subheader("出走表")
        for r in racers:
            col1, col2, col3 = st.columns([1, 3, 2])
            with col1:
                st.markdown(f"**{r.lane}枠**")
            with col2:
                st.markdown(f"{r.name}　{r.rank}")
            with col3:
                st.markdown(f"勝率 {r.win_rate_all or '-'}")

        st.divider()

        if st.button("🎯 予想する", type="primary"):
            predictor = BoatracePredictor()
            pred = predictor.predict(
                racers,
                race_date=date.today().strftime("%Y%m%d"),
                venue_name=VENUE_MAP[selected_venue],
                race_no=selected_race,
            )
            st.session_state.prediction = pred
            st.session_state.last_venue = selected_venue
            st.session_state.last_race = selected_race

            # 予想を自動記録
            record = {
                "race_date":   date.today().strftime("%Y%m%d"),
                "venue_code":  selected_venue,
                "venue_name":  VENUE_MAP[selected_venue],
                "race_no":     selected_race,
                "tansho":      pred.tansho,
                "sanren_tan":  pred.sanren_tan,
                "sanren_fuku": pred.sanren_fuku,
                "hit":         None,
                "actual":      "",
                "payout":      None,
            }
            save_record(record)
            st.success("✅ 予想を記録しました")

    else:
        st.info("上のボタンで今日の出走表を取得してください")
        col1, col2 = st.columns(2)
        with col1:
            venue_options = {v: k for k, v in VENUE_MAP.items()}
            venue_name = st.selectbox("会場", list(venue_options.keys()),
                                      index=list(venue_options.keys()).index("住之江"))
            venue_code = venue_options[venue_name]
        with col2:
            race_no = st.selectbox("レース", list(range(1, 13)), format_func=lambda x: f"{x}R")

        if st.button("📥 この出走表を取得", type="secondary"):
            sc = get_scraper()
            if sc:
                with st.spinner("取得中..."):
                    racers = sc.get_racelist(date.today(), venue_code, race_no)
                    if racers:
                        if venue_code not in st.session_state.today_racers:
                            st.session_state.today_racers[venue_code] = {}
                        st.session_state.today_racers[venue_code][race_no] = racers
                        st.success(f"✅ {venue_name} {race_no}R を取得しました")
                        st.rerun()
                    else:
                        st.warning("出走表が取得できませんでした")

    # 予想結果表示
    if st.session_state.prediction:
        pred = st.session_state.prediction
        st.subheader("🏆 予想結果")
        medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣"]
        for s in sorted(pred.scores, key=lambda s: s.predicted_rank):
            st.markdown(f"{medals[s.predicted_rank-1]} **{s.lane}枠** {s.name}　{s.rank}　{s.total_score:.1f}点")
        st.divider()
        st.markdown('<div class="buy-box">', unsafe_allow_html=True)
        st.markdown("### 💰 推奨買い目")
        st.markdown(f"**単勝:** {pred.tansho}")
        st.markdown("**3連単:**")
        for combo in pred.sanren_tan:
            st.markdown(f"　✅ {combo}")
        st.markdown(f"**3連複:** {pred.sanren_fuku}")
        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# タブ2: 直前情報
# ─────────────────────────────────────────────
with tab2:
    st.subheader("展示タイム・気象情報")
    st.caption("レース約1時間前から取得可能")
    col1, col2 = st.columns(2)
    with col1:
        venue_options2 = {v: k for k, v in VENUE_MAP.items()}
        venue_name2 = st.selectbox("会場", list(venue_options2.keys()), key="venue2",
                                   index=list(venue_options2.keys()).index("住之江"))
        venue_code2 = venue_options2[venue_name2]
    with col2:
        race_no2 = st.selectbox("レース", list(range(1, 13)), key="race2",
                                 format_func=lambda x: f"{x}R")

    if st.button("📥 直前情報を取得", type="primary", key="before_btn"):
        before_scraper = BeforeInfoScraper(delay=1.0)
        with st.spinner("取得中..."):
            info = before_scraper.get_before_info(date.today(), venue_code2, race_no2)
            if info:
                st.session_state.before_info = info
                st.success("✅ 取得しました")
            else:
                st.warning("直前情報がまだ公開されていません")

    if st.session_state.before_info:
        info = st.session_state.before_info
        if info.exhibitions:
            st.subheader("展示タイム")
            for e in info.exhibitions:
                col1, col2, col3 = st.columns([1, 3, 2])
                with col1:
                    st.markdown(f"**{e.lane}枠**")
                with col2:
                    st.markdown(e.name)
                with col3:
                    et = e.exhibition_time
                    if et:
                        st.markdown(f"🔥 **{et}**" if et <= 6.70 else f"{et}")
                    else:
                        st.markdown("-")
        if info.weather:
            st.subheader("🌤 気象情報")
            w = info.weather
            cols = st.columns(3)
            with cols[0]:
                st.metric("気温", f"{w.temperature}℃" if w.temperature else "-")
            with cols[1]:
                st.metric("風速", f"{w.wind_speed}m" if w.wind_speed else "-")
            with cols[2]:
                st.metric("波高", f"{w.wave_height}cm" if w.wave_height else "-")
        if info.start_exhibition:
            st.subheader("スタート展示")
            for s in info.start_exhibition:
                st.markdown(f"{s.course}コース　艇{s.boat_no}　ST: {s.st or '-'}")

# ─────────────────────────────────────────────
# タブ3: 結果確認
# ─────────────────────────────────────────────
with tab3:
    st.subheader("レース結果")
    col1, col2 = st.columns(2)
    with col1:
        venue_options3 = {v: k for k, v in VENUE_MAP.items()}
        venue_name3 = st.selectbox("会場", list(venue_options3.keys()), key="venue3",
                                   index=list(venue_options3.keys()).index("住之江"))
        venue_code3 = venue_options3[venue_name3]
    with col2:
        race_no3 = st.selectbox("レース", list(range(1, 13)), key="race3",
                                 format_func=lambda x: f"{x}R")

    if st.button("📥 結果を取得して記録を更新", type="primary", key="result_btn"):
        sc = get_scraper()
        if sc:
            with st.spinner("取得中..."):
                result = sc.get_result(date.today(), venue_code3, race_no3)
                if result and result.arrival:
                    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣"]
                    st.subheader("着順")
                    for i, boat in enumerate(result.arrival):
                        st.markdown(f"{medals[i]} **{boat}号艇**")

                    if result.payouts:
                        st.subheader("払戻金")
                        for key, val in result.payouts.items():
                            if val:
                                st.markdown(f"**{key}**: ¥{val:,}")

                    # 予想記録を更新
                    records = load_records()
                    date_str = date.today().strftime("%Y%m%d")
                    updated = False
                    for r in records:
                        if r["race_date"] == date_str and r["venue_code"] == venue_code3 and r["race_no"] == race_no3:
                            hit, actual = check_hit(r["sanren_tan"], result.arrival)
                            r["hit"] = hit
                            r["actual"] = f"{result.arrival[0]}-{result.arrival[1]}-{result.arrival[2]}" if len(result.arrival) >= 3 else ""
                            sanren_key = f"3連単_{r['actual']}"
                            r["payout"] = result.payouts.get(sanren_key)
                            updated = True
                            if hit:
                                st.success(f"🎉 的中！ {actual} ¥{r['payout']:,}" if r['payout'] else "🎉 的中！")
                            else:
                                st.error(f"❌ ハズレ　実際: {r['actual']}　予想: {' / '.join(r['sanren_tan'])}")
                    if updated:
                        RECORD_FILE.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
                else:
                    st.warning("結果がまだ出ていません")

# ─────────────────────────────────────────────
# タブ4: 成績記録
# ─────────────────────────────────────────────
with tab4:
    st.subheader("📈 予想成績")

    records = load_records()
    if not records:
        st.info("まだ予想記録がありません。予想タブで予想すると自動で記録されます。")
    else:
        # 集計
        total = len(records)
        checked = [r for r in records if r["hit"] is not None]
        hits = [r for r in checked if r["hit"]]
        hit_rate = len(hits) / len(checked) * 100 if checked else 0
        total_payout = sum(r["payout"] or 0 for r in hits)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("予想数", f"{total}レース")
        with col2:
            st.metric("的中率", f"{hit_rate:.1f}%" if checked else "-")
        with col3:
            st.metric("総払戻", f"¥{total_payout:,}" if hits else "-")

        st.divider()
        st.subheader("記録一覧")

        for r in sorted(records, reverse=True, key=lambda x: (x["race_date"], x["venue_code"], x["race_no"])):
            hit_icon = "🎉" if r["hit"] else ("❌" if r["hit"] is False else "⏳")
            date_str = r["race_date"]
            formatted = f"{date_str[:4]}/{date_str[4:6]}/{date_str[6:]}"
            with st.expander(f"{hit_icon} {formatted} {r['venue_name']} {r['race_no']}R"):
                st.markdown(f"**予想3連単:** {' / '.join(r['sanren_tan'])}")
                if r["actual"]:
                    st.markdown(f"**実際の結果:** {r['actual']}")
                if r["hit"]:
                    st.markdown(f"**払戻金:** ¥{r['payout']:,}" if r["payout"] else "**払戻金:** -")
                elif r["hit"] is False:
                    st.markdown("**結果:** ハズレ")
                else:
                    st.markdown("**結果:** 未確認（結果確認タブで更新）")

        if st.button("🗑 記録をリセット", type="secondary"):
            RECORD_FILE.unlink(missing_ok=True)
            st.success("記録をリセットしました")
            st.rerun()