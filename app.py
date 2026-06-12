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
# セッション初期化
# ─────────────────────────────────────────────
if "scraper" not in st.session_state:
    st.session_state.scraper = None
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "today_racers" not in st.session_state:
    st.session_state.today_racers = {}   # {venue_code: {race_no: [racers]}}
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "before_info" not in st.session_state:
    st.session_state.before_info = None

# ─────────────────────────────────────────────
# スクレイパー取得
# ─────────────────────────────────────────────
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
tab1, tab2, tab3 = st.tabs(["🎯 予想", "📊 直前情報", "📋 結果確認"])

# ─────────────────────────────────────────────
# タブ1: 予想
# ─────────────────────────────────────────────
with tab1:

    # ── 今日の出走表を一括取得ボタン ──────────
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
                total_races = sum(len(v) for v in today_racers.values())
                st.success(f"✅ 取得完了！{len(venues)}会場 {total_races}レース分のデータを取得しました")

    st.divider()

    # ── レース選択・予想 ─────────────────────
    st.subheader("レース選択")

    # 取得済みデータがあれば会場・レースを選択
    if st.session_state.today_racers:
        available_venues = list(st.session_state.today_racers.keys())
        venue_labels = [f"{VENUE_MAP[v]}（{v}）" for v in available_venues]
        selected_venue_label = st.selectbox("会場", venue_labels)
        selected_venue = available_venues[venue_labels.index(selected_venue_label)]

        available_races = sorted(st.session_state.today_racers[selected_venue].keys())
        selected_race = st.selectbox("レース", available_races, format_func=lambda x: f"{x}R")

        racers = st.session_state.today_racers[selected_venue][selected_race]

        # 出走表表示
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

    else:
        # 取得済みデータがない場合は個別取得
        st.info("上の「今日の出走表を一括取得」ボタンを押してください")

        col1, col2 = st.columns(2)
        with col1:
            venue_options = {v: k for k, v in VENUE_MAP.items()}
            venue_name = st.selectbox("会場", list(venue_options.keys()), index=list(venue_options.keys()).index("住之江"))
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
        sorted_scores = sorted(pred.scores, key=lambda s: s.predicted_rank)
        for s in sorted_scores:
            st.markdown(f"{medals[s.predicted_rank-1]} **{s.lane}枠** {s.name}　{s.rank}　{s.total_score:.1f}点")

        st.divider()
        st.markdown('<div class="buy-box">', unsafe_allow_html=True)
        st.markdown("### 💰 推奨買い目")
        st.markdown(f"**単勝:** {pred.tansho}")
        st.markdown(f"**3連単:**")
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
        race_no2 = st.selectbox("レース", list(range(1, 13)), key="race2", format_func=lambda x: f"{x}R")

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
        race_no3 = st.selectbox("レース", list(range(1, 13)), key="race3", format_func=lambda x: f"{x}R")

    if st.button("📥 結果を取得", type="primary", key="result_btn"):
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
                    if result.start_times:
                        st.subheader("スタートタイム")
                        for lane, st_time in result.start_times.items():
                            st.markdown(f"{lane}号艇: {st_time}")
                else:
                    st.warning("結果がまだ出ていません")py -m streamlit run app.py