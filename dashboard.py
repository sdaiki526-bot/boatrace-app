"""
ダッシュボード（会場グリッド）
app.py から render_dashboard(supabase, today_str) を呼ぶだけ。
既存コードには影響しない独立モジュール。
"""
import json
from datetime import datetime, timezone, timedelta

import streamlit as st

JST = timezone(timedelta(hours=9))

# 狙い目の判定しきい値（app.py の VALUE_GAP_MAX / VALUE_SCORE_MAX と揃える）
VALUE_GAP_MAX = 15.0
VALUE_SCORE_MAX = 30.0


def _fetch_venue_summary(supabase, today_str):
    """今日の会場ごとの集計（レース数・狙い目数・締切時刻リスト）を返す"""
    if not supabase:
        return []

    # 出走表（締切時刻つき）
    try:
        tr = (supabase.table("today_racelist")
              .select("venue_code,venue_name,race_no,deadline_time")
              .eq("race_date", today_str).execute()).data or []
    except Exception as e:
        st.warning(f"会場データ取得失敗: {e}")
        return []

    # 予想（狙い目判定用）
    try:
        pr = (supabase.table("prediction_records")
              .select("venue_code,race_no,top_score,score_gap")
              .eq("race_date", today_str).execute()).data or []
    except Exception:
        pr = []

    # 予想を (venue, race_no) で引けるように
    pred_map = {(p["venue_code"], p["race_no"]): p for p in pr}

    venues = {}
    for row in tr:
        vc = row["venue_code"]
        v = venues.setdefault(vc, {
            "venue_code": vc,
            "venue_name": row["venue_name"],
            "races": 0,
            "value_count": 0,
            "deadlines": [],
        })
        v["races"] += 1
        if row.get("deadline_time"):
            v["deadlines"].append(row["deadline_time"])
        p = pred_map.get((vc, row["race_no"]))
        if p and p.get("top_score") is not None and p.get("score_gap") is not None:
            if p["score_gap"] <= VALUE_GAP_MAX or p["top_score"] <= VALUE_SCORE_MAX:
                v["value_count"] += 1

    return list(venues.values())


def _next_deadline(deadlines, now_hm):
    """まだ締切が来ていない最も近い時刻を返す。全部過ぎていたら None"""
    future = sorted([d for d in deadlines if d > now_hm])
    return future[0] if future else None


def render_dashboard(supabase, today_str):
    """会場グリッド型ダッシュボードを描画する"""
    now = datetime.now(JST)
    now_hm = now.strftime("%H:%M")

    venues = _fetch_venue_summary(supabase, today_str)

    # サマリー集計
    total_value = sum(v["value_count"] for v in venues)
    total_races = sum(v["races"] for v in venues)

    # サマリーカード
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{total_value}</div>'
                    f'<div class="stat-label">本日の狙い目</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{len(venues)}</div>'
                    f'<div class="stat-label">開催会場</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{total_races}</div>'
                    f'<div class="stat-label">総レース数</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p style='color:#475569;font-size:0.9rem;margin-bottom:0.6rem'>"
                "🏟 本日の開催会場</p>", unsafe_allow_html=True)

    if not venues:
        st.info("本日の開催会場データがまだありません。朝のバッチを待ってください。")
        return

    # 狙い目数が多い順に並べる
    venues.sort(key=lambda v: (-v["value_count"], v["venue_name"]))

    # 4列グリッドで会場マスを並べる
    cols_per_row = 4
    for i in range(0, len(venues), cols_per_row):
        row_venues = venues[i:i + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, v in zip(cols, row_venues):
            nd = _next_deadline(v["deadlines"], now_hm)
            soon = (now + timedelta(minutes=15)).strftime("%H:%M")
            # マスの色を状況で変える
            if nd and nd <= soon:
                bg, border = "#eff6ff", "#1d4ed8"
                badge, badge_color = "⏰ まもなく", "#1d4ed8"
            elif v["value_count"] > 0:
                bg, border = "#fffbeb", "#f59e0b"
                badge, badge_color = f"🔥 狙い{v['value_count']}", "#d97706"
            else:
                bg, border = "#f8fafc", "#e2e8f0"
                badge, badge_color = "狙い目なし", "#94a3b8"

            next_label = f"次 {nd}" if nd else "本日終了"
            with col:
                st.markdown(
                    f"<div style='background:{bg};border:1px solid {border};border-radius:10px;"
                    f"padding:0.7rem;text-align:center;margin-bottom:0.5rem'>"
                    f"<p style='font-size:0.95rem;font-weight:700;color:#1e293b;margin:0 0 0.4rem'>{v['venue_name']}</p>"
                    f"<span style='font-size:0.72rem;color:{badge_color};font-weight:700'>{badge}</span>"
                    f"<p style='font-size:0.72rem;color:#64748b;margin:0.4rem 0 0'>{next_label}</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.markdown(
        "<p style='color:#94a3b8;font-size:0.72rem;margin-top:0.5rem'>"
        "枠の色: <span style='color:#d97706'>オレンジ=狙い目あり</span> / "
        "<span style='color:#1d4ed8'>青=まもなく締切</span> / グレー=狙い目なし</p>",
        unsafe_allow_html=True,
    )