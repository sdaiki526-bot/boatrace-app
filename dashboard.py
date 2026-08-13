"""
ダッシュボード（会場グリッド）
app.py から render_dashboard(supabase, today_str) を呼ぶだけ。
既存コードには影響しない独立モジュール。
"""
import json
from datetime import datetime, timezone, timedelta

import streamlit as st

JST = timezone(timedelta(hours=9))

# 狙い目の判定しきい値（app.py の VALUE_GAP_MAX と揃える）
# エンタメ表示: 回収率100%達成は不可能と検証済みのため、「儲かるレース」ではなく
# 「モデルが決着を読みにくい、荒れそうで見ていて面白いレース」を選ぶ目的で使う。
# score_gap<=0.6の単一条件のみ(top_scoreは使わない)。
VALUE_GAP_MAX = 0.6


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
        if p and p.get("score_gap") is not None:
            if p["score_gap"] <= VALUE_GAP_MAX:
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
                bg, border = "#1a2540", "#3b82f6"
                badge, badge_color = "⏰ まもなく", "#60a5fa"
            elif v["value_count"] > 0:
                bg, border = "#1a2540", "#f5c542"
                badge, badge_color = f"🔥 狙い{v['value_count']}", "#f5c542"
            else:
                bg, border = "#161e30", "#2d3a52"
                badge, badge_color = "狙い目なし", "#5a6b85"

            next_label = f"次 {nd}" if nd else "本日終了"
            with col:
                st.markdown(
                    f"<div style='background:{bg};border:1px solid {border};border-radius:10px;"
                    f"padding:0.7rem;text-align:center;margin-bottom:0.5rem'>"
                    f"<p style='font-size:0.95rem;font-weight:700;color:#f1f5f9;margin:0 0 0.4rem'>{v['venue_name']}</p>"
                    f"<span style='font-size:0.72rem;color:{badge_color};font-weight:700'>{badge}</span>"
                    f"<p style='font-size:0.72rem;color:#8b9bb4;margin:0.4rem 0 0'>{next_label}</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.markdown(
        "<p style='color:#94a3b8;font-size:0.72rem;margin-top:0.5rem'>"
        "枠の色: <span style='color:#d97706'>オレンジ=狙い目あり</span> / "
        "<span style='color:#1d4ed8'>青=まもなく締切</span> / グレー=狙い目なし</p>",
        unsafe_allow_html=True,
    )

    # ─────────────────────────────────────────────
    # 締切が近い順の狙い目レースリスト
    # ─────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p style='color:#475569;font-size:0.9rem;margin-bottom:0.6rem'>"
                "💰 狙い目レース（締切が近い順・エンタメ）</p>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#64748b;font-size:0.78rem;margin-bottom:0.6rem'>"
        "モデルが決着を読みにくい、荒れそうなレースです（回収率の保証はありません）。</p>",
        unsafe_allow_html=True,
    )

    # 予想と締切を結合して狙い目だけ抽出
    try:
        pr = (supabase.table("prediction_records")
              .select("venue_name,venue_code,race_no,top_score,score_gap,odds_value,sanren_tan")
              .eq("race_date", today_str).execute()).data or []
    except Exception:
        pr = []

    # 締切時刻を辞書化
    deadline_map = {}
    try:
        tr = (supabase.table("today_racelist")
              .select("venue_code,race_no,deadline_time")
              .eq("race_date", today_str).execute()).data or []
        for row in tr:
            if row.get("deadline_time"):
                deadline_map[(row["venue_code"], row["race_no"])] = row["deadline_time"]
    except Exception:
        pass

    value_races = []
    for p in pr:
        if p.get("score_gap") is None:
            continue
        if p["score_gap"] <= VALUE_GAP_MAX:
            dl = deadline_map.get((p["venue_code"], p["race_no"]), "")
            value_races.append({**p, "deadline": dl})

    # 締切がまだ来ていないものを時刻順に、その後に締切済みを並べる
    upcoming = sorted([v for v in value_races if v["deadline"] and v["deadline"] > now_hm],
                      key=lambda v: v["deadline"])
    past = sorted([v for v in value_races if not v["deadline"] or v["deadline"] <= now_hm],
                  key=lambda v: v["deadline"], reverse=True)
    ordered = upcoming + past

    if not ordered:
        st.info("本日の狙い目レースはまだありません。")
    else:
        for v in ordered:
            is_past = not (v["deadline"] and v["deadline"] > now_hm)
            sanren = v.get("sanren_tan")
            if isinstance(sanren, str):
                try:
                    sanren = json.loads(sanren)
                except Exception:
                    sanren = []
            combo_text = " / ".join(sanren) if sanren else ""
            odds = v.get("odds_value")
            odds_text = f"<span style='color:#0891b2;font-weight:700;margin-left:8px'>{odds:.1f}倍</span>" if odds else ""
            time_label = v["deadline"] or "--:--"
            opacity = "0.5" if is_past else "1"
            bg = "#161e30" if is_past else "#1a2540"
            border = "#2d3a52" if is_past else "#06b6d4"
            check = "✅" if is_past else "⏰"
            st.markdown(
                f"<div style='background:{bg};border:1px solid {border};border-radius:8px;"
                f"padding:0.5rem 0.9rem;margin:0.25rem 0;display:flex;align-items:center;"
                f"gap:0.8rem;flex-wrap:wrap;opacity:{opacity}'>"
                f"<span style='font-size:0.82rem;color:#155e75'>{check} {time_label}</span>"
                f"<span style='font-weight:700;color:#f1f5f9'>{v['venue_name']} {v['race_no']}R</span>"
                f"<span style='color:#1d4ed8;font-size:0.85rem'>{combo_text}</span>"
                f"{odds_text}"
                f"<span style='margin-left:auto;font-size:0.78rem;color:#64748b'>"
                f"確信度{v['top_score']:.2f} / 差{v['score_gap']:.2f}</span>"
                "</div>",
                unsafe_allow_html=True,
            )