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

# BETボタンの仮想投資額。app.pyのBET_AMOUNT_PER_RACEと揃える(3点×500円=1500円)。
BET_AMOUNT_PER_RACE = 1500

# ─────────────────────────────────────────────
# 艇番カラー（競艇公式の艇番カラー。app.pyのLANE_COLORSと同じ内容を維持すること。
# dashboard.pyからapp.pyを逆importできないため、同じ定義をここにも置いている）
# ─────────────────────────────────────────────
LANE_COLORS = {
    1: ("#ffffff", "#1e293b", "#94a3b8"),  # 白（枠線グレー・黒文字）
    2: ("#111827", "#ffffff", "#111827"),  # 黒
    3: ("#dc2626", "#ffffff", "#dc2626"),  # 赤
    4: ("#2563eb", "#ffffff", "#2563eb"),  # 青
    5: ("#eab308", "#1e293b", "#eab308"),  # 黄（黒文字）
    6: ("#16a34a", "#ffffff", "#16a34a"),  # 緑
}

def lane_color(n):
    """艇番(1-6) -> (背景色, 文字色) を返す。範囲外はグレーにフォールバック。"""
    bg, fg, _ = LANE_COLORS.get(int(n), ("#94a3b8", "#ffffff", "#94a3b8"))
    return bg, fg

def lane_badge_html(n, size="1.4rem"):
    """艇番1つを公式カラーの丸バッジHTMLにする。"""
    bg, fg, border = LANE_COLORS.get(int(n), ("#94a3b8", "#ffffff", "#94a3b8"))
    return (
        f"<span style='display:inline-flex;align-items:center;justify-content:center;"
        f"width:{size};height:{size};border-radius:50%;background:{bg};color:{fg};"
        f"border:1px solid {border};font-weight:800;font-size:0.75rem;line-height:1'>{n}</span>"
    )

def combos_badges_html(combos, size="1.4rem"):
    """買い目複数点(またはハイフン区切りの単一組み合わせ)を艇番バッジ列にする。"""
    if isinstance(combos, str):
        combos = [combos]
    items = []
    for c in combos:
        parts = str(c).split("-")
        badges = "".join(lane_badge_html(p, size=size) for p in parts if p.strip().isdigit())
        items.append(f"<span style='display:inline-flex;align-items:center;gap:3px'>{badges}</span>")
    sep = "<span style='color:#94a3b8;margin:0 0.3rem'>/</span>"
    return sep.join(items)


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


def render_dashboard(supabase, today_str, save_bet_record_fn=None, load_bet_records_fn=None):
    """
    会場グリッド型ダッシュボードを描画する。

    save_bet_record_fn / load_bet_records_fn: app.pyのsave_bet_record/load_bet_records
    を呼び出し元から渡してもらう（dashboard.pyからapp.pyを逆importすると循環importに
    なるため、関数を引数として受け取る形にしている）。未指定時はBETボタンを表示しない。
    """
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
                bg, border = "#e0f2fe", "#3b82f6"
                badge, badge_color = "⏰ まもなく", "#1d4ed8"
            elif v["value_count"] > 0:
                bg, border = "#f0f9ff", "#0891b2"
                badge, badge_color = f"🔥 狙い{v['value_count']}", "#0891b2"
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
        "<p style='color:#64748b;font-size:0.72rem;margin-top:0.5rem'>"
        "枠の色: <span style='color:#0891b2'>水色=狙い目あり</span> / "
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
        # 本日すでにBET済みのレースを(venue_code, race_no)で引けるようにする
        today_bet_map = {}
        if load_bet_records_fn:
            today_bet_map = {
                (b["venue_code"], b["race_no"]): b
                for b in load_bet_records_fn()
                if b.get("race_date") == today_str
            }

        for v in ordered:
            is_past = not (v["deadline"] and v["deadline"] > now_hm)
            sanren = v.get("sanren_tan")
            if isinstance(sanren, str):
                try:
                    sanren = json.loads(sanren)
                except Exception:
                    sanren = []
            combo_text = combos_badges_html(sanren) if sanren else ""
            odds = v.get("odds_value")
            odds_text = f"<span style='color:#0891b2;font-weight:700;margin-left:8px'>{odds:.1f}倍</span>" if odds else ""
            time_label = v["deadline"] or "--:--"
            opacity = "0.6" if is_past else "1"
            bg = "#f8fafc" if is_past else "#f0f9ff"
            border = "#e2e8f0" if is_past else "#06b6d4"
            check = "✅" if is_past else "⏰"
            st.markdown(
                f"<div style='background:{bg};border:1px solid {border};border-radius:8px;"
                f"padding:0.5rem 0.9rem;margin:0.25rem 0;display:flex;align-items:center;"
                f"gap:0.8rem;flex-wrap:wrap;opacity:{opacity}'>"
                f"<span style='font-size:0.82rem;color:#155e75'>{check} {time_label}</span>"
                f"<span style='font-weight:700;color:#1e293b'>{v['venue_name']} {v['race_no']}R</span>"
                f"<span style='color:#1d4ed8;font-size:0.85rem'>{combo_text}</span>"
                f"{odds_text}"
                f"<span style='margin-left:auto;font-size:0.78rem;color:#64748b'>"
                f"確信度{v['top_score']:.2f} / 差{v['score_gap']:.2f}</span>"
                "</div>",
                unsafe_allow_html=True,
            )

            if save_bet_record_fn:
                existing_bet = today_bet_map.get((v["venue_code"], v["race_no"]))
                dash_bet_key = f"dash_bet_{today_str}_{v['venue_code']}_{v['race_no']}"
                if existing_bet:
                    st.button(
                        f"✅ BET済み（¥{existing_bet.get('bet_amount', 0):,}）",
                        key=dash_bet_key, use_container_width=True, disabled=True,
                    )
                elif st.button("🎯 BET（3点×500円）", key=dash_bet_key, use_container_width=True):
                    ok = save_bet_record_fn({
                        "race_date": today_str,
                        "venue_code": v["venue_code"],
                        "venue_name": v["venue_name"],
                        "race_no": v["race_no"],
                        "sanren_tan": sanren,
                        "bet_amount": BET_AMOUNT_PER_RACE,
                        "top_score": v["top_score"],
                        "score_gap": v["score_gap"],
                    })
                    if ok:
                        st.success(f"BETしました（{v['venue_name']} {v['race_no']}R　¥{BET_AMOUNT_PER_RACE:,}）")
                        st.rerun()