"""
value race（狙い目レース）の回収率を再集計するスクリプト。

value raceの定義: score_gap <= 15 または top_score <= 30
（app.py / dashboard.py の狙い目判定と同じ基準）

8月中旬、value raceの確定件数が約1,000件に到達したタイミングで実行し、
サンプルが増えても回収率が高いままかを確認する想定。

使い方:
  python recompute_value_race_roi.py
"""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from supabase import create_client

VALUE_GAP_MAX = 15.0
VALUE_SCORE_MAX = 30.0
COST_PER_RACE = 300  # 3連単3点 x 100円
TARGET_SAMPLE_SIZE = 1000


def load_confirmed_records():
    supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    all_data = []
    offset = 0
    page_size = 1000
    while True:
        res = (
            supabase.table("prediction_records")
            .select("race_date,venue_code,race_no,hit,payout,top_score,score_gap")
            .not_.is_("hit", "null")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        if not res.data:
            break
        all_data.extend(res.data)
        if len(res.data) < page_size:
            break
        offset += page_size
    return all_data


def summarize(records, label):
    n = len(records)
    if n == 0:
        print(f"{label}: 0件")
        return
    hits = [r for r in records if r["hit"]]
    total_payout = sum(r.get("payout") or 0 for r in hits)
    total_cost = n * COST_PER_RACE
    roi = total_payout / total_cost * 100 if total_cost else 0
    hit_rate = len(hits) / n * 100
    print(f"{label}:")
    print(f"  件数        : {n:,}")
    print(f"  的中数/的中率: {len(hits):,} / {hit_rate:.1f}%")
    print(f"  総投資額    : ¥{total_cost:,}")
    print(f"  総払戻額    : ¥{total_payout:,}")
    print(f"  回収率      : {roi:.1f}%")


def main():
    records = load_confirmed_records()
    print(f"確定済み予想記録: {len(records):,}件\n")

    value_races = [
        r for r in records
        if r.get("top_score") is not None and r.get("score_gap") is not None
        and (r["score_gap"] <= VALUE_GAP_MAX or r["top_score"] <= VALUE_SCORE_MAX)
    ]

    summarize(records, "全体")
    print()
    summarize(value_races, f"value race (score_gap<={VALUE_GAP_MAX} or top_score<={VALUE_SCORE_MAX})")

    print()
    n_value = len(value_races)
    if n_value >= TARGET_SAMPLE_SIZE:
        print(f"✅ value race確定数が目標の{TARGET_SAMPLE_SIZE:,}件に到達しました（{n_value:,}件）。"
              "回収率の再評価に十分なサンプルサイズです。")
    else:
        remaining = TARGET_SAMPLE_SIZE - n_value
        print(f"⏳ value race確定数は{n_value:,}件（目標{TARGET_SAMPLE_SIZE:,}件まであと{remaining:,}件）。")


if __name__ == "__main__":
    main()
