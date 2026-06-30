"""
prediction_records から展示タイム・進入・気象をCSVに書き出す。
build_dataset.py がこれを読んで学習データに結合する。
出力: dataset/exhibition_data.csv（race_date,venue_code,race_no,lane,exhibition_time,start_course,start_st）
"""
import os
import json
import csv
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from supabase import create_client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_all():
    rows = []
    page_size = 1000
    offset = 0
    while True:
        res = (supabase.table("prediction_records")
               .select("race_date,venue_code,race_no,exhibition_times,start_courses,start_sts,weather,wind_speed,wave_height,water_temp,wind_direction")
               .range(offset, offset + page_size - 1).execute())
        if not res.data:
            break
        rows.extend(res.data)
        if len(res.data) < page_size:
            break
        offset += page_size
    return rows


def main():
    rows = fetch_all()
    out_path = Path("dataset/exhibition_data.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "race_date", "venue_code", "race_no", "lane",
            "exhibition_time", "start_course", "start_st",
            "weather", "wind_speed", "wave_height", "water_temp", "wind_direction",
        ])
        for r in rows:
            ex = r.get("exhibition_times")
            if isinstance(ex, str):
                try:
                    ex = json.loads(ex)
                except Exception:
                    ex = {}
            ex = ex or {}

            sc = r.get("start_courses")
            if isinstance(sc, str):
                try:
                    sc = json.loads(sc)
                except Exception:
                    sc = {}
            sc = sc or {}

            sts = r.get("start_sts")
            if isinstance(sts, str):
                try:
                    sts = json.loads(sts)
                except Exception:
                    sts = {}
            sts = sts or {}

            # 展示タイムがある艇だけ行にする
            for lane_str, et in ex.items():
                lane = int(lane_str)
                writer.writerow([
                    r["race_date"], r["venue_code"], r["race_no"], lane,
                    et,
                    sc.get(lane_str, ""),
                    sts.get(lane_str, ""),
                    r.get("weather", ""),
                    r.get("wind_speed", ""),
                    r.get("wave_height", ""),
                    r.get("water_temp", ""),
                    r.get("wind_direction", ""),
                ])
                count += 1

    print(f"書き出し完了: {out_path} ({count}行)")


if __name__ == "__main__":
    main()