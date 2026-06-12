#!/usr/bin/env python3
"""
競艇スクレイピング CLI

使い方:
  # 今日の桐生1Rの出走表を取得
  python main.py racelist --venue 01 --race 1

  # 今日の戸田3Rのオッズを取得
  python main.py odds --venue 02 --race 3

  # 昨日の平和島5Rの結果を取得
  python main.py result --date 20260521 --venue 04 --race 5

  # 今日の住之江の全レースを一括取得してJSONに保存
  python main.py all --venue 12 --out data/住之江_20260522.json

  # 会場コード一覧を表示
  python main.py venues
"""

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

from boatrace_scraper import BoatraceScraper, VENUE_MAP


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y%m%d").date()


def cmd_venues(_):
    print("\n▼ 会場コード一覧")
    print("-" * 30)
    for code, name in VENUE_MAP.items():
        print(f"  {code}: {name}")
    print()


def cmd_racelist(args):
    sc = BoatraceScraper(delay=args.delay)
    d = parse_date(args.date) if args.date else date.today()
    racers = sc.get_racelist(d, args.venue, args.race)

    if not racers:
        print("⚠ 出走表を取得できませんでした")
        return

    print(f"\n▼ {VENUE_MAP.get(args.venue.zfill(2), args.venue)} {args.race}R 出走表")
    print("-" * 80)
    header = f"{'枠':>3} {'選手名':<10} {'番号':>6} {'支部':<5} {'齢':>3} {'体重':>5} {'級':>3} {'全勝率':>6} {'当地':>6} {'モーター':>4}"
    print(header)
    print("-" * 80)
    for r in racers:
        print(
            f"{r.lane:>3} {r.name:<10} {r.racer_no:>6} {r.branch:<5} "
            f"{r.age or '-':>3} {r.weight or '-':>5} {r.rank:>3} "
            f"{r.win_rate_all or '-':>6} {r.local_win_rate or '-':>6} {r.motor_no or '-':>4}"
        )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        sc.save_racers_csv(racers, args.out)


def cmd_odds(args):
    sc = BoatraceScraper(delay=args.delay)
    d = parse_date(args.date) if args.date else date.today()
    odds = sc.get_odds(d, args.venue, args.race)

    print(f"\n▼ {odds.venue_name} {args.race}R オッズ")

    if odds.tansho:
        print("\n【単勝】")
        for k, v in odds.tansho.items():
            print(f"  {k}: {v}")

    if odds.niren_tan:
        print(f"\n【2連単】(上位10件)")
        top = sorted(odds.niren_tan.items(), key=lambda x: x[1] or 9999)[:10]
        for k, v in top:
            print(f"  {k}: {v}")

    if odds.sanren_tan:
        print(f"\n【3連単】(上位10件)")
        top = sorted(odds.sanren_tan.items(), key=lambda x: x[1] or 9999)[:10]
        for k, v in top:
            print(f"  {k}: {v}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        sc.save_odds_json(odds, args.out)


def cmd_result(args):
    sc = BoatraceScraper(delay=args.delay)
    d = parse_date(args.date) if args.date else date.today()
    result = sc.get_result(d, args.venue, args.race)

    if result is None:
        print("⚠ 結果を取得できませんでした（レース未終了の可能性があります）")
        return

    print(f"\n▼ {result.venue_name} {args.race}R 結果")
    print(f"  着順: {' → '.join(str(b) for b in result.arrival)}")
    if result.payouts:
        print("\n【払戻金】")
        for k, v in result.payouts.items():
            print(f"  {k}: {v:,}円")
    if result.start_times:
        print("\n【スタートタイム】")
        for lane, st in result.start_times.items():
            print(f"  {lane}号艇: {st}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        sc.save_result_json(result, args.out)


def cmd_all(args):
    sc = BoatraceScraper(delay=args.delay)
    d = parse_date(args.date) if args.date else date.today()
    races = sc.get_all_races(d, args.venue, max_race=args.max_race)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        sc.save_all_json(races, args.out)
        print(f"\n✅ {len(races)}レース分のデータを {args.out} に保存しました")
    else:
        print(json.dumps(races, ensure_ascii=False, indent=2))


# ─────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="競艇（BOAT RACE）スクレイピングツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--delay", type=float, default=1.5,
                        help="リクエスト間隔(秒) デフォルト: 1.5")

    sub = parser.add_subparsers(dest="cmd", required=True)

    # venues
    sub.add_parser("venues", help="会場コード一覧を表示")

    # 共通引数を持つサブコマンド
    def add_common(p):
        p.add_argument("--date", help="開催日 YYYYMMDD (省略時=今日)")
        p.add_argument("--venue", required=True, help="会場コード 例: 01=桐生")
        p.add_argument("--race", type=int, required=True, help="レース番号 1〜12")
        p.add_argument("--out", help="出力ファイルパス (.csv または .json)")

    p_racelist = sub.add_parser("racelist", help="出走表・選手情報を取得")
    add_common(p_racelist)

    p_odds = sub.add_parser("odds", help="オッズを取得")
    add_common(p_odds)

    p_result = sub.add_parser("result", help="レース結果を取得")
    add_common(p_result)

    p_all = sub.add_parser("all", help="全レースを一括取得")
    p_all.add_argument("--date", help="開催日 YYYYMMDD (省略時=今日)")
    p_all.add_argument("--venue", required=True, help="会場コード")
    p_all.add_argument("--max-race", type=int, default=12, help="最大レース数 (デフォルト: 12)")
    p_all.add_argument("--out", help="出力JSONファイルパス")
    p_all.add_argument("--delay", type=float, default=1.5)

    args = parser.parse_args()

    dispatch = {
        "venues": cmd_venues,
        "racelist": cmd_racelist,
        "odds": cmd_odds,
        "result": cmd_result,
        "all": cmd_all,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
