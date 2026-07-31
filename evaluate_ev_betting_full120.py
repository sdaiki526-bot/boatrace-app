"""
期待値ベースの3連単買い方の最終検証（全120通り版）。

前回(evaluate_ev_betting.py)は「モデルが選んだ3点」に限定したEVフィルタで
回収率74%が上限だった。今回はその限定を外し、各レースの全120通りの3連単
から期待値プラスの買い目を選ぶ、理論的に正しい形で検証する。

方法:
  1. dataset/training_data.csv を train_model.py と同じリーク無しパイプラインで
     処理し、model_rank.pklで各レース・各艇の生スコアを算出する。
  2. fetch_full_odds_cache.py が事前に取得したfull_odds_cache.json
     (対象レースの3連単全120通りの実オッズ)を読み込む。
  3. Plackett-Luce sequential eliminationで全120通りそれぞれの予測確率を計算する。
  4. 期待値 = 予測確率 × 実オッズ。しきい値を超える買い目を全て100円買ったと仮定。
  5. 実際の着順(actual)と一致する買い目だけに払戻(そのオッズ×100)を計上する。

オッズの時点について（重要・修正済み）:
  当初は「決着後のoddstop3tページは締切時点の確定オッズに近いはず」という前提で
  odds3tの値をそのまま払戻計算にも使っていたが、61件の独立クロスチェックで
  odds3t由来の払戻が公式確定払戻の平均6.9倍(中央値5.2倍、最大24.3倍)に
  達することが判明した。odds3tページは過去レースの確定オッズを正確に
  表示しないため、勝ち組み合わせの払戻は公式結果ページ(raceresult)から
  fetch_official_payouts.pyで別途取得した正確な値に置き換えている。
  一方、EV(期待値)によるフィルタ判定=どの買い目を「買う」と判断するかは、
  全120通り分の公式オッズを取得する手段が無いため、引き続きodds3t値を
  保守的に使わざるを得ない。この判定用オッズが不正確であるという限界は残る。

厳格なルール:
  - payoutは公式確定払戻を、actualと完全一致する買い目にのみ計上。それ以外は0。
  - 最高配当1本を除いた回収率を必ず併記する。
  - 1レースあたりの平均購入点数を必ず報告する(点数が多すぎれば非現実的)。

使い方:
  python evaluate_ev_betting_full120.py
"""
import json
import math
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import pandas as pd
import pickle

from train_model import RACE_KEY_COLS, add_racer_course_stats, add_venue_lane_stats, build_features

CSV_PATH = "dataset/training_data.csv"
MODEL_DIR = "models"
ODDS_CACHE_PATH = "full_odds_cache.json"
STAKE = 100


def rescore_training_data():
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig", low_memory=False)
    df = df[df["finish"].notna()].copy()
    df["race_date"] = df["race_date"].astype(str)
    df["venue_code_clean"] = df["venue_code"].astype(int).astype(str).str.zfill(2)
    df["venue_code"] = df["venue_code"].astype(str)
    df["race_no_num"] = df["race_no"].astype(int)
    df["race_no"] = df["race_no_num"].astype(str).str.zfill(2)
    df["race_id"] = df[RACE_KEY_COLS].agg("_".join, axis=1)
    df = df.sort_values(["race_date", "venue_code", "race_no", "lane"]).reset_index(drop=True)
    df = add_racer_course_stats(df)
    df = add_venue_lane_stats(df)
    feats = build_features(df)

    with open(f"{MODEL_DIR}/model_rank.pkl", "rb") as f:
        model = pickle.load(f)

    X = df[feats].copy()
    X["lane"] = X["lane"].astype("category")
    df["score"] = model.predict(X)

    race_scores, race_actual = {}, {}
    for race_id, g in df.groupby("race_id"):
        row0 = g.iloc[0]
        key = (row0["race_date"], row0["venue_code_clean"], int(row0["race_no_num"]))
        race_scores[key] = dict(zip(g["lane"].astype(int), g["score"].astype(float)))
        g_sorted = g.sort_values("lane")
        finish_map = {int(row["lane"]): row["finish"] for _, row in g_sorted.iterrows()}
        top3 = sorted(finish_map, key=lambda lane: finish_map[lane])[:3]
        if all(finish_map[lane] is not None and finish_map[lane] <= 3 for lane in top3):
            race_actual[key] = "-".join(str(lane) for lane in top3)
    return race_scores, race_actual


def all_combo_probabilities(lane_scores: dict) -> dict:
    """6艇の生スコアから全120通り(3連単)の予測確率を計算する"""
    from itertools import permutations
    weights = {lane: math.exp(s) for lane, s in lane_scores.items()}
    lanes = list(weights.keys())
    probs = {}
    for order in permutations(lanes, 3):
        remaining = dict(weights)
        prob = 1.0
        ok = True
        for lane in order:
            total = sum(remaining.values())
            if total <= 0:
                ok = False
                break
            prob *= remaining[lane] / total
            del remaining[lane]
        if ok:
            probs["-".join(str(x) for x in order)] = prob
    return probs


def main():
    print("training_data.csvを再スコアリング中(各艇の生スコア・実際の着順を算出)...")
    race_scores, race_actual = rescore_training_data()
    print(f"再スコアリング済みレース数: {len(race_scores):,}")

    print(f"{ODDS_CACHE_PATH} を読み込み中...")
    odds_cache = json.loads(open(ODDS_CACHE_PATH, encoding="utf-8").read())
    print(f"オッズキャッシュ済みレース数: {len(odds_cache):,}")

    # odds3tページの過去レースオッズは実際の確定払戻と大きく乖離する(平均6.9倍、
    # 61件のクロスチェックで確認済み)ため、勝ち組み合わせの払戻は公式結果ページ
    # (raceresult)から取得した正確な値で置き換える。EVのフィルタ判定(=どの買い目を
    # 買うかの選択)には、他に手段が無いためodds3t値をそのまま保守的に使う。
    official_path = Path("official_payouts_cache.json")
    official_cache = json.loads(official_path.read_text(encoding="utf-8")) if official_path.exists() else {}
    print(f"公式払戻キャッシュ: {len(official_cache):,}件")

    bets = []
    n_races_used = 0
    n_skipped_no_official = 0
    for cache_key, odds_map in odds_cache.items():
        race_date, venue_code, race_no = cache_key.split("_")
        key = (race_date, venue_code, int(race_no))
        lane_scores = race_scores.get(key)
        actual = race_actual.get(key)
        if lane_scores is None or actual is None or len(lane_scores) != 6:
            continue
        if len(odds_map) < 100:
            continue
        official = official_cache.get(cache_key)
        if not official or official.get("payout") is None:
            n_skipped_no_official = n_skipped_no_official + 1
            continue
        if official.get("actual") != actual:
            # training_data.csv側の着順再構成と公式結果が食い違うレースは信頼できないため除外
            n_skipped_no_official = n_skipped_no_official + 1
            continue
        official_payout = float(official["payout"])

        probs = all_combo_probabilities(lane_scores)
        n_races_used += 1
        for combo, prob in probs.items():
            odds = odds_map.get(combo)
            if odds is None:
                continue
            ev = prob * float(odds)  # EV判定は保守的にodds3t値を使用(限界として明記)
            is_winner = (combo == actual)
            payout_yen = official_payout if is_winner else 0  # 払戻は公式確定値を使用
            bets.append({
                "race_date": race_date, "venue_code": venue_code, "race_no": race_no,
                "combo": combo, "pred_prob": prob, "odds": float(odds), "ev": ev,
                "is_winner": is_winner, "payout": payout_yen,
            })

    bets_df = pd.DataFrame(bets)
    print(f"評価に使えたレース数: {n_races_used:,} / {len(odds_cache):,} "
          f"(公式払戻が無い/着順不一致で除外: {n_skipped_no_official:,})")
    print(f"評価対象の買い目総数(1レース最大120通り): {len(bets_df):,}")

    def summarize(sub, label, n_races_total):
        n = len(sub)
        n_races_bet = sub[["race_date", "venue_code", "race_no"]].drop_duplicates().shape[0] if n else 0
        avg_per_race = n / n_races_total if n_races_total else 0
        if n == 0:
            print(f"{label:16s} n_bets=     0  対象レース={n_races_total}")
            return
        cost = n * STAKE
        payout = sub["payout"].sum()
        roi = payout / cost * 100
        hit_rate = sub["is_winner"].mean() * 100
        if sub["payout"].max() > 0:
            sub_wo_max = sub.drop(sub["payout"].idxmax())
            roi_wo_max = sub_wo_max["payout"].sum() / (len(sub_wo_max) * STAKE) * 100 if len(sub_wo_max) else 0
            max_payout = sub["payout"].max()
        else:
            roi_wo_max = roi
            max_payout = 0
        print(f"{label:16s} n_bets={n:6,d}  平均購入点数/レース={avg_per_race:5.1f}  "
              f"的中率={hit_rate:5.2f}%  投資={cost:9,d}円  払戻={int(payout):9,d}円  "
              f"回収率={roi:6.1f}%  (最高配当{int(max_payout):,}円除外時: {roi_wo_max:6.1f}%)")

    print(f"\n=== 全買い目(フィルタなしベースライン、対象{n_races_used}レース) ===")
    summarize(bets_df, "全買い目", n_races_used)

    print("\n=== 期待値(EV)しきい値ごと ===")
    for thresh in [1.0, 1.1, 1.2, 1.3, 1.5, 2.0]:
        sub = bets_df[bets_df["ev"] > thresh]
        summarize(sub, f"EV > {thresh}", n_races_used)

    out_path = "ev_betting_full120_result.csv"
    bets_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n買い目ごとの詳細データを {out_path} に保存しました")


if __name__ == "__main__":
    main()
