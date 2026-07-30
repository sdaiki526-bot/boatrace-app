"""
期待値ベースの3連単買い方が回収率100%を超えられるか検証するスクリプト。

方法:
  1. dataset/training_data.csv を train_model.py と同じリーク無しパイプライン
     (race_no zero-padding, add_racer_course_stats, add_venue_lane_stats,
     build_features)で処理し、model_rank.pklで各レース・各艇の生スコアを算出する。
     このパイプラインはリーク防止のshift(1)を使っているため、
     各レースの評価には未来の情報を使っていない。
  2. Supabaseのprediction_recordsから、sanren_tan(買い目3点)・sanren_tan_odds
     (予想時点のオッズ)・hit/actual/payout(実際の結果)が揃ったレコードを取得する。
  3. 各レースの6艇の生スコアから、Plackett-Luce流のsequential eliminationで
     「買い目3点それぞれ」が実現する確率を計算する:
       P(1着=a, 2着=b, 3着=c)
         = [w_a / Σw_i] * [w_b / Σ(w_i, i≠a)] * [w_c / Σ(w_i, i≠a,b)]
       w_i = exp(score_i) をPlackett-Luceの「強さ」とみなす簡易的な近似。
       (lambdarankはNDCG最適化であり厳密なPlackett-Luce尤度で学習されて
       いないため、これはあくまで近似である点に注意。)
  4. 期待値 = 予測確率 × sanren_tan_odds(予想時点オッズ)。
  5. 期待値がしきい値を超える買い目「だけ」を100円買ったと仮定し、
     実際の結果(actual文字列と一致するか)・実際の払戻(公式payout)で
     回収率を計算する。

重要な注意（ユーザー指定の検証ルール）:
  - payoutは「実際にactualと一致した買い目にのみ」計上する。一致しない買い目は
    払戻0として扱う(過去に規模水増しのバグがあったための明示的な対策)。
  - EV計算に使うsanren_tan_oddsは予想時点のオッズであり、実際の締切時オッズとは
    異なりうる(楽観バイアスの可能性)。ROI計算に使うpayoutは実際の確定払戻
    (公式スクレイピング値)であり、この2つの情報源の違いを明記する。
  - サンプル数が少ないしきい値は、点数を併記し信頼性の低さを明示する。
  - 最高配当1本を除いた回収率も併記し、少数の高配当への依存を確認する。

使い方:
  python evaluate_ev_betting.py
"""
import os
import json
import math
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import pickle

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from train_model import RACE_KEY_COLS, add_racer_course_stats, add_venue_lane_stats, build_features

CSV_PATH = "dataset/training_data.csv"
MODEL_DIR = "models"
STAKE = 100  # 3連単1点100円


def load_prediction_records():
    from supabase import create_client
    supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    all_data = []
    offset = 0
    while True:
        res = (
            supabase.table("prediction_records")
            .select("race_date,venue_code,race_no,sanren_tan,sanren_tan_odds,hit,actual,payout")
            .not_.is_("hit", "null")
            .not_.is_("sanren_tan_odds", "null")
            .range(offset, offset + 999)
            .execute()
        )
        if not res.data:
            break
        all_data.extend(res.data)
        if len(res.data) < 1000:
            break
        offset += 1000
    return pd.DataFrame(all_data)


def rescore_training_data():
    """各レースの6艇分の生スコアを {race_date,venue_code,race_no: {lane: score}} で返す"""
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

    race_scores = {}
    for race_id, g in df.groupby("race_id"):
        row0 = g.iloc[0]
        key = (row0["race_date"], row0["venue_code_clean"], int(row0["race_no_num"]))
        race_scores[key] = dict(zip(g["lane"].astype(int), g["score"].astype(float)))
    return race_scores


def combo_probability(lane_scores: dict, combo: str) -> float:
    """Plackett-Luce sequential eliminationで3連単combo("a-b-c")の確率を計算する"""
    lanes = list(lane_scores.keys())
    weights = {lane: math.exp(lane_scores[lane]) for lane in lanes}
    order = [int(x) for x in combo.split("-")]
    if any(lane not in weights for lane in order):
        return None
    remaining = dict(weights)
    prob = 1.0
    for lane in order:
        total = sum(remaining.values())
        if total <= 0:
            return None
        prob *= remaining[lane] / total
        del remaining[lane]
    return prob


def main():
    print("training_data.csvを再スコアリング中(各艇の生スコアを算出)...")
    race_scores = rescore_training_data()
    print(f"再スコアリング済みレース数: {len(race_scores):,}")

    print("Supabaseからprediction_records(オッズ・結果あり)を取得中...")
    pred_df = load_prediction_records()
    print(f"オッズ・結果が両方揃ったレコード数: {len(pred_df):,}")

    bets = []  # 1行 = 1買い目(100円)
    unmatched_race = 0
    for _, r in pred_df.iterrows():
        key = (str(r["race_date"]), str(r["venue_code"]), int(r["race_no"]))
        lane_scores = race_scores.get(key)
        if lane_scores is None or len(lane_scores) != 6:
            unmatched_race += 1
            continue
        odds_map = r["sanren_tan_odds"]
        if isinstance(odds_map, str):
            odds_map = json.loads(odds_map)
        actual = r.get("actual") or ""
        official_payout = r.get("payout") or 0
        for combo, odds in odds_map.items():
            if odds is None:
                continue
            p = combo_probability(lane_scores, combo)
            if p is None:
                continue
            ev = p * float(odds)
            is_winner = (combo == actual)
            payout_yen = official_payout if is_winner else 0  # 実際の公式払戻のみ計上
            bets.append({
                "race_date": key[0], "venue_code": key[1], "race_no": key[2],
                "combo": combo, "pred_prob": p, "pred_odds": float(odds), "ev": ev,
                "is_winner": is_winner, "payout": payout_yen,
            })

    bets_df = pd.DataFrame(bets)
    print(f"評価対象レース数(再スコアリングと結合できた): "
          f"{pred_df.shape[0] - unmatched_race:,} / {pred_df.shape[0]:,}")
    print(f"評価対象買い目数(1レース3点): {len(bets_df):,}")
    print(f"買い目全体の的中数: {bets_df['is_winner'].sum():,} "
          f"(的中率 {bets_df['is_winner'].mean()*100:.2f}%)")

    def summarize(sub, label):
        n = len(sub)
        if n == 0:
            print(f"{label:28s} n=    0")
            return
        cost = n * STAKE
        payout = sub["payout"].sum()
        roi = payout / cost * 100
        hit_rate = sub["is_winner"].mean() * 100
        # 最高配当を除いた回収率
        if sub["payout"].max() > 0:
            sub_wo_max = sub.drop(sub["payout"].idxmax())
            roi_wo_max = sub_wo_max["payout"].sum() / (len(sub_wo_max) * STAKE) * 100 if len(sub_wo_max) else 0
            max_payout = sub["payout"].max()
        else:
            roi_wo_max = roi
            max_payout = 0
        print(f"{label:28s} n={n:6,d}  的中率={hit_rate:5.2f}%  "
              f"投資={cost:9,d}円  払戻={int(payout):9,d}円  回収率={roi:6.1f}%  "
              f"(最高配当{int(max_payout):,}円除外時: {roi_wo_max:6.1f}%)")

    print("\n=== 全買い目(フィルタなしベースライン) ===")
    summarize(bets_df, "全買い目")

    print("\n=== 期待値(EV)しきい値ごと ===")
    for thresh in [1.0, 1.05, 1.1, 1.2, 1.3, 1.5]:
        sub = bets_df[bets_df["ev"] > thresh]
        summarize(sub, f"EV > {thresh}")

    out_path = "ev_betting_result.csv"
    bets_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n買い目ごとの詳細データを {out_path} に保存しました")


if __name__ == "__main__":
    main()
