"""
新モデル(model_rank.pkl)で過去レースを再スコアリングし、
新スケールでの狙い目レースしきい値を検証するスクリプト。

方法:
  1. dataset/training_data.csv を train_model.py と全く同じパイプライン
     (race_no zero-padding, add_racer_course_stats, add_venue_lane_stats,
     build_features)で処理し、リーク無く各レースの新top_score/score_gapを算出する。
     このパイプラインはリーク防止のshift(1)を使っているため、
     各レースの評価には未来の情報を使っていない。
  2. Supabaseのprediction_recordsの実績(hit/payout/actual)を
     (race_date, venue_code, race_no)で結合する。
  3. 様々なしきい値候補で「そのレースを買った場合の回収率」を集計する。

注意（方法論上の限界）:
  prediction_recordsのhit/payoutは「旧モデルが選んだ3連単3点」を実際に買った場合の
  結果であり、新モデルが選ぶ3点とは一致しないことがある(過去オッズデータが無いため、
  新モデルが選ぶ任意の組み合わせの正確な払戻は再現できない)。
  ここでは「新モデルの確信度(score_gap/top_score)でレースを絞り込む」効果を、
  旧モデルの実績を報酬シグナルとして評価している。過去オッズ取得(タスク8)が
  完了すれば、新モデル自身が選ぶ組み合わせで正確な回収率検証ができるようになる。

使い方:
  python re_evaluate_value_race_threshold.py
"""
import os
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
COST_PER_RACE = 300  # 3連単3点 x 100円


def load_prediction_records():
    from supabase import create_client
    supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    all_data = []
    offset = 0
    while True:
        res = (
            supabase.table("prediction_records")
            .select("race_date,venue_code,race_no,hit,payout,actual")
            .not_.is_("hit", "null")
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
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig", low_memory=False)
    df = df[df["finish"].notna()].copy()
    df["race_date"] = df["race_date"].astype(str)
    # venue_codeはfloat64由来で"10.0"のような文字列になるため、結合キー用に
    # "10"のような2桁ゼロ埋め表記(Supabase側と同じ形式)へ正規化した列を別途持つ
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

    records = []
    for race_id, g in df.groupby("race_id"):
        g = g.sort_values("score", ascending=False)
        top_score = float(g["score"].iloc[0])
        score_gap = float(g["score"].iloc[0] - g["score"].iloc[1])
        row0 = g.iloc[0]
        records.append({
            "race_date": row0["race_date"], "venue_code": row0["venue_code_clean"],
            "race_no": int(row0["race_no_num"]),
            "new_top_score": round(top_score, 4), "new_score_gap": round(score_gap, 4),
        })
    return pd.DataFrame(records)


def summarize(sub, label):
    n = len(sub)
    if n == 0:
        print(f"{label:40s} n=0")
        return
    hits = sub[sub["hit"] == True]
    total_payout = hits["payout"].fillna(0).sum()
    total_cost = n * COST_PER_RACE
    roi = total_payout / total_cost * 100 if total_cost else 0
    hit_rate = len(hits) / n * 100
    print(f"{label:40s} n={n:5,d}  的中率={hit_rate:5.1f}%  "
          f"投資={total_cost:9,d}円  払戻={int(total_payout):9,d}円  回収率={roi:6.1f}%")


def main():
    print("training_data.csvを再スコアリング中...")
    rescored = rescore_training_data()
    print(f"再スコアリング済みレース数: {len(rescored):,}")

    print("Supabaseからprediction_recordsを取得中...")
    pred_df = load_prediction_records()
    print(f"確定済みprediction_records: {len(pred_df):,}")

    merged = pred_df.merge(rescored, on=["race_date", "venue_code", "race_no"], how="inner")
    print(f"結合できたレース数: {len(merged):,} "
          f"(training_data.csvに無い直近日付分は結合できません)")

    print("\n=== 全体(ベースライン) ===")
    summarize(merged, "全体")

    print("\n=== score_gap候補しきい値ごと ===")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        sub = merged[merged["new_score_gap"] <= thresh]
        summarize(sub, f"score_gap <= {thresh}")

    print("\n=== top_score候補しきい値ごと ===")
    for thresh in [0.6, 0.8, 1.0, 1.2, 1.4]:
        sub = merged[merged["new_top_score"] <= thresh]
        summarize(sub, f"top_score <= {thresh}")

    print("\n=== 組み合わせ(gap<=X or top<=Y)候補 ===")
    combos = [
        (0.4, 0.8), (0.5, 0.9), (0.6, 0.98), (0.6, 1.0), (0.7, 1.1), (0.8, 1.2),
    ]
    for gap_t, top_t in combos:
        sub = merged[(merged["new_score_gap"] <= gap_t) | (merged["new_top_score"] <= top_t)]
        summarize(sub, f"gap<={gap_t} or top<={top_t}")


if __name__ == "__main__":
    main()
