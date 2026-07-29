"""
② モデル学習スクリプト (LightGBM / lambdarank版)

dataset/training_data.csv を読み込んでモデルを学習し models/ に保存する。

旧版との違い:
  - 艇ごとの独立二値分類 → レース単位のランキング学習(lambdarank)に変更
    各レース6艇を1グループとして「相対的にどの艇が上位に来るか」を学習するため、
    1コースの事前勝率(約55%)へ予測が張り付く偏りが大幅に緩和される。
  - 評価をレース単位の実戦指標(単勝的中率/3連対カバー率)に変更。
  - course_base_winrate を特徴量に追加。

使い方:
  python train_model.py
  python train_model.py --csv dataset/training_data.csv --out models/
"""

import argparse
import json
import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def log_metrics_to_supabase(metrics: dict, lane_mode: str, n_train_races: int,
                             n_features: int, recent_days: int):
    """モデル精度モニタリング用に、学習結果をSupabaseのmodel_metricsテーブルへ記録する。
    テーブルが無い/接続情報が無い場合は何もせず静かに続行する（学習自体は失敗させない）。"""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        return
    try:
        from supabase import create_client
        supabase = create_client(url, key)
        supabase.table("model_metrics").insert({
            "model_type": "lambdarank",
            "lane_mode": lane_mode,
            "hit_win": metrics["hit_win"],
            "cover3": metrics["cover3"],
            "n_train_races": n_train_races,
            "n_features": n_features,
            "recent_days": recent_days,
        }).execute()
        logger.info("学習結果をmodel_metricsに記録しました")
    except Exception as e:
        logger.warning(f"model_metricsへの記録に失敗（学習結果は保存済みなので続行）: {e}")


# ─────────────────────────────────────────────
# 特徴量カラム
# ─────────────────────────────────────────────
# コース情報(序列を固定化するので最小限に絞る)
BASE_COLS = [
    "lane",
    "age",
    "weight",
    "flying_count",
    "late_count",
    "rc_races",
]

# 選手・機力の「強さ」を表す指標。これらをレース内で相対化する。
# 生値そのものはコース絶対序列と相関しやすいので、相対値(偏差・順位)を主役にする。
RELATIVE_SRC_COLS = [
    "rank_num",
    "avg_start_time",
    "win_rate_all",
    "win_rate_2",
    "win_rate_3",
    "local_win_rate",
    "local_win_rate_2",
    "local_win_rate_3",
    "motor_2rate",
    "boat_2rate",
    "exhibition_time",
    "start_st",
    "start_course",
    "rc_win_rt",
    "rc_top3_rt",
    "vc_win_rt",
    "vc_top3_rt",
]

# rank_num と avg_start_time は「小さいほど強い」ため、相対化時に符号を反転する
LOWER_IS_BETTER = {"rank_num", "avg_start_time", "exhibition_time", "start_st", "start_course"}

# 最終的にモデルへ渡す特徴量は train 時に動的生成する(下記 build_features 参照)
FEATURE_COLS: list[str] = []

# レースを一意に識別するキー (race_id 列が無いため複合キーで代用)
RACE_KEY_COLS = ["race_date", "venue_code", "race_no"]

# lambdarank パラメータ
# 2026-07-29: Optunaで24試行チューニングし、baseline(hit_win=0.5689)を上回る
# 組み合わせ(hit_win=0.5709, cover3=0.8672)を採用。
LGB_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [1, 3],
    "learning_rate": 0.10697077067228637,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 38,
    "feature_fraction": 0.5136838498519677,
    "bagging_fraction": 0.7807874737135436,
    "bagging_freq": 2,
    "reg_alpha": 0.001362522051403353,
    "reg_lambda": 0.017822787430829164,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
    "label_gain": [0, 1, 3, 7, 15, 31],  # relevance 0..5 用のゲイン
}


def finish_to_relevance(finish: pd.Series) -> pd.Series:
    """着順(1..6)を relevance(1着=5 ... 6着=0)に変換。欠損/失格は0。"""
    rel = (6 - pd.to_numeric(finish, errors="coerce")).clip(lower=0, upper=5)
    return rel.fillna(0).astype(int)

def add_racer_course_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    選手×コースの過去成績を特徴量として追加する。
    データ漏洩を防ぐため、各レース時点で「それより前」の成績だけを使う（shiftで1つずらす）。

    追加される列:
      rc_races  : その選手がそのコースで過去に走った回数
      rc_win_rt : そのコースでの過去1着率
      rc_top3_rt: そのコースでの過去3連対率
    """
    df = df.sort_values(["race_date", "venue_code", "race_no"]).reset_index(drop=True)

    # 着順から勝ち/3連対フラグを作る
    finish = pd.to_numeric(df["finish"], errors="coerce")
    df["_is_win"] = (finish == 1).astype(float)
    df["_is_top3"] = (finish <= 3).astype(float)

    # 選手×コースでグループ化し、累積を「1つ前まで」で計算（shift(1)で自分を除外）
    grp = df.groupby(["racer_no", "lane"], sort=False)

    # 過去の走行回数（自分を含まない）
    df["rc_races"] = grp.cumcount()

    # 過去の勝利数・3連対数（自分を含まない = shift してから cumsum）
    df["rc_wins"] = grp["_is_win"].transform(lambda s: s.shift(1).cumsum())
    df["rc_top3s"] = grp["_is_top3"].transform(lambda s: s.shift(1).cumsum())

    # 率に変換（経験0回のときは NaN → 後で平均で埋まる）
    df["rc_win_rt"] = df["rc_wins"] / df["rc_races"].replace(0, np.nan)
    df["rc_top3_rt"] = df["rc_top3s"] / df["rc_races"].replace(0, np.nan)

    # 後片付け
    df = df.drop(columns=["_is_win", "_is_top3", "rc_wins", "rc_top3s"], errors="ignore")

    n_have = df["rc_win_rt"].notna().sum()
    logger.info(f"  選手×コース成績: {n_have:,} 行に付与（経験1走以上）")
    return df


def add_venue_lane_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    会場×コースの過去入着率を特徴量として追加する（会場ごとのコース有利不利の違いを学習させる）。
    データ漏洩を防ぐため、各レース時点で「それより前」の成績だけを使う（shiftで1つずらす）。

    追加される列:
      vc_win_rt : その会場・そのコースでの過去1着率
      vc_top3_rt: その会場・そのコースでの過去3連対率
    """
    df = df.sort_values(["race_date", "venue_code", "race_no"]).reset_index(drop=True)

    finish = pd.to_numeric(df["finish"], errors="coerce")
    df["_is_win"] = (finish == 1).astype(float)
    df["_is_top3"] = (finish <= 3).astype(float)

    # 会場×コースでグループ化し、累積を「1つ前まで」で計算（shift(1)で自分を除外）
    grp = df.groupby(["venue_code", "lane"], sort=False)

    vc_races = grp.cumcount()
    vc_wins = grp["_is_win"].transform(lambda s: s.shift(1).cumsum())
    vc_top3s = grp["_is_top3"].transform(lambda s: s.shift(1).cumsum())

    df["vc_win_rt"] = vc_wins / vc_races.replace(0, np.nan)
    df["vc_top3_rt"] = vc_top3s / vc_races.replace(0, np.nan)

    df = df.drop(columns=["_is_win", "_is_top3"], errors="ignore")

    n_have = df["vc_win_rt"].notna().sum()
    logger.info(f"  会場×コース成績: {n_have:,} 行に付与（経験1走以上）")
    return df


def compute_course_stats_snapshot(df: pd.DataFrame) -> dict:
    """
    選手×コースの「全履歴」過去成績スナップショットを作る（ライブ推論用）。

    add_racer_course_stats() は学習時のリーク防止のため shift(1) して
    「そのレースより前まで」の値を使うが、ライブ推論で予想する当日のレースは
    学習データより未来なので、シフト不要で全履歴をそのまま集計してよい。
    """
    racer_no_num = pd.to_numeric(df["racer_no"], errors="coerce")
    finish = pd.to_numeric(df["finish"], errors="coerce")
    tmp = pd.DataFrame({
        # CSV上のracer_noはfloat64（欠損混じり）で "5060.0" のような文字列になりやすいため、
        # 整数文字列（"5060"）に正規化してライブ推論側の選手番号と一致させる
        "racer_no": racer_no_num,
        "lane": df["lane"],
        "is_win": (finish == 1).astype(float),
        "is_top3": (finish <= 3).astype(float),
    })
    tmp = tmp[tmp["racer_no"].notna()].copy()
    tmp["racer_no"] = tmp["racer_no"].astype(int).astype(str)
    stats = tmp.groupby(["racer_no", "lane"]).agg(
        races=("is_win", "size"),
        win_rate=("is_win", "mean"),
        top3_rate=("is_top3", "mean"),
    )
    snapshot = {}
    for (racer_no, lane), row in stats.iterrows():
        snapshot[f"{racer_no}_{int(lane)}"] = {
            "races": int(row["races"]),
            "win_rate": round(float(row["win_rate"]), 4),
            "top3_rate": round(float(row["top3_rate"]), 4),
        }
    return snapshot

def build_features(df: pd.DataFrame) -> list[str]:
    """
    レース内相対化特徴量を df に追加し、最終的な特徴量カラム名のリストを返す。

    各「強さ指標」について、レース(race_id)内で次を生成する:
      - <col>_dev  : レース平均との差 (value - race_mean)。「このレースで平均よりどれだけ強いか」
      - <col>_rank : レース内順位を 0..1 に正規化。「6艇中の相対位置」
    rank_num / avg_start_time は小さいほど強いため符号を反転してから相対化する。

    生値(絶対値)はモデルに渡さず、相対値のみを渡すことで
    「コース番号の固定序列」に頼れないようにする。
    """
    feats = list(BASE_COLS)
    g = df.groupby("race_id", sort=False)

    for col in RELATIVE_SRC_COLS:
        src = pd.to_numeric(df[col], errors="coerce")
        if col in LOWER_IS_BETTER:
            src = -src  # 小さいほど強い → 符号反転で「大きいほど強い」に揃える
        src = src.fillna(src.mean())

        # 平均との差
        race_mean = src.groupby(df["race_id"]).transform("mean")
        df[f"{col}_dev"] = src - race_mean

        # レース内順位(大きいほど強い→ rank 高)。0..1 に正規化(艇数差を吸収)
        rnk = src.groupby(df["race_id"]).rank(method="average")
        cnt = src.groupby(df["race_id"]).transform("count")
        df[f"{col}_rank"] = (rnk - 1) / (cnt - 1).clip(lower=1)

        feats += [f"{col}_dev", f"{col}_rank"]

    return feats


class BoatraceModelTrainer:
    # ── データ読み込み ───────────────────────
    def __init__(self, csv_path: Path, model_dir: Path, recent_days: int = 0):
        self.csv_path = csv_path
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.feature_importance = None
        self.recent_days = recent_days

    def load_data(self) -> pd.DataFrame:
        logger.info(f"CSV読み込み: {self.csv_path}")
        df = pd.read_csv(self.csv_path, encoding="utf-8-sig")
        logger.info(f"  全行数: {len(df):,}")

        df = df[df["finish"].notna()].copy()
        logger.info(f"  着順あり: {len(df):,} 行")

        # レースキーを文字列結合して race_id を生成（recent_days絞り込みの前に実施）
        # race_noは2桁ゼロ埋めにする。ゼロ埋めしないと文字列ソートが
        # "1","10","11","12","2","3",...の順になり、同日内のレース順が壊れ
        # add_racer_course_stats/add_venue_lane_statsのshift(1)が同日の未来レースを
        # 「過去」として取り込んでしまう（データ漏洩）ため。
        df["race_date"] = df["race_date"].astype(str)
        df["venue_code"] = df["venue_code"].astype(str)
        df["race_no"] = df["race_no"].astype(int).astype(str).str.zfill(2)
        df["race_id"] = df[RACE_KEY_COLS].agg("_".join, axis=1)
        df = df.sort_values(["race_date", "venue_code", "race_no", "lane"]).reset_index(drop=True)

        # 選手×コースの「全履歴」スナップショット（recent_days絞り込みの影響を受けない。ライブ推論用）
        self.course_stats_snapshot = compute_course_stats_snapshot(df)

        # 直近N日だけに絞る（展示データの比率を上げるため）
        if self.recent_days and self.recent_days > 0:
            df["_rd"] = df["race_date"].astype(str)
            cutoff = df["_rd"].max()
            from datetime import datetime as _dt, timedelta as _tdelta
            try:
                cutoff_date = _dt.strptime(cutoff, "%Y%m%d") - _tdelta(days=self.recent_days)
                cutoff_str = cutoff_date.strftime("%Y%m%d")
                df = df[df["_rd"] >= cutoff_str].copy()
                logger.info(f"  直近{self.recent_days}日に絞り込み: {len(df):,} 行（{cutoff_str}以降）")
            except Exception as e:
                logger.warning(f"日付絞り込み失敗（全期間で続行）: {e}")
            df = df.drop(columns=["_rd"], errors="ignore")

        # 選手×コースの過去成績を追加（データ漏洩なし）
        df = add_racer_course_stats(df)

        # 会場×コースの過去入着率を追加（データ漏洩なし）
        df = add_venue_lane_stats(df)

        # レース内相対化特徴量を生成し、使用カラムを確定する
        global FEATURE_COLS
        FEATURE_COLS = build_features(df)
        logger.info(f"  特徴量数: {len(FEATURE_COLS)} (相対化後)")
        return df

    # ── レース単位で時系列分割 ──────────────────
    @staticmethod
    def split_by_race(df: pd.DataFrame, val_ratio: float = 0.2):
        """レース単位で末尾 val_ratio を検証に回す(レースが train/val をまたがない)。"""
        race_ids = df["race_id"].drop_duplicates().tolist()  # 既に時系列順
        n_val = max(1, int(len(race_ids) * val_ratio))
        val_ids = set(race_ids[-n_val:])
        is_val = df["race_id"].isin(val_ids)
        return df[~is_val].copy(), df[is_val].copy()

    @staticmethod
    def make_group(df: pd.DataFrame):
        """LightGBM ranking 用の group(各レースの行数リスト)。順序は df の並び通り。"""
        return df.groupby("race_id", sort=False).size().tolist()

    # ── 学習 ─────────────────────────────────
    def train(self, lane_mode: str = "categorical", save: bool = True, verbose: bool = True):
        """
        lane_mode:
          "numeric"     : lane を数値特徴として使用(従来。コース過剰依存になりやすい)
          "categorical" : lane をカテゴリ特徴として使用(連続的大小を断ち切る)
          "drop"        : lane を特徴量から除外(コース有利を完全に捨てる)
        戻り値: 診断用 dict
        """
        df = self.load_data()
        if len(df) < 500:
            print("⚠ データが少なすぎます(500行以上推奨)。クローラーでデータを集めてください。")
            return None

        # lane_mode に応じて使用する特徴量を決める
        feats = list(FEATURE_COLS)
        cat_features = []
        if lane_mode == "drop":
            feats = [c for c in feats if c != "lane"]
        elif lane_mode == "categorical":
            cat_features = ["lane"]
        # numeric は feats そのまま

        train_df, val_df = self.split_by_race(df, val_ratio=0.2)
        if verbose:
            logger.info(f"[lane_mode={lane_mode}] 学習: {len(train_df):,}行 / "
                        f"{train_df['race_id'].nunique():,}レース  検証: {len(val_df):,}行 / "
                        f"{val_df['race_id'].nunique():,}レース")

        X_train, X_val = train_df[feats].copy(), val_df[feats].copy()
        if lane_mode == "categorical":
            X_train["lane"] = X_train["lane"].astype("category")
            X_val["lane"]   = X_val["lane"].astype("category")

        rel_train = finish_to_relevance(train_df["finish"])
        rel_val   = finish_to_relevance(val_df["finish"])
        grp_train = self.make_group(train_df)
        grp_val   = self.make_group(val_df)

        if verbose:
            print(f"\n📊 ランキングモデル(lambdarank) 学習中... [lane_mode={lane_mode}]")
        dtrain = lgb.Dataset(X_train, label=rel_train, group=grp_train,
                             categorical_feature=cat_features or "auto")
        dval   = lgb.Dataset(X_val, label=rel_val, group=grp_val, reference=dtrain)

        self.model = lgb.train(
            LGB_PARAMS,
            dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(100 if verbose else 0)],
        )
        self.feature_names = feats

        val_df = val_df.copy()
        val_df["score"] = self.model.predict(X_val)
        metrics = self._evaluate(val_df, verbose=verbose)
        bias = self._diagnose_bias(val_df, verbose=verbose)

        imp = pd.DataFrame({
            "feature": feats,
            "importance": self.model.feature_importance(importance_type="gain"),
        }).sort_values("importance", ascending=False)
        self.feature_importance = imp
        if verbose:
            print("\n📈 特徴量重要度")
            print(imp.to_string(index=False))

        # lane への依存度(全gainに占める lane の割合)
        lane_share = 0.0
        if "lane" in imp["feature"].values and imp["importance"].sum() > 0:
            lane_share = float(imp.loc[imp["feature"] == "lane", "importance"].iloc[0]
                               / imp["importance"].sum())

        result = {
            "lane_mode": lane_mode,
            "hit_win": metrics["hit_win"],
            "cover3": metrics["cover3"],
            "pred_top1_lane1": bias["pred_top1_lane1"],
            "actual_win_lane1": bias["actual_win_lane1"],
            "lane_importance_share": round(lane_share, 4),
        }
        if save:
            self._save(metrics, lane_mode)
            log_metrics_to_supabase(
                metrics, lane_mode,
                n_train_races=train_df["race_id"].nunique(),
                n_features=len(feats),
                recent_days=self.recent_days,
            )
        return result

    # ── 実戦指標の評価 ───────────────────────
    @staticmethod
    def _evaluate(val_df: pd.DataFrame, verbose: bool = True) -> dict:
        # 各レースでスコア最大の艇を本命(単勝)とする
        idx_top1 = val_df.groupby("race_id")["score"].idxmax()
        top1 = val_df.loc[idx_top1]
        hit_win = (top1["finish"].astype(float) == 1).mean()  # 単勝的中率

        # 各レースでスコア上位3艇に実際の1着が含まれる率(3連系の本命カバー率)
        def top3_covers_winner(g):
            top3_lanes = g.nlargest(3, "score")["lane"].tolist()
            winner = g.loc[g["finish"].astype(float) == 1, "lane"]
            return (not winner.empty) and (winner.iloc[0] in top3_lanes)
        cover3 = val_df.groupby("race_id").apply(top3_covers_winner).mean()

        if verbose:
            print(f"\n  単勝的中率(本命=スコア最大): {hit_win:.4f}")
            print(f"  上位3艇に1着を含む率      : {cover3:.4f}")
            print("\n  目安: 単勝的中率はコース1号艇ベタ張り(約0.50前後)を上回れば学習効果あり")
        return {"hit_win": round(float(hit_win), 4), "cover3": round(float(cover3), 4)}

    # ── 1コース偏りの診断 ────────────────────
    @staticmethod
    def _diagnose_bias(val_df: pd.DataFrame, verbose: bool = True) -> dict:
        # 各レースで予測1位になった艇のコース分布
        idx_top1 = val_df.groupby("race_id")["score"].idxmax()
        pred1_lane = val_df.loc[idx_top1, "lane"]
        pred_dist = pred1_lane.value_counts(normalize=True).sort_index()

        # 実際の1着のコース分布
        winners = val_df[val_df["finish"].astype(float) == 1]
        actual_dist = winners["lane"].value_counts(normalize=True).sort_index()

        comp = pd.DataFrame({
            "pred_top1_rate": pred_dist,
            "actual_win_rate": actual_dist,
        }).fillna(0).round(3)
        if verbose:
            print("\n🔍 1コース偏り診断 (コース別)")
            print(comp.to_string())
            print("  → pred と actual が近ければ健全。pred が極端に lane=1 に寄るなら要対策。")
        return {
            "pred_top1_lane1": round(float(pred_dist.get(1, 0.0)), 3),
            "actual_win_lane1": round(float(actual_dist.get(1, 0.0)), 3),
        }

    # ── 保存 ─────────────────────────────────
    def _save(self, metrics: dict, lane_mode: str):
        model_path = self.model_dir / "model_rank.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        meta = {
            "model_type": "lambdarank",
            "lane_mode": lane_mode,
            "feature_cols": getattr(self, "feature_names", FEATURE_COLS),
            "race_key_cols": RACE_KEY_COLS,
            "metrics": metrics,
            "lgb_params": LGB_PARAMS,
        }
        (self.model_dir / "model_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        if self.feature_importance is not None:
            self.feature_importance.to_csv(
                self.model_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")

        snapshot = getattr(self, "course_stats_snapshot", None)
        if snapshot is not None:
            (self.model_dir / "course_stats.json").write_text(
                json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")
            print(f"   選手×コース成績スナップショット: {len(snapshot):,}件"
                  f"({self.model_dir / 'course_stats.json'})")

        print(f"\n✅ モデル保存完了! (lane_mode={lane_mode})")
        print(f"   ランキングモデル: {model_path.resolve()}")
        print(f"   単勝的中率={metrics['hit_win']}  上位3カバー率={metrics['cover3']}")


def main():
    parser = argparse.ArgumentParser(description="競艇予想モデル学習 (lambdarank版)")
    parser.add_argument("--csv", default="dataset/training_data.csv")
    parser.add_argument("--out", default="models/")
    parser.add_argument("--lane-mode", default="categorical",
                        choices=["numeric", "categorical", "drop"],
                        help="laneの扱い(単一学習時)")
    parser.add_argument("--compare", action="store_true",
                        help="numeric/categorical/drop の3モードを比較する")
    parser.add_argument("--recent-days", type=int, default=0,
                        help="直近N日だけで学習（0=全期間）")
    args = parser.parse_args()

    if args.compare:
        # 3モードを静かに学習して比較表を出し、最良モードを保存する
        results = []
        for mode in ["numeric", "categorical", "drop"]:
            trainer = BoatraceModelTrainer(Path(args.csv), Path(args.out))
            r = trainer.train(lane_mode=mode, save=False, verbose=False)
            if r:
                results.append(r)
                print(f"  完了: {mode:11s}  単勝={r['hit_win']:.4f}  "
                      f"pred_top1_lane1={r['pred_top1_lane1']:.3f}  "
                      f"lane依存={r['lane_importance_share']:.3f}")

        if results:
            comp = pd.DataFrame(results).set_index("lane_mode")
            print("\n" + "=" * 60)
            print("📊 lane_mode 比較 (actual_win_lane1 ≈ 0.55 が現実値)")
            print("=" * 60)
            print(comp.to_string())

            # 選定: 単勝的中率を保ちつつ、予測1位のlane1比率が現実値に最も近いモード
            actual = comp["actual_win_lane1"].iloc[0]
            comp["bias_gap"] = (comp["pred_top1_lane1"] - actual).abs()
            # 的中率が最良-0.01以内に収まるモードの中で bias_gap 最小を選ぶ
            best_hit = comp["hit_win"].max()
            cand = comp[comp["hit_win"] >= best_hit - 0.01]
            best_mode = cand["bias_gap"].idxmin()
            print(f"\n🏆 推奨モード: {best_mode}")
            print(f"   (的中率を保ちつつ偏りが現実値に最も近い)")

            # 推奨モードで再学習して保存
            print(f"\n>>> {best_mode} で再学習して保存します...")
            trainer = BoatraceModelTrainer(Path(args.csv), Path(args.out))
            trainer.train(lane_mode=best_mode, save=True, verbose=True)
    else:
        trainer = BoatraceModelTrainer(Path(args.csv), Path(args.out),
                                       recent_days=args.recent_days)
        trainer.train(lane_mode=args.lane_mode, save=True, verbose=True)


if __name__ == "__main__":
    main()