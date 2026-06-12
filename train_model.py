"""
② モデル学習スクリプト (LightGBM)

dataset/training_data.csv を読み込んでモデルを学習し
models/ に保存する。

使い方:
  python train_model.py
  python train_model.py --csv dataset/training_data.csv --out models/
"""

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb
import pickle

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 特徴量カラム
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "lane",
    "course_base_winrate",
    "rank_num",
    "age",
    "weight",
    "flying_count",
    "late_count",
    "avg_start_time",
    "win_rate_all",
    "win_rate_2",
    "win_rate_3",
    "local_win_rate",
    "local_win_rate_2",
    "local_win_rate_3",
    "motor_2rate",
    "boat_2rate",
]

# LightGBM パラメータ
LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}


# ─────────────────────────────────────────────
# 学習
# ─────────────────────────────────────────────
class BoatraceModelTrainer:
    def __init__(self, csv_path: Path, model_dir: Path):
        self.csv_path  = csv_path
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_win   = None   # 1着予測モデル
        self.model_top3  = None   # 3着内予測モデル
        self.feature_importance = None

    def load_data(self) -> pd.DataFrame:
        logger.info(f"CSV読み込み: {self.csv_path}")
        df = pd.read_csv(self.csv_path, encoding="utf-8-sig")
        logger.info(f"  全行数: {len(df):,}")

        # 着順データがある行のみ使用
        df = df[df["finish"].notna()].copy()
        logger.info(f"  着順あり: {len(df):,} 行")

        # 日付でソート（時系列分割のため）
        df["race_date"] = df["race_date"].astype(str)
        df = df.sort_values("race_date").reset_index(drop=True)

        return df

    def train(self):
        df = self.load_data()
        if len(df) < 500:
            print("⚠ データが少なすぎます（500行以上推奨）。クローラーでもっとデータを集めてください。")
            return

        X = df[FEATURE_COLS].copy()
        y_win  = df["is_win"].astype(int)
        y_top3 = df["is_top3"].astype(int)

        # 時系列クロスバリデーション（未来でテスト）
        tscv = TimeSeriesSplit(n_splits=5)
        splits = list(tscv.split(X))
        train_idx, val_idx = splits[-1]  # 最後の分割を使用

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        yw_train, yw_val   = y_win.iloc[train_idx],  y_win.iloc[val_idx]
        yt_train, yt_val   = y_top3.iloc[train_idx], y_top3.iloc[val_idx]

        logger.info(f"学習データ: {len(X_train):,} 行 / 検証データ: {len(X_val):,} 行")

        # ── 1着予測モデル ─────────────────────
        print("\n📊 1着予測モデル 学習中...")
        dtrain_w = lgb.Dataset(X_train, label=yw_train)
        dval_w   = lgb.Dataset(X_val,   label=yw_val, reference=dtrain_w)

        self.model_win = lgb.train(
            LGB_PARAMS,
            dtrain_w,
            num_boost_round=500,
            valid_sets=[dval_w],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
        )

        pred_w = self.model_win.predict(X_val)
        auc_w  = roc_auc_score(yw_val, pred_w)
        acc_w  = accuracy_score(yw_val, (pred_w > 0.5).astype(int))
        print(f"  AUC: {auc_w:.4f}  Accuracy: {acc_w:.4f}")

        # ── 3着内予測モデル ───────────────────
        print("\n📊 3着内予測モデル 学習中...")
        dtrain_t = lgb.Dataset(X_train, label=yt_train)
        dval_t   = lgb.Dataset(X_val,   label=yt_val, reference=dtrain_t)

        self.model_top3 = lgb.train(
            LGB_PARAMS,
            dtrain_t,
            num_boost_round=500,
            valid_sets=[dval_t],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
        )

        pred_t = self.model_top3.predict(X_val)
        auc_t  = roc_auc_score(yt_val, pred_t)
        acc_t  = accuracy_score(yt_val, (pred_t > 0.5).astype(int))
        print(f"  AUC: {auc_t:.4f}  Accuracy: {acc_t:.4f}")

        # ── 特徴量重要度 ──────────────────────
        imp = pd.DataFrame({
            "feature": FEATURE_COLS,
            "importance_win":  self.model_win.feature_importance(importance_type="gain"),
            "importance_top3": self.model_top3.feature_importance(importance_type="gain"),
        }).sort_values("importance_win", ascending=False)
        self.feature_importance = imp

        print("\n📈 特徴量重要度 (1着モデル)")
        print(imp[["feature", "importance_win"]].to_string(index=False))

        # ── 保存 ─────────────────────────────
        self._save(auc_w, auc_t)

    def _save(self, auc_win: float, auc_top3: float):
        # モデル保存
        win_path  = self.model_dir / "model_win.pkl"
        top3_path = self.model_dir / "model_top3.pkl"
        with open(win_path,  "wb") as f: pickle.dump(self.model_win,  f)
        with open(top3_path, "wb") as f: pickle.dump(self.model_top3, f)

        # メタ情報保存
        meta = {
            "feature_cols": FEATURE_COLS,
            "auc_win":  round(auc_win,  4),
            "auc_top3": round(auc_top3, 4),
            "lgb_params": LGB_PARAMS,
        }
        meta_path = self.model_dir / "model_meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        # 特徴量重要度保存
        if self.feature_importance is not None:
            imp_path = self.model_dir / "feature_importance.csv"
            self.feature_importance.to_csv(imp_path, index=False, encoding="utf-8-sig")

        print(f"\n✅ モデル保存完了!")
        print(f"   1着モデル  : {win_path.resolve()}  (AUC={auc_win:.4f})")
        print(f"   3着内モデル: {top3_path.resolve()}  (AUC={auc_top3:.4f})")
        print(f"\n  AUCの目安: 0.55以上で有意、0.60以上で良好、0.65以上で優秀")


def main():
    parser = argparse.ArgumentParser(description="競艇予想モデル学習")
    parser.add_argument("--csv", default="dataset/training_data.csv")
    parser.add_argument("--out", default="models/")
    args = parser.parse_args()

    trainer = BoatraceModelTrainer(Path(args.csv), Path(args.out))
    trainer.train()


if __name__ == "__main__":
    main()
