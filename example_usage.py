"""
予想ロジック使用例

このファイルを編集して自分の予想ロジックをカスタマイズできます。
"""

from datetime import date, timedelta
from boatrace_scraper import BoatraceScraper, RacerInfo, VENUE_MAP
from predictor import BoatracePredictor, PredictorConfig

# ─────────────────────────────────────────────
# 例1: シンプルな1レース予想
# ─────────────────────────────────────────────
def example_single_race():
    sc = BoatraceScraper(delay=1.5)
    predictor = BoatracePredictor()

    today = date.today()
    venue = "01"   # 桐生
    race_no = 1

    print(f"\n{'='*40}")
    print(f" {VENUE_MAP[venue]} {race_no}R の予想")
    print(f"{'='*40}")

    racers = sc.get_racelist(today, venue, race_no)
    if not racers:
        print("データが取得できませんでした（開催なし・通信エラーなど）")
        return

    result = predictor.predict(
        racers,
        race_date=today.strftime("%Y%m%d"),
        venue_name=VENUE_MAP[venue],
        race_no=race_no,
    )
    print(result.summary())


# ─────────────────────────────────────────────
# 例2: 重みをカスタマイズした予想
# ─────────────────────────────────────────────
def example_custom_weights():
    """
    モーター重視・コース重視に設定した例
    """
    cfg = PredictorConfig(
        w_course=40.0,      # コース重視（デフォルト35）
        w_win_rate=15.0,
        w_local_rate=10.0,
        w_rank=5.0,
        w_motor=20.0,       # モーター重視（デフォルト10）
        w_boat=5.0,
        w_start=5.0,
        penalty_per_flying=5.0,  # フライングを重ペナルティに
    )
    predictor = BoatracePredictor(cfg)

    # ダミーデータで動作確認
    racers = _sample_racers()
    result = predictor.predict(racers, "20260522", "住之江", 6)
    print(result.summary())

    # スコア内訳も確認
    print("\n▼ スコア内訳")
    for s in sorted(result.scores, key=lambda x: x.lane):
        print(f"  {s.lane}枠 {s.name}: {s.total_score:.2f} | {s.breakdown}")


# ─────────────────────────────────────────────
# 例3: 全レース一括予想（JSONから読み込み）
# ─────────────────────────────────────────────
def example_from_json(json_path: str):
    import json
    predictor = BoatracePredictor()

    with open(json_path, encoding="utf-8") as f:
        races_data = json.load(f)

    results = predictor.predict_all_races(races_data)
    for r in results:
        print(r.summary())

    # 全レースの推奨3連単をまとめて表示
    print("\n▼ 全レース 3連単 推奨買い目一覧")
    for r in results:
        print(f"  {r.race_no}R: {' / '.join(r.sanren_tan)}")


# ─────────────────────────────────────────────
# 例4: 自分でロジックを追加する
# ─────────────────────────────────────────────
def my_custom_predictor(racers: list[RacerInfo]) -> list[tuple[int, float]]:
    """
    独自ロジックでスコアリングするサンプル。
    (lane, score) のリストを返す。

    ここに自分のアイデアを実装してください：
    - 天候・風速・水面状況を加味する
    - 直前情報（展示タイム）を加味する
    - 選手の得意コース・不得意コースを分析する
    - 過去の対戦データを使う
    """
    scores = []
    for r in racers:
        score = 0.0

        # === ここに独自ロジックを記述 ===

        # 例: A1選手が1コースなら大幅加点
        if r.rank == "A1" and r.lane == 1:
            score += 50

        # 例: 当地勝率が高い選手を重視
        if r.local_win_rate and r.local_win_rate >= 7.0:
            score += 20

        # 例: モーター2連率が55%以上なら加点
        if r.motor_2rate and r.motor_2rate >= 55.0:
            score += 15

        # 例: フライング持ちは大きく減点
        if r.flying_count and r.flying_count >= 2:
            score -= 30

        # ================================

        scores.append((r.lane, score))

    return sorted(scores, key=lambda x: x[1], reverse=True)


# ─────────────────────────────────────────────
# サンプルデータ（API疎通前の動作確認用）
# ─────────────────────────────────────────────
def _sample_racers() -> list[RacerInfo]:
    return [
        RacerInfo(lane=1, racer_no="4123", name="山田太郎", branch="大阪",
                  age=35, weight=52.0, rank="A1", flying_count=0, late_count=0,
                  avg_start_time=0.14, win_rate_all=7.2, win_rate_2=42.0,
                  local_win_rate=6.8, motor_no="35", motor_2rate=48.5,
                  boat_no="12", boat_2rate=32.1),
        RacerInfo(lane=2, racer_no="3987", name="鈴木一郎", branch="愛知",
                  age=28, weight=51.0, rank="A2", flying_count=1, late_count=0,
                  avg_start_time=0.16, win_rate_all=6.1, win_rate_2=36.2,
                  local_win_rate=5.5, motor_no="22", motor_2rate=55.0,
                  boat_no="08", boat_2rate=41.0),
        RacerInfo(lane=3, racer_no="4501", name="田中花子", branch="福岡",
                  age=32, weight=49.5, rank="A1", flying_count=0, late_count=0,
                  avg_start_time=0.12, win_rate_all=6.8, win_rate_2=39.0,
                  local_win_rate=7.1, motor_no="11", motor_2rate=38.0,
                  boat_no="05", boat_2rate=28.5),
        RacerInfo(lane=4, racer_no="5012", name="佐藤次郎", branch="東京",
                  age=25, weight=53.0, rank="B1", flying_count=0, late_count=1,
                  avg_start_time=0.18, win_rate_all=4.5, win_rate_2=28.0,
                  local_win_rate=4.0, motor_no="44", motor_2rate=31.2,
                  boat_no="17", boat_2rate=35.0),
        RacerInfo(lane=5, racer_no="4789", name="高橋三郎", branch="兵庫",
                  age=41, weight=54.5, rank="A2", flying_count=0, late_count=0,
                  avg_start_time=0.19, win_rate_all=5.8, win_rate_2=33.0,
                  local_win_rate=5.2, motor_no="03", motor_2rate=42.0,
                  boat_no="21", boat_2rate=29.0),
        RacerInfo(lane=6, racer_no="5234", name="渡辺四郎", branch="岡山",
                  age=22, weight=52.5, rank="B2", flying_count=0, late_count=0,
                  avg_start_time=0.21, win_rate_all=3.2, win_rate_2=18.0,
                  local_win_rate=2.8, motor_no="29", motor_2rate=27.0,
                  boat_no="33", boat_2rate=22.0),
    ]


# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("▼ サンプルデータによる予想デモ")
    predictor = BoatracePredictor()
    racers = _sample_racers()
    result = predictor.predict(racers, "20260522", "住之江", 6)
    print(result.summary())

    print("\n▼ カスタム重みによる予想")
    example_custom_weights()

    print("\n▼ 独自ロジックによるスコアリング")
    scores = my_custom_predictor(racers)
    print("  独自スコアランキング:")
    for lane, score in scores:
        name = next(r.name for r in racers if r.lane == lane)
        print(f"    {lane}枠 {name}: {score:.1f}点")
