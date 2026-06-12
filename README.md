# 競艇スクレイピングツール

BOAT RACE公式サイト（boatrace.jp）から出走表・オッズ・レース結果を取得するPythonツールです。

## 取得できるデータ

| データ種別 | 内容 |
|-----------|------|
| 出走表 | 選手名・番号・支部・年齢・体重・級別・成績・モーター・ボート |
| オッズ | 単勝・複勝・2連単・2連複・3連単・3連複 |
| レース結果 | 着順・払戻金・スタートタイム |

## セットアップ

```bash
pip install -r requirements.txt
```

## 使い方

### 会場コード確認
```bash
python main.py venues
```

### 出走表を取得
```bash
# 今日の桐生1Rの出走表
python main.py racelist --venue 01 --race 1

# 指定日・CSV出力
python main.py racelist --date 20260522 --venue 12 --race 6 --out output/住之江6R.csv
```

### オッズを取得
```bash
# 今日の戸田3Rのオッズ
python main.py odds --venue 02 --race 3

# JSON出力
python main.py odds --venue 02 --race 3 --out output/odds.json
```

### レース結果を取得
```bash
# 昨日の平和島5Rの結果
python main.py result --date 20260521 --venue 04 --race 5 --out output/result.json
```

### 全レースを一括取得
```bash
# 今日の住之江全レースをJSONに保存
python main.py all --venue 12 --out output/住之江_20260522.json
```

## ライブラリとして使う

```python
from datetime import date
from boatrace_scraper import BoatraceScraper

sc = BoatraceScraper(delay=1.5)
today = date.today()

# 出走表
racers = sc.get_racelist(today, "01", 1)
for r in racers:
    print(r.lane, r.name, r.rank, r.win_rate_all)

# オッズ
odds = sc.get_odds(today, "01", 1)
print(odds.sanren_tan)  # 3連単オッズ dict

# レース結果
result = sc.get_result(today, "01", 1)
print(result.arrival)   # 着順 [1着艇番, 2着艇番, ...]
print(result.payouts)   # 払戻金 dict
```

## 注意事項

- リクエスト間隔はデフォルト **1.5秒** に設定されています（サーバー負荷軽減のため）
- 短縮しすぎるとIP制限やエラーが発生する場合があります（`--delay 2.0` 推奨）
- このツールは個人の予想研究目的でご利用ください
- boatrace.jpの利用規約を必ずご確認ください

## 場コード一覧

| コード | 競艇場 | コード | 競艇場 |
|--------|--------|--------|--------|
| 01 | 桐生 | 13 | 尼崎 |
| 02 | 戸田 | 14 | 鳴門 |
| 03 | 江戸川 | 15 | 丸亀 |
| 04 | 平和島 | 16 | 児島 |
| 05 | 多摩川 | 17 | 宮島 |
| 06 | 浜名湖 | 18 | 徳山 |
| 07 | 蒲郡 | 19 | 下関 |
| 08 | 常滑 | 20 | 若松 |
| 09 | 津 | 21 | 芦屋 |
| 10 | 三国 | 22 | 福岡 |
| 11 | びわこ | 23 | 唐津 |
| 12 | 住之江 | 24 | 大村 |
