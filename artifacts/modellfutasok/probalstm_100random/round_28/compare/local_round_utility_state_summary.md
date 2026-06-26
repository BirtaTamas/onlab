# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-vitality-vs-tyloo-bo3-WTYOidpO-mHqROoLZlA7Li/vitality-vs-tyloo-m1-overpass.csv`
- round_num: `14`
- rows: `237`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 237 | 1.000 | 0.221316 | 0.264245 | -0.042929 | 221 | 16 | 1.000000 | 1.000000 |
| active/recent utility | 237 | 1.000 | 0.221316 | 0.264245 | -0.042929 | 221 | 16 | 1.000000 | 1.000000 |
| strong utility action | 166 | 0.700 | 0.176262 | 0.219678 | -0.043416 | 165 | 1 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 149 | 0.629 | 0.154185 | 0.198250 | -0.044065 | 148 | 1 | 1.000000 | 1.000000 |
| recent utility last 5s | 21 | 0.089 | 0.367178 | 0.404784 | -0.037605 | 21 | 0 | 1.000000 | 1.000000 |
| flash effect present | 237 | 1.000 | 0.221316 | 0.264245 | -0.042929 | 221 | 16 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.0s` - `27.5s`, rows `44`
- `66.0s` - `118.0s`, rows `105`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `11.0`, LSTM `0.3051`, XGBoost `0.4314`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.3122`, XGBoost `0.4314`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.3172`, XGBoost `0.4314`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.3175`, XGBoost `0.4314`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `86.0`, LSTM `0.0950`, XGBoost `0.1996`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `86.5`, LSTM `0.0959`, XGBoost `0.1998`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.3331`, XGBoost `0.4324`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `85.5`, LSTM `0.0791`, XGBoost `0.1767`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.5`, LSTM `0.0721`, XGBoost `0.1661`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `70.0`, LSTM `0.0727`, XGBoost `0.1661`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
