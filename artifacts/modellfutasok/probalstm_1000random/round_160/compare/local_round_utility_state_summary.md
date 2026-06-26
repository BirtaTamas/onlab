# Local Round Utility State Analysis

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-lynn-vision-vs-tyloo-bo3-tXRL8tbpb2Kb27VbUgWfK1/lynn-vision-vs-tyloo-m2-ancient.csv`
- round_num: `12`
- rows: `162`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 162 | 1.000 | 0.522200 | 0.492400 | 0.029800 | 124 | 38 | 0.493827 | 0.475309 |
| active/recent utility | 162 | 1.000 | 0.522200 | 0.492400 | 0.029800 | 124 | 38 | 0.493827 | 0.475309 |
| strong utility action | 147 | 0.907 | 0.519829 | 0.489196 | 0.030633 | 116 | 31 | 0.496599 | 0.489796 |
| utility damage | 15 | 0.093 | 0.623145 | 0.592333 | 0.030811 | 15 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 139 | 0.858 | 0.510239 | 0.481495 | 0.028744 | 108 | 31 | 0.467626 | 0.460432 |
| recent utility last 5s | 10 | 0.062 | 0.683162 | 0.622491 | 0.060671 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 162 | 1.000 | 0.522200 | 0.492400 | 0.029800 | 124 | 38 | 0.493827 | 0.475309 |

## Active Smoke/Inferno Intervals

- `6.0s` - `49.5s`, rows `88`
- `55.0s` - `80.0s`, rows `51`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `79.0`, LSTM `0.6825`, XGBoost `0.8920`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `64.0`, LSTM `0.4554`, XGBoost `0.2547`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.5`, LSTM `0.5419`, XGBoost `0.7235`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.5856`, XGBoost `0.7664`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.4175`, XGBoost `0.2633`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.7380`, XGBoost `0.8920`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `64.5`, LSTM `0.4175`, XGBoost `0.2701`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.4050`, XGBoost `0.2612`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `65.0`, LSTM `0.3968`, XGBoost `0.2767`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `80.0`, LSTM `0.7751`, XGBoost `0.8920`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
