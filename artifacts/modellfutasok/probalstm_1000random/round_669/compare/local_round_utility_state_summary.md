# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-fluxo-bo3-q_dqfGh9bi4kDnaRAX0wjf/chinggis-warriors-vs-fluxo-m2-mirage.csv`
- round_num: `5`
- rows: `210`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 210 | 1.000 | 0.488095 | 0.486163 | 0.001932 | 88 | 122 | 0.438095 | 0.442857 |
| active/recent utility | 210 | 1.000 | 0.488095 | 0.486163 | 0.001932 | 88 | 122 | 0.438095 | 0.442857 |
| strong utility action | 187 | 0.890 | 0.508490 | 0.502739 | 0.005750 | 78 | 109 | 0.438503 | 0.374332 |
| utility damage | 30 | 0.143 | 0.689706 | 0.614281 | 0.075425 | 0 | 30 | 0.000000 | 0.033333 |
| active smoke/inferno | 187 | 0.890 | 0.508490 | 0.502739 | 0.005750 | 78 | 109 | 0.438503 | 0.374332 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 210 | 1.000 | 0.488095 | 0.486163 | 0.001932 | 88 | 122 | 0.438095 | 0.442857 |

## Active Smoke/Inferno Intervals

- `6.5s` - `99.5s`, rows `187`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `62.0`, LSTM `0.2626`, XGBoost `0.5238`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.5`, LSTM `0.2753`, XGBoost `0.5206`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.2866`, XGBoost `0.5175`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.3203`, XGBoost `0.5271`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.0`, LSTM `0.3168`, XGBoost `0.5185`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.3182`, XGBoost `0.5175`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.3369`, XGBoost `0.5271`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `71.0`, LSTM `0.3165`, XGBoost `0.5048`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `74.0`, LSTM `0.3469`, XGBoost `0.5228`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `71.5`, LSTM `0.2677`, XGBoost `0.4362`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
