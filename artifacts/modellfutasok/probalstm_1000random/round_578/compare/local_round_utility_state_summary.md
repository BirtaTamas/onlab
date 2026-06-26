# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-imperial-vs-metizport-bo3-yMtoBsoZq-jiQ0fSUscH7u/imperial-vs-metizport-m2-dust2.csv`
- round_num: `2`
- rows: `149`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 149 | 1.000 | 0.406271 | 0.485758 | -0.079488 | 129 | 20 | 0.496644 | 0.476510 |
| active/recent utility | 149 | 1.000 | 0.406271 | 0.485758 | -0.079488 | 129 | 20 | 0.496644 | 0.476510 |
| strong utility action | 131 | 0.879 | 0.436842 | 0.513046 | -0.076205 | 112 | 19 | 0.427481 | 0.404580 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 131 | 0.879 | 0.436842 | 0.513046 | -0.076205 | 112 | 19 | 0.427481 | 0.404580 |
| recent utility last 5s | 10 | 0.067 | 0.412932 | 0.483784 | -0.070852 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 149 | 1.000 | 0.406271 | 0.485758 | -0.079488 | 129 | 20 | 0.496644 | 0.476510 |

## Active Smoke/Inferno Intervals

- `9.0s` - `74.0s`, rows `131`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `26.0`, LSTM `0.5713`, XGBoost `0.7616`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.5705`, XGBoost `0.7593`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.5660`, XGBoost `0.7534`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.5196`, XGBoost `0.6970`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.5260`, XGBoost `0.7029`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.5295`, XGBoost `0.7004`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.5908`, XGBoost `0.7597`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.5297`, XGBoost `0.6984`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.5839`, XGBoost `0.7521`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.5942`, XGBoost `0.7597`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
