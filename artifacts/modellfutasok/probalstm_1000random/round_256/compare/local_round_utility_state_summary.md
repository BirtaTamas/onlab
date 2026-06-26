# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-fluxo-bo3-q_dqfGh9bi4kDnaRAX0wjf/chinggis-warriors-vs-fluxo-m2-mirage.csv`
- round_num: `23`
- rows: `216`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 216 | 1.000 | 0.266255 | 0.321838 | -0.055583 | 213 | 3 | 0.722222 | 0.462963 |
| active/recent utility | 216 | 1.000 | 0.266255 | 0.321838 | -0.055583 | 213 | 3 | 0.722222 | 0.462963 |
| strong utility action | 183 | 0.847 | 0.309746 | 0.373504 | -0.063758 | 180 | 3 | 0.672131 | 0.377049 |
| utility damage | 18 | 0.083 | 0.532168 | 0.559570 | -0.027402 | 16 | 2 | 0.111111 | 0.000000 |
| active smoke/inferno | 172 | 0.796 | 0.303597 | 0.363673 | -0.060076 | 169 | 3 | 0.651163 | 0.401163 |
| recent utility last 5s | 11 | 0.051 | 0.405901 | 0.527236 | -0.121335 | 11 | 0 | 1.000000 | 0.000000 |
| flash effect present | 216 | 1.000 | 0.266255 | 0.321838 | -0.055583 | 213 | 3 | 0.722222 | 0.462963 |

## Active Smoke/Inferno Intervals

- `6.5s` - `92.0s`, rows `172`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `59.5`, LSTM `0.0547`, XGBoost `0.3663`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.0656`, XGBoost `0.3720`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.0`, LSTM `0.0645`, XGBoost `0.3666`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.0846`, XGBoost `0.3714`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.1909`, XGBoost `0.4608`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.5`, LSTM `0.0879`, XGBoost `0.3563`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.2143`, XGBoost `0.4616`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.2171`, XGBoost `0.4645`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.0`, LSTM `0.1401`, XGBoost `0.3571`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.0`, LSTM `0.1938`, XGBoost `0.4099`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
