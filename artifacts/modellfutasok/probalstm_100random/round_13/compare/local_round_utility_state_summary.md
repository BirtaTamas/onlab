# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-flyquest-vs-tyloo-bo3-b6a1tT091Xo0vOjw70TVd9/flyquest-vs-tyloo-m3-anubis.csv`
- round_num: `9`
- rows: `158`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 158 | 1.000 | 0.558411 | 0.616715 | -0.058303 | 149 | 9 | 0.196203 | 0.132911 |
| active/recent utility | 158 | 1.000 | 0.558411 | 0.616715 | -0.058303 | 149 | 9 | 0.196203 | 0.132911 |
| strong utility action | 140 | 0.886 | 0.546380 | 0.605308 | -0.058928 | 131 | 9 | 0.221429 | 0.150000 |
| utility damage | 20 | 0.127 | 0.657498 | 0.699096 | -0.041598 | 17 | 3 | 0.000000 | 0.000000 |
| active smoke/inferno | 140 | 0.886 | 0.546380 | 0.605308 | -0.058928 | 131 | 9 | 0.221429 | 0.150000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 158 | 1.000 | 0.558411 | 0.616715 | -0.058303 | 149 | 9 | 0.196203 | 0.132911 |

## Active Smoke/Inferno Intervals

- `9.0s` - `78.5s`, rows `140`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `64.5`, LSTM `0.3544`, XGBoost `0.6520`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.0`, LSTM `0.3596`, XGBoost `0.6504`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.3973`, XGBoost `0.6504`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `67.0`, LSTM `0.2041`, XGBoost `0.4065`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.0`, LSTM `0.0965`, XGBoost `0.2857`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.5`, LSTM `0.2454`, XGBoost `0.4331`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `71.0`, LSTM `0.1036`, XGBoost `0.2896`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `70.5`, LSTM `0.1117`, XGBoost `0.2899`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `71.5`, LSTM `0.1125`, XGBoost `0.2857`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `70.0`, LSTM `0.1242`, XGBoost `0.2901`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
