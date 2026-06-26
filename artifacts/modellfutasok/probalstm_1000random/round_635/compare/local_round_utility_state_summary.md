# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-flyquest-vs-tyloo-bo3-b6a1tT091Xo0vOjw70TVd9/flyquest-vs-tyloo-m2-mirage.csv`
- round_num: `4`
- rows: `212`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 212 | 1.000 | 0.638565 | 0.698652 | -0.060087 | 47 | 165 | 0.943396 | 0.797170 |
| active/recent utility | 212 | 1.000 | 0.638565 | 0.698652 | -0.060087 | 47 | 165 | 0.943396 | 0.797170 |
| strong utility action | 125 | 0.590 | 0.643147 | 0.692504 | -0.049357 | 41 | 84 | 0.992000 | 0.776000 |
| utility damage | 20 | 0.094 | 0.631236 | 0.642951 | -0.011714 | 10 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 115 | 0.542 | 0.654062 | 0.709950 | -0.055888 | 31 | 84 | 1.000000 | 0.843478 |
| recent utility last 5s | 10 | 0.047 | 0.517621 | 0.491881 | 0.025740 | 10 | 0 | 0.900000 | 0.000000 |
| flash effect present | 212 | 1.000 | 0.638565 | 0.698652 | -0.060087 | 47 | 165 | 0.943396 | 0.797170 |

## Active Smoke/Inferno Intervals

- `6.5s` - `63.5s`, rows `115`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `37.5`, LSTM `0.7274`, XGBoost `0.9089`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.7314`, XGBoost `0.9089`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.7372`, XGBoost `0.9089`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.7383`, XGBoost `0.9051`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.7382`, XGBoost `0.9031`, closer `xgboost`, smoke `9`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.7476`, XGBoost `0.9089`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.7437`, XGBoost `0.9031`, closer `xgboost`, smoke `8`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.7495`, XGBoost `0.9089`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.7461`, XGBoost `0.9052`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.7452`, XGBoost `0.9031`, closer `xgboost`, smoke `8`, inferno `1`, utility_damage `0.0`, recent_utility `0`
