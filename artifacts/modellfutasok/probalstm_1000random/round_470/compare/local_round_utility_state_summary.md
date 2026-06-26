# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-mouz-vs-vitality-bo3-pYxpz34IEN-t8y4DgB-MSD/mouz-vs-vitality-m3-train.csv`
- round_num: `14`
- rows: `184`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 184 | 1.000 | 0.937148 | 0.984924 | -0.047776 | 0 | 184 | 1.000000 | 1.000000 |
| active/recent utility | 184 | 1.000 | 0.937148 | 0.984924 | -0.047776 | 0 | 184 | 1.000000 | 1.000000 |
| strong utility action | 90 | 0.489 | 0.931989 | 0.983811 | -0.051823 | 0 | 90 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 90 | 0.489 | 0.931989 | 0.983811 | -0.051823 | 0 | 90 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 184 | 1.000 | 0.937148 | 0.984924 | -0.047776 | 0 | 184 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `11.0s` - `33.0s`, rows `45`
- `58.5s` - `80.5s`, rows `45`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `13.5`, LSTM `0.9088`, XGBoost `0.9800`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.9092`, XGBoost `0.9802`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.9093`, XGBoost `0.9802`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.9090`, XGBoost `0.9795`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.9095`, XGBoost `0.9801`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `19.0`, LSTM `0.9114`, XGBoost `0.9817`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.9152`, XGBoost `0.9849`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.9107`, XGBoost `0.9802`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.9107`, XGBoost `0.9800`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.9124`, XGBoost `0.9817`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
