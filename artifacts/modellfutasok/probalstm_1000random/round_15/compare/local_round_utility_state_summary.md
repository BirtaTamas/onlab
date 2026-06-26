# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-virtuspro-vs-og-inferno-UyQlNJx_rptvvsTtINI5j3/virtus-pro-vs-og-inferno.csv`
- round_num: `8`
- rows: `263`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 263 | 1.000 | 0.393720 | 0.317550 | 0.076170 | 71 | 192 | 0.387833 | 0.532319 |
| active/recent utility | 263 | 1.000 | 0.393720 | 0.317550 | 0.076170 | 71 | 192 | 0.387833 | 0.532319 |
| strong utility action | 177 | 0.673 | 0.505785 | 0.398014 | 0.107771 | 12 | 165 | 0.214689 | 0.429379 |
| utility damage | 34 | 0.129 | 0.328460 | 0.224077 | 0.104383 | 7 | 27 | 0.588235 | 0.735294 |
| active smoke/inferno | 177 | 0.673 | 0.505785 | 0.398014 | 0.107771 | 12 | 165 | 0.214689 | 0.429379 |
| recent utility last 5s | 10 | 0.038 | 0.642116 | 0.507084 | 0.135032 | 0 | 10 | 0.000000 | 0.100000 |
| flash effect present | 263 | 1.000 | 0.393720 | 0.317550 | 0.076170 | 71 | 192 | 0.387833 | 0.532319 |

## Active Smoke/Inferno Intervals

- `10.5s` - `40.5s`, rows `61`
- `41.5s` - `99.0s`, rows `116`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `79.0`, LSTM `0.5175`, XGBoost `0.2102`, closer `xgboost`, smoke `4`, inferno `2`, utility_damage `20.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.4250`, XGBoost `0.1968`, closer `xgboost`, smoke `4`, inferno `2`, utility_damage `21.0`, recent_utility `0`
- seconds `83.0`, LSTM `0.3628`, XGBoost `0.1355`, closer `xgboost`, smoke `4`, inferno `2`, utility_damage `17.0`, recent_utility `0`
- seconds `80.0`, LSTM `0.4068`, XGBoost `0.1900`, closer `xgboost`, smoke `4`, inferno `2`, utility_damage `23.0`, recent_utility `0`
- seconds `80.5`, LSTM `0.4049`, XGBoost `0.1885`, closer `xgboost`, smoke `4`, inferno `2`, utility_damage `23.0`, recent_utility `0`
- seconds `76.5`, LSTM `0.6404`, XGBoost `0.4586`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.6466`, XGBoost `0.4652`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.6443`, XGBoost `0.4652`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.5`, LSTM `0.6424`, XGBoost `0.4657`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `82.0`, LSTM `0.3038`, XGBoost `0.1273`, closer `xgboost`, smoke `4`, inferno `2`, utility_damage `23.0`, recent_utility `0`
