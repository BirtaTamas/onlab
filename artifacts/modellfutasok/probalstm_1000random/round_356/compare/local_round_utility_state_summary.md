# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-imperial-vs-legacy-bo3-GRvbnL5Q4zT_JzAd-0AXgo/imperial-vs-legacy-m3-mirage.csv`
- round_num: `14`
- rows: `290`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 290 | 1.000 | 0.285067 | 0.397039 | -0.111973 | 51 | 239 | 0.086207 | 0.089655 |
| active/recent utility | 290 | 1.000 | 0.285067 | 0.397039 | -0.111973 | 51 | 239 | 0.086207 | 0.089655 |
| strong utility action | 133 | 0.459 | 0.263206 | 0.361811 | -0.098606 | 2 | 131 | 0.052632 | 0.082707 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 122 | 0.421 | 0.263147 | 0.367201 | -0.104053 | 2 | 120 | 0.057377 | 0.090164 |
| recent utility last 5s | 11 | 0.038 | 0.263853 | 0.302036 | -0.038183 | 0 | 11 | 0.000000 | 0.000000 |
| flash effect present | 290 | 1.000 | 0.285067 | 0.397039 | -0.111973 | 51 | 239 | 0.086207 | 0.089655 |

## Active Smoke/Inferno Intervals

- `7.5s` - `30.5s`, rows `47`
- `54.0s` - `91.0s`, rows `75`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `90.5`, LSTM `0.0622`, XGBoost `0.3434`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `91.0`, LSTM `0.0588`, XGBoost `0.3389`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `89.0`, LSTM `0.0708`, XGBoost `0.3506`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `89.5`, LSTM `0.0730`, XGBoost `0.3498`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `87.5`, LSTM `0.0759`, XGBoost `0.3507`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `88.0`, LSTM `0.0768`, XGBoost `0.3507`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `90.0`, LSTM `0.0744`, XGBoost `0.3481`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `88.5`, LSTM `0.0777`, XGBoost `0.3507`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.3271`, XGBoost `0.5976`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `87.0`, LSTM `0.0872`, XGBoost `0.3514`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
