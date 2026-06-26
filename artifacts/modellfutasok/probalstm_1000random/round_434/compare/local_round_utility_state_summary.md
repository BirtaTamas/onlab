# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-spirit-vs-vitality-bo3-KtWhzrlsNkWCCS0U9BIlr3/spirit-vs-vitality-m2-nuke.csv`
- round_num: `27`
- rows: `220`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 220 | 1.000 | 0.862751 | 0.891812 | -0.029061 | 0 | 220 | 1.000000 | 1.000000 |
| active/recent utility | 220 | 1.000 | 0.862751 | 0.891812 | -0.029061 | 0 | 220 | 1.000000 | 1.000000 |
| strong utility action | 195 | 0.886 | 0.885226 | 0.914693 | -0.029467 | 0 | 195 | 1.000000 | 1.000000 |
| utility damage | 20 | 0.091 | 0.849995 | 0.903745 | -0.053750 | 0 | 20 | 1.000000 | 1.000000 |
| active smoke/inferno | 195 | 0.886 | 0.885226 | 0.914693 | -0.029467 | 0 | 195 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 220 | 1.000 | 0.862751 | 0.891812 | -0.029061 | 0 | 220 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `47.5s`, rows `79`
- `52.0s` - `109.5s`, rows `116`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `25.0`, LSTM `0.6539`, XGBoost `0.7568`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.8260`, XGBoost `0.9059`, closer `xgboost`, smoke `7`, inferno `2`, utility_damage `38.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.7422`, XGBoost `0.8211`, closer `xgboost`, smoke `7`, inferno `2`, utility_damage `67.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.6811`, XGBoost `0.7511`, closer `xgboost`, smoke `6`, inferno `2`, utility_damage `2.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.5214`, XGBoost `0.5902`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.7363`, XGBoost `0.8037`, closer `xgboost`, smoke `7`, inferno `2`, utility_damage `107.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.6864`, XGBoost `0.7531`, closer `xgboost`, smoke `7`, inferno `2`, utility_damage `12.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.9294`, XGBoost `0.9958`, closer `xgboost`, smoke `7`, inferno `2`, utility_damage `113.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.5214`, XGBoost `0.5872`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.5190`, XGBoost `0.5847`, closer `xgboost`, smoke `4`, inferno `2`, utility_damage `0.0`, recent_utility `0`
