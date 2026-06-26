# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-legacy-vs-nrg-bo3-_uO_eo-VIGwp_pYoaUs9Le/legacy-vs-nrg-m3-dust2.csv`
- round_num: `5`
- rows: `290`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 290 | 1.000 | 0.216529 | 0.291160 | -0.074631 | 274 | 16 | 0.951724 | 0.951724 |
| active/recent utility | 290 | 1.000 | 0.216529 | 0.291160 | -0.074631 | 274 | 16 | 0.951724 | 0.951724 |
| strong utility action | 178 | 0.614 | 0.314496 | 0.427469 | -0.112973 | 163 | 15 | 0.921348 | 0.921348 |
| utility damage | 10 | 0.034 | 0.232543 | 0.413181 | -0.180638 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 178 | 0.614 | 0.314496 | 0.427469 | -0.112973 | 163 | 15 | 0.921348 | 0.921348 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 290 | 1.000 | 0.216529 | 0.291160 | -0.074631 | 274 | 16 | 0.951724 | 0.951724 |

## Active Smoke/Inferno Intervals

- `6.0s` - `72.0s`, rows `133`
- `74.0s` - `96.0s`, rows `45`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `88.0`, LSTM `0.1586`, XGBoost `0.4492`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `87.5`, LSTM `0.1602`, XGBoost `0.4492`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `88.5`, LSTM `0.1579`, XGBoost `0.4342`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `87.0`, LSTM `0.1870`, XGBoost `0.4492`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `89.0`, LSTM `0.1810`, XGBoost `0.4316`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.1904`, XGBoost `0.4166`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `2.0`, recent_utility `0`
- seconds `86.5`, LSTM `0.2240`, XGBoost `0.4492`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `89.5`, LSTM `0.1936`, XGBoost `0.4174`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.2098`, XGBoost `0.4303`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.2005`, XGBoost `0.4195`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
