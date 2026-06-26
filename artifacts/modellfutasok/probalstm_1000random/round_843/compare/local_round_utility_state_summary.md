# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-g2-vs-liquid-bo3-w6HylYj4nF7GNnrWujmZUZ/g2-vs-liquid-m2-inferno.csv`
- round_num: `1`
- rows: `224`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 224 | 1.000 | 0.692104 | 0.818053 | -0.125949 | 22 | 202 | 1.000000 | 1.000000 |
| active/recent utility | 224 | 1.000 | 0.692104 | 0.818053 | -0.125949 | 22 | 202 | 1.000000 | 1.000000 |
| strong utility action | 95 | 0.424 | 0.699529 | 0.820735 | -0.121206 | 4 | 91 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 95 | 0.424 | 0.699529 | 0.820735 | -0.121206 | 4 | 91 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.045 | 0.530459 | 0.724178 | -0.193719 | 0 | 10 | 1.000000 | 1.000000 |
| flash effect present | 224 | 1.000 | 0.692104 | 0.818053 | -0.125949 | 22 | 202 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.0s` - `31.0s`, rows `45`
- `47.0s` - `71.5s`, rows `50`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `29.5`, LSTM `0.5425`, XGBoost `0.7812`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.5398`, XGBoost `0.7782`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.5413`, XGBoost `0.7782`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.5393`, XGBoost `0.7738`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.5484`, XGBoost `0.7804`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.5507`, XGBoost `0.7770`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.5662`, XGBoost `0.7770`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.5691`, XGBoost `0.7774`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.5753`, XGBoost `0.7825`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.5218`, XGBoost `0.7260`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
