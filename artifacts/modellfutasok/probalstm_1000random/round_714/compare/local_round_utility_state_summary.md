# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-furia-vs-g2-bo3-QMek4tXQesgbTlulfGKOmD/furia-vs-g2-m1-inferno.csv`
- round_num: `8`
- rows: `178`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 178 | 1.000 | 0.280707 | 0.389603 | -0.108896 | 14 | 164 | 0.078652 | 0.398876 |
| active/recent utility | 178 | 1.000 | 0.280707 | 0.389603 | -0.108896 | 14 | 164 | 0.078652 | 0.398876 |
| strong utility action | 89 | 0.500 | 0.231064 | 0.363863 | -0.132798 | 0 | 89 | 0.000000 | 0.415730 |
| utility damage | 25 | 0.140 | 0.271138 | 0.385258 | -0.114120 | 0 | 25 | 0.000000 | 0.400000 |
| active smoke/inferno | 89 | 0.500 | 0.231064 | 0.363863 | -0.132798 | 0 | 89 | 0.000000 | 0.415730 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 178 | 1.000 | 0.280707 | 0.389603 | -0.108896 | 14 | 164 | 0.078652 | 0.398876 |

## Active Smoke/Inferno Intervals

- `10.0s` - `54.0s`, rows `89`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `40.0`, LSTM `0.0481`, XGBoost `0.2254`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.0488`, XGBoost `0.2254`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.0508`, XGBoost `0.2254`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.1488`, XGBoost `0.3222`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.3475`, XGBoost `0.5196`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `8.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.0556`, XGBoost `0.2254`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.0572`, XGBoost `0.2254`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.3652`, XGBoost `0.5329`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.0626`, XGBoost `0.2274`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.3622`, XGBoost `0.5235`, closer `xgboost`, smoke `0`, inferno `3`, utility_damage `0.0`, recent_utility `0`
