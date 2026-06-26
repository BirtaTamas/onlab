# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-flyquest-vs-tyloo-bo3-b6a1tT091Xo0vOjw70TVd9/flyquest-vs-tyloo-m3-anubis.csv`
- round_num: `16`
- rows: `181`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 181 | 1.000 | 0.180146 | 0.286968 | -0.106822 | 181 | 0 | 1.000000 | 0.928177 |
| active/recent utility | 181 | 1.000 | 0.180146 | 0.286968 | -0.106822 | 181 | 0 | 1.000000 | 0.928177 |
| strong utility action | 162 | 0.895 | 0.163744 | 0.275954 | -0.112210 | 162 | 0 | 1.000000 | 0.993827 |
| utility damage | 26 | 0.144 | 0.169820 | 0.193004 | -0.023184 | 26 | 0 | 1.000000 | 0.961538 |
| active smoke/inferno | 162 | 0.895 | 0.163744 | 0.275954 | -0.112210 | 162 | 0 | 1.000000 | 0.993827 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 181 | 1.000 | 0.180146 | 0.286968 | -0.106822 | 181 | 0 | 1.000000 | 0.928177 |

## Active Smoke/Inferno Intervals

- `7.0s` - `87.5s`, rows `162`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `60.0`, LSTM `0.0529`, XGBoost `0.3140`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `32.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.0533`, XGBoost `0.3142`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `32.0`, recent_utility `0`
- seconds `61.0`, LSTM `0.0608`, XGBoost `0.3157`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `24.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.1423`, XGBoost `0.3840`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.0717`, XGBoost `0.3118`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `32.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.1304`, XGBoost `0.3629`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.1518`, XGBoost `0.3840`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `58.5`, LSTM `0.0722`, XGBoost `0.3020`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `32.0`, recent_utility `0`
- seconds `59.0`, LSTM `0.0804`, XGBoost `0.3094`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `32.0`, recent_utility `0`
- seconds `18.5`, LSTM `0.1402`, XGBoost `0.3678`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
