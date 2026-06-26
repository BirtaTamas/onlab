# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-furia-bo5-6eeTFVdtPEH4qPNc6w4Z3Y/the-mongolz-vs-furia-m5-dust2.csv`
- round_num: `18`
- rows: `157`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 157 | 1.000 | 0.775767 | 0.771176 | 0.004592 | 59 | 98 | 1.000000 | 1.000000 |
| active/recent utility | 157 | 1.000 | 0.775767 | 0.771176 | 0.004592 | 59 | 98 | 1.000000 | 1.000000 |
| strong utility action | 105 | 0.669 | 0.710623 | 0.698852 | 0.011771 | 54 | 51 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 96 | 0.611 | 0.723326 | 0.713949 | 0.009377 | 45 | 51 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.064 | 0.580324 | 0.537226 | 0.043098 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 157 | 1.000 | 0.775767 | 0.771176 | 0.004592 | 59 | 98 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.0s` - `42.0s`, rows `71`
- `55.5s` - `62.0s`, rows `14`
- `65.0s` - `70.0s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `25.0`, LSTM `0.6204`, XGBoost `0.5236`, closer `lstm`, smoke `7`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `7.0`, LSTM `0.6272`, XGBoost `0.5319`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `24.5`, LSTM `0.6147`, XGBoost `0.5262`, closer `lstm`, smoke `7`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.6076`, XGBoost `0.5227`, closer `lstm`, smoke `7`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `6.5`, LSTM `0.6151`, XGBoost `0.5374`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `7.5`, LSTM `0.6004`, XGBoost `0.5274`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.6096`, XGBoost `0.5372`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.5830`, XGBoost `0.5108`, closer `lstm`, smoke `7`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.5861`, XGBoost `0.5152`, closer `lstm`, smoke `7`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `70.0`, LSTM `0.7600`, XGBoost `0.8270`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
