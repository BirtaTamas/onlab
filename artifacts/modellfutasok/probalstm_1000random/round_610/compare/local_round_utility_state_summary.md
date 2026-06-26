# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-furia-vs-virtuspro-bo3-E_bOFuD3YUjLJCO2xRj0mq/furia-vs-virtus-pro-m1-mirage.csv`
- round_num: `11`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.697684 | 0.665403 | 0.032280 | 153 | 77 | 1.000000 | 0.782609 |
| active/recent utility | 230 | 1.000 | 0.697684 | 0.665403 | 0.032280 | 153 | 77 | 1.000000 | 0.782609 |
| strong utility action | 163 | 0.709 | 0.687364 | 0.654675 | 0.032688 | 112 | 51 | 1.000000 | 0.809816 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 155 | 0.674 | 0.691458 | 0.662015 | 0.029443 | 104 | 51 | 1.000000 | 0.800000 |
| recent utility last 5s | 10 | 0.043 | 0.608888 | 0.512504 | 0.096384 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 230 | 1.000 | 0.697684 | 0.665403 | 0.032280 | 153 | 77 | 1.000000 | 0.782609 |

## Active Smoke/Inferno Intervals

- `6.0s` - `34.0s`, rows `57`
- `53.0s` - `101.5s`, rows `98`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `6.5`, LSTM `0.6274`, XGBoost `0.5127`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `1`
- seconds `4.5`, LSTM `0.6202`, XGBoost `0.5106`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `4.0`, LSTM `0.6126`, XGBoost `0.5105`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `30.0`, LSTM `0.5874`, XGBoost `0.4878`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `2.5`, LSTM `0.6123`, XGBoost `0.5143`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `3.0`, LSTM `0.6120`, XGBoost `0.5143`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `3.5`, LSTM `0.6101`, XGBoost `0.5143`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `31.0`, LSTM `0.5871`, XGBoost `0.4917`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `5.0`, LSTM `0.6076`, XGBoost `0.5127`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `31.5`, LSTM `0.5822`, XGBoost `0.4874`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
