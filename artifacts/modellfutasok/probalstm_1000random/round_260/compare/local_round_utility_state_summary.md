# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-furia-vs-virtuspro-bo3-E_bOFuD3YUjLJCO2xRj0mq/furia-vs-virtus-pro-m1-mirage.csv`
- round_num: `10`
- rows: `245`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 245 | 1.000 | 0.580072 | 0.546664 | 0.033407 | 61 | 184 | 0.236735 | 0.248980 |
| active/recent utility | 245 | 1.000 | 0.580072 | 0.546664 | 0.033407 | 61 | 184 | 0.236735 | 0.248980 |
| strong utility action | 204 | 0.833 | 0.636668 | 0.600586 | 0.036082 | 36 | 168 | 0.161765 | 0.176471 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 198 | 0.808 | 0.632689 | 0.597280 | 0.035409 | 36 | 162 | 0.166667 | 0.181818 |
| recent utility last 5s | 10 | 0.041 | 0.769701 | 0.706935 | 0.062766 | 0 | 10 | 0.000000 | 0.000000 |
| flash effect present | 245 | 1.000 | 0.580072 | 0.546664 | 0.033407 | 61 | 184 | 0.236735 | 0.248980 |

## Active Smoke/Inferno Intervals

- `7.0s` - `48.5s`, rows `84`
- `53.0s` - `109.5s`, rows `114`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `93.5`, LSTM `0.4694`, XGBoost `0.3442`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `92.0`, LSTM `0.5751`, XGBoost `0.4501`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `92.5`, LSTM `0.5627`, XGBoost `0.4385`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.8202`, XGBoost `0.7039`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `95.5`, LSTM `0.0294`, XGBoost `0.1450`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.8204`, XGBoost `0.7060`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `93.0`, LSTM `0.5317`, XGBoost `0.4279`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `95.0`, LSTM `0.0526`, XGBoost `0.1507`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.8110`, XGBoost `0.7170`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.8077`, XGBoost `0.7151`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
