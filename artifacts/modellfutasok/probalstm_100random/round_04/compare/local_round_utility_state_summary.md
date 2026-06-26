# Local Round Utility State Analysis

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-spirit-vs-astralis-bo3-GZVTrKsE-zdG9dH6juITei/spirit-vs-astralis-m1-nuke.csv`
- round_num: `5`
- rows: `182`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 182 | 1.000 | 0.417420 | 0.397520 | 0.019900 | 61 | 121 | 0.236264 | 0.346154 |
| active/recent utility | 182 | 1.000 | 0.417420 | 0.397520 | 0.019900 | 61 | 121 | 0.236264 | 0.346154 |
| strong utility action | 130 | 0.714 | 0.519994 | 0.488218 | 0.031776 | 11 | 119 | 0.053846 | 0.207692 |
| utility damage | 10 | 0.055 | 0.537480 | 0.515309 | 0.022171 | 0 | 10 | 0.000000 | 0.000000 |
| active smoke/inferno | 130 | 0.714 | 0.519994 | 0.488218 | 0.031776 | 11 | 119 | 0.053846 | 0.207692 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 182 | 1.000 | 0.417420 | 0.397520 | 0.019900 | 61 | 121 | 0.236264 | 0.346154 |

## Active Smoke/Inferno Intervals

- `8.0s` - `72.5s`, rows `130`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `69.5`, LSTM `0.3404`, XGBoost `0.2299`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.0`, LSTM `0.5473`, XGBoost `0.4783`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.5`, LSTM `0.5462`, XGBoost `0.4783`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.5704`, XGBoost `0.5026`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.5693`, XGBoost `0.5027`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.0`, LSTM `0.5431`, XGBoost `0.4783`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.5416`, XGBoost `0.4781`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.5658`, XGBoost `0.5026`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.5649`, XGBoost `0.5026`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.5617`, XGBoost `0.5026`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
