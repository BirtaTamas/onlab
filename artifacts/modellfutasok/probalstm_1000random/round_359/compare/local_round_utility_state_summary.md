# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-g2-vs-virtuspro-bo3-lXkBTaEEYeJRsoa-wcGwPP/g2-vs-virtus-pro-m3-dust2.csv`
- round_num: `1`
- rows: `115`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 115 | 1.000 | 0.523084 | 0.617694 | -0.094610 | 12 | 103 | 0.434783 | 1.000000 |
| active/recent utility | 115 | 1.000 | 0.523084 | 0.617694 | -0.094610 | 12 | 103 | 0.434783 | 1.000000 |
| strong utility action | 61 | 0.530 | 0.466100 | 0.616708 | -0.150608 | 0 | 61 | 0.163934 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 61 | 0.530 | 0.466100 | 0.616708 | -0.150608 | 0 | 61 | 0.163934 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 115 | 1.000 | 0.523084 | 0.617694 | -0.094610 | 12 | 103 | 0.434783 | 1.000000 |

## Active Smoke/Inferno Intervals

- `16.5s` - `46.5s`, rows `61`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `22.0`, LSTM `0.3788`, XGBoost `0.6077`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.5516`, XGBoost `0.7784`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.4842`, XGBoost `0.7025`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.4353`, XGBoost `0.6405`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.4443`, XGBoost `0.6394`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.4577`, XGBoost `0.6504`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.4448`, XGBoost `0.6353`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.4515`, XGBoost `0.6405`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.4723`, XGBoost `0.6520`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.4622`, XGBoost `0.6405`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
