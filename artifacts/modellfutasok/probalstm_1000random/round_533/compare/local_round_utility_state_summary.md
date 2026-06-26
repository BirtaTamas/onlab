# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-vitality-vs-mouz-bo3-kZzxcq2ibUgPOmQh0hZOgn/vitality-vs-mouz-m2-train.csv`
- round_num: `18`
- rows: `142`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 142 | 1.000 | 0.394302 | 0.458343 | -0.064041 | 140 | 2 | 0.464789 | 0.253521 |
| active/recent utility | 142 | 1.000 | 0.394302 | 0.458343 | -0.064041 | 140 | 2 | 0.464789 | 0.253521 |
| strong utility action | 109 | 0.768 | 0.408148 | 0.476665 | -0.068517 | 109 | 0 | 0.357798 | 0.229358 |
| utility damage | 10 | 0.070 | 0.496504 | 0.601955 | -0.105451 | 10 | 0 | 0.600000 | 0.000000 |
| active smoke/inferno | 109 | 0.768 | 0.408148 | 0.476665 | -0.068517 | 109 | 0 | 0.357798 | 0.229358 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 142 | 1.000 | 0.394302 | 0.458343 | -0.064041 | 140 | 2 | 0.464789 | 0.253521 |

## Active Smoke/Inferno Intervals

- `8.0s` - `49.5s`, rows `84`
- `55.5s` - `63.5s`, rows `17`
- `67.0s` - `70.5s`, rows `8`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `14.0`, LSTM `0.4839`, XGBoost `0.6144`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `10.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.4880`, XGBoost `0.6157`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.5291`, XGBoost `0.6563`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.4888`, XGBoost `0.6137`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `10.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.4944`, XGBoost `0.6157`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.4898`, XGBoost `0.6111`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `10.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.5383`, XGBoost `0.6583`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.5385`, XGBoost `0.6583`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.5381`, XGBoost `0.6541`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.4972`, XGBoost `0.6119`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `9.0`, recent_utility `0`
