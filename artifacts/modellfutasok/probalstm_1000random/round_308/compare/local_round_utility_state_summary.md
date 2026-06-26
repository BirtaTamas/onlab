# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-faze-vs-aurora-bo3-ZssSxRC3p7Nn5A_BOLQ-lD/faze-vs-aurora-m2-mirage.csv`
- round_num: `4`
- rows: `181`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 181 | 1.000 | 0.573554 | 0.565369 | 0.008185 | 118 | 63 | 0.983425 | 0.839779 |
| active/recent utility | 181 | 1.000 | 0.573554 | 0.565369 | 0.008185 | 118 | 63 | 0.983425 | 0.839779 |
| strong utility action | 169 | 0.934 | 0.576201 | 0.570094 | 0.006106 | 106 | 63 | 0.982249 | 0.869822 |
| utility damage | 10 | 0.055 | 0.551197 | 0.559852 | -0.008654 | 0 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 169 | 0.934 | 0.576201 | 0.570094 | 0.006106 | 106 | 63 | 0.982249 | 0.869822 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 181 | 1.000 | 0.573554 | 0.565369 | 0.008185 | 118 | 63 | 0.983425 | 0.839779 |

## Active Smoke/Inferno Intervals

- `6.0s` - `90.0s`, rows `169`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `81.0`, LSTM `0.3589`, XGBoost `0.4900`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `80.5`, LSTM `0.3605`, XGBoost `0.4861`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `80.0`, LSTM `0.6184`, XGBoost `0.7335`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `84.5`, LSTM `0.7201`, XGBoost `0.8091`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `75.0`, LSTM `0.6632`, XGBoost `0.7521`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `89.0`, LSTM `0.8768`, XGBoost `0.9625`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `55.0`, recent_utility `0`
- seconds `85.0`, LSTM `0.7302`, XGBoost `0.8102`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `82.0`, LSTM `0.6187`, XGBoost `0.6986`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `90.0`, LSTM `0.8865`, XGBoost `0.9629`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `55.0`, recent_utility `0`
- seconds `89.5`, LSTM `0.8882`, XGBoost `0.9631`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `55.0`, recent_utility `0`
