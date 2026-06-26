# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-faze-vs-aurora-bo3-ZssSxRC3p7Nn5A_BOLQ-lD/faze-vs-aurora-m2-mirage.csv`
- round_num: `5`
- rows: `170`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 170 | 1.000 | 0.801555 | 0.759864 | 0.041690 | 142 | 28 | 1.000000 | 1.000000 |
| active/recent utility | 170 | 1.000 | 0.801555 | 0.759864 | 0.041690 | 142 | 28 | 1.000000 | 1.000000 |
| strong utility action | 165 | 0.971 | 0.802421 | 0.760674 | 0.041747 | 137 | 28 | 1.000000 | 1.000000 |
| utility damage | 14 | 0.082 | 0.968931 | 0.990993 | -0.022062 | 1 | 13 | 1.000000 | 1.000000 |
| active smoke/inferno | 153 | 0.900 | 0.806387 | 0.762791 | 0.043596 | 127 | 26 | 1.000000 | 1.000000 |
| recent utility last 5s | 17 | 0.100 | 0.752257 | 0.726620 | 0.025638 | 15 | 2 | 1.000000 | 1.000000 |
| flash effect present | 170 | 1.000 | 0.801555 | 0.759864 | 0.041690 | 142 | 28 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `11.5s`, rows `11`
- `13.5s` - `35.5s`, rows `45`
- `36.5s` - `84.5s`, rows `97`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `67.0`, LSTM `0.7936`, XGBoost `0.6816`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `67.5`, LSTM `0.7859`, XGBoost `0.6816`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `65.0`, LSTM `0.8056`, XGBoost `0.7193`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.5`, LSTM `0.8054`, XGBoost `0.7193`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.7961`, XGBoost `0.7109`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.7955`, XGBoost `0.7109`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.7901`, XGBoost `0.7069`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `65.5`, LSTM `0.8010`, XGBoost `0.7193`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.7887`, XGBoost `0.7071`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.7935`, XGBoost `0.7124`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
