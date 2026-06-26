# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-furia-vs-g2-bo3-_Q5oHsnKyowoqEbJh5o3f8/furia-vs-g2-m2-overpass.csv`
- round_num: `10`
- rows: `222`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 222 | 1.000 | 0.396104 | 0.504125 | -0.108021 | 178 | 44 | 0.468468 | 0.387387 |
| active/recent utility | 222 | 1.000 | 0.396104 | 0.504125 | -0.108021 | 178 | 44 | 0.468468 | 0.387387 |
| strong utility action | 178 | 0.802 | 0.422952 | 0.518069 | -0.095117 | 144 | 34 | 0.443820 | 0.370787 |
| utility damage | 10 | 0.045 | 0.556456 | 0.537641 | 0.018815 | 1 | 9 | 0.000000 | 0.000000 |
| active smoke/inferno | 168 | 0.757 | 0.444820 | 0.531723 | -0.086903 | 134 | 34 | 0.410714 | 0.333333 |
| recent utility last 5s | 10 | 0.045 | 0.055568 | 0.288678 | -0.233110 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 222 | 1.000 | 0.396104 | 0.504125 | -0.108021 | 178 | 44 | 0.468468 | 0.387387 |

## Active Smoke/Inferno Intervals

- `8.0s` - `51.5s`, rows `88`
- `53.5s` - `93.0s`, rows `80`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `64.0`, LSTM `0.1347`, XGBoost `0.5245`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.5`, LSTM `0.1362`, XGBoost `0.5245`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.1470`, XGBoost `0.5237`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `65.0`, LSTM `0.1536`, XGBoost `0.5245`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.1626`, XGBoost `0.5231`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.2173`, XGBoost `0.5224`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.0`, LSTM `0.2335`, XGBoost `0.5224`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.0`, LSTM `0.2393`, XGBoost `0.5224`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.5`, LSTM `0.2488`, XGBoost `0.5224`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.2568`, XGBoost `0.5224`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
