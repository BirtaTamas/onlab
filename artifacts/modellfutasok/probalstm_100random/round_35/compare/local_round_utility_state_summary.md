# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-spirit-vs-heroic-bo3-NSK1XoxyhXK-sIe0204dzp/spirit-vs-heroic-m2-nuke.csv`
- round_num: `5`
- rows: `148`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 148 | 1.000 | 0.600410 | 0.707581 | -0.107171 | 1 | 147 | 0.702703 | 1.000000 |
| active/recent utility | 148 | 1.000 | 0.600410 | 0.707581 | -0.107171 | 1 | 147 | 0.702703 | 1.000000 |
| strong utility action | 113 | 0.764 | 0.570672 | 0.701618 | -0.130945 | 0 | 113 | 0.752212 | 1.000000 |
| utility damage | 10 | 0.068 | 0.542718 | 0.636607 | -0.093889 | 0 | 10 | 0.800000 | 1.000000 |
| active smoke/inferno | 113 | 0.764 | 0.570672 | 0.701618 | -0.130945 | 0 | 113 | 0.752212 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 148 | 1.000 | 0.600410 | 0.707581 | -0.107171 | 1 | 147 | 0.702703 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `64.0s`, rows `113`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `44.0`, LSTM `0.5329`, XGBoost `0.7633`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.5348`, XGBoost `0.7633`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.5378`, XGBoost `0.7643`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.5416`, XGBoost `0.7637`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.5470`, XGBoost `0.7633`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.5595`, XGBoost `0.7686`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.5560`, XGBoost `0.7617`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.5610`, XGBoost `0.7631`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.5649`, XGBoost `0.7641`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.5665`, XGBoost `0.7641`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
