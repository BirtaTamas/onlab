# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-faze-vs-aurora-bo3-OcxcOl9bFIHQQ2588nwUWG/faze-vs-aurora-m1-inferno.csv`
- round_num: `18`
- rows: `115`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 115 | 1.000 | 0.671421 | 0.675889 | -0.004468 | 61 | 54 | 1.000000 | 1.000000 |
| active/recent utility | 115 | 1.000 | 0.671421 | 0.675889 | -0.004468 | 61 | 54 | 1.000000 | 1.000000 |
| strong utility action | 94 | 0.817 | 0.688085 | 0.700493 | -0.012408 | 40 | 54 | 1.000000 | 1.000000 |
| utility damage | 14 | 0.122 | 0.663875 | 0.649188 | 0.014688 | 12 | 2 | 1.000000 | 1.000000 |
| active smoke/inferno | 94 | 0.817 | 0.688085 | 0.700493 | -0.012408 | 40 | 54 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 115 | 1.000 | 0.671421 | 0.675889 | -0.004468 | 61 | 54 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.5s` - `57.0s`, rows `94`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `41.5`, LSTM `0.5890`, XGBoost `0.6556`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.5909`, XGBoost `0.6570`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `8.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.5920`, XGBoost `0.6570`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `8.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.5923`, XGBoost `0.6570`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `8.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.5924`, XGBoost `0.6570`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `8.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.5972`, XGBoost `0.6570`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `8.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.5972`, XGBoost `0.6570`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `8.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.5981`, XGBoost `0.6570`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `8.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.5989`, XGBoost `0.6570`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `8.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.7511`, XGBoost `0.8086`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
