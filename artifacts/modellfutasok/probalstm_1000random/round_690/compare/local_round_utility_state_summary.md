# Local Round Utility State Analysis

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-faze-vs-bcgame-bo3-daIGc_M_y7Qq42fF9AoQhi/faze-vs-bcgame-m2-anubis.csv`
- round_num: `1`
- rows: `134`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 134 | 1.000 | 0.534894 | 0.617436 | -0.082541 | 27 | 107 | 0.649254 | 0.902985 |
| active/recent utility | 134 | 1.000 | 0.534894 | 0.617436 | -0.082541 | 27 | 107 | 0.649254 | 0.902985 |
| strong utility action | 51 | 0.381 | 0.595029 | 0.760318 | -0.165289 | 5 | 46 | 0.764706 | 0.960784 |
| utility damage | 10 | 0.075 | 0.643803 | 0.833205 | -0.189403 | 0 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 51 | 0.381 | 0.595029 | 0.760318 | -0.165289 | 5 | 46 | 0.764706 | 0.960784 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 134 | 1.000 | 0.534894 | 0.617436 | -0.082541 | 27 | 107 | 0.649254 | 0.902985 |

## Active Smoke/Inferno Intervals

- `39.5s` - `64.5s`, rows `51`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `48.0`, LSTM `0.2103`, XGBoost `0.5430`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.1979`, XGBoost `0.5139`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.4256`, XGBoost `0.7143`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.2671`, XGBoost `0.5416`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.5377`, XGBoost `0.8012`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.2759`, XGBoost `0.5390`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.5507`, XGBoost `0.8114`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.3419`, XGBoost `0.5968`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.5631`, XGBoost `0.8159`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.5763`, XGBoost `0.8218`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
