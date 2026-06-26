# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-vitality-vs-the-mongolz-bo3-vhm_UWcBfYfYcOeLh9JIDA/vitality-vs-the-mongolz-m2-dust2.csv`
- round_num: `8`
- rows: `157`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 157 | 1.000 | 0.558506 | 0.608182 | -0.049676 | 2 | 155 | 0.439490 | 1.000000 |
| active/recent utility | 157 | 1.000 | 0.558506 | 0.608182 | -0.049676 | 2 | 155 | 0.439490 | 1.000000 |
| strong utility action | 153 | 0.975 | 0.559545 | 0.610433 | -0.050887 | 1 | 152 | 0.424837 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 146 | 0.930 | 0.562902 | 0.614823 | -0.051921 | 1 | 145 | 0.445205 | 1.000000 |
| recent utility last 5s | 10 | 0.064 | 0.490062 | 0.526761 | -0.036699 | 0 | 10 | 0.000000 | 1.000000 |
| flash effect present | 157 | 1.000 | 0.558506 | 0.608182 | -0.049676 | 2 | 155 | 0.439490 | 1.000000 |

## Active Smoke/Inferno Intervals

- `5.5s` - `78.0s`, rows `146`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `78.0`, LSTM `0.5731`, XGBoost `0.7441`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.3590`, XGBoost `0.5126`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.5658`, XGBoost `0.7165`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.3757`, XGBoost `0.5132`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.3760`, XGBoost `0.5126`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.5870`, XGBoost `0.7185`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.3827`, XGBoost `0.5132`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `61.0`, LSTM `0.5874`, XGBoost `0.7156`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.3937`, XGBoost `0.5132`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `64.5`, LSTM `0.5767`, XGBoost `0.6931`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
