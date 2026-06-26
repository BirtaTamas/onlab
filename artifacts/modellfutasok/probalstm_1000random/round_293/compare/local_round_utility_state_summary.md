# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-vitality-vs-the-mongolz-bo3-vhm_UWcBfYfYcOeLh9JIDA/vitality-vs-the-mongolz-m1-mirage.csv`
- round_num: `7`
- rows: `167`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 167 | 1.000 | 0.520317 | 0.533582 | -0.013265 | 82 | 85 | 0.586826 | 0.371257 |
| active/recent utility | 167 | 1.000 | 0.520317 | 0.533582 | -0.013265 | 82 | 85 | 0.586826 | 0.371257 |
| strong utility action | 123 | 0.737 | 0.497770 | 0.512358 | -0.014587 | 62 | 61 | 0.552846 | 0.308943 |
| utility damage | 10 | 0.060 | 0.545618 | 0.489236 | 0.056382 | 9 | 1 | 1.000000 | 0.100000 |
| active smoke/inferno | 123 | 0.737 | 0.497770 | 0.512358 | -0.014587 | 62 | 61 | 0.552846 | 0.308943 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 167 | 1.000 | 0.520317 | 0.533582 | -0.013265 | 82 | 85 | 0.586826 | 0.371257 |

## Active Smoke/Inferno Intervals

- `6.5s` - `56.5s`, rows `101`
- `67.5s` - `74.0s`, rows `14`
- `79.5s` - `83.0s`, rows `8`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `83.0`, LSTM `0.5727`, XGBoost `0.8245`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `82.0`, LSTM `0.6297`, XGBoost `0.8267`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `82.5`, LSTM `0.6301`, XGBoost `0.8267`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `81.0`, LSTM `0.6555`, XGBoost `0.8267`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `81.5`, LSTM `0.6628`, XGBoost `0.8267`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `80.5`, LSTM `0.6652`, XGBoost `0.8267`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `80.0`, LSTM `0.6687`, XGBoost `0.8252`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.6823`, XGBoost `0.8221`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.1810`, XGBoost `0.3116`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.1817`, XGBoost `0.3083`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
