# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-vitality-vs-mouz-bo3-kZzxcq2ibUgPOmQh0hZOgn/vitality-vs-mouz-m2-train.csv`
- round_num: `9`
- rows: `285`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 285 | 1.000 | 0.554702 | 0.647246 | -0.092545 | 11 | 274 | 0.750877 | 0.863158 |
| active/recent utility | 285 | 1.000 | 0.554702 | 0.647246 | -0.092545 | 11 | 274 | 0.750877 | 0.863158 |
| strong utility action | 151 | 0.530 | 0.584437 | 0.656491 | -0.072054 | 10 | 141 | 0.887417 | 0.927152 |
| utility damage | 31 | 0.109 | 0.545377 | 0.577589 | -0.032211 | 4 | 27 | 0.741935 | 0.806452 |
| active smoke/inferno | 148 | 0.519 | 0.581499 | 0.654740 | -0.073241 | 10 | 138 | 0.885135 | 0.925676 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 285 | 1.000 | 0.554702 | 0.647246 | -0.092545 | 11 | 274 | 0.750877 | 0.863158 |

## Active Smoke/Inferno Intervals

- `6.0s` - `67.0s`, rows `123`
- `92.5s` - `97.5s`, rows `11`
- `101.5s` - `108.0s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `107.0`, LSTM `0.5131`, XGBoost `0.7098`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `32.0`, recent_utility `0`
- seconds `108.0`, LSTM `0.5139`, XGBoost `0.7085`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `40.0`, recent_utility `0`
- seconds `107.5`, LSTM `0.5193`, XGBoost `0.7098`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `32.0`, recent_utility `0`
- seconds `106.5`, LSTM `0.5172`, XGBoost `0.7007`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `32.0`, recent_utility `0`
- seconds `106.0`, LSTM `0.5208`, XGBoost `0.6922`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `32.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.5860`, XGBoost `0.7550`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.5874`, XGBoost `0.7555`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.5888`, XGBoost `0.7550`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.5919`, XGBoost `0.7566`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.5938`, XGBoost `0.7550`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
