# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-faze-vs-virtuspro-bo3-YK5CkfojMMEdQ3U0_CVA2l/faze-vs-virtuspro-m1-anubis.csv`
- round_num: `17`
- rows: `196`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 196 | 1.000 | 0.539870 | 0.629032 | -0.089162 | 0 | 196 | 0.459184 | 1.000000 |
| active/recent utility | 196 | 1.000 | 0.539870 | 0.629032 | -0.089162 | 0 | 196 | 0.459184 | 1.000000 |
| strong utility action | 178 | 0.908 | 0.545441 | 0.634765 | -0.089324 | 0 | 178 | 0.483146 | 1.000000 |
| utility damage | 20 | 0.102 | 0.718045 | 0.790805 | -0.072761 | 0 | 20 | 0.800000 | 1.000000 |
| active smoke/inferno | 168 | 0.857 | 0.548585 | 0.639526 | -0.090941 | 0 | 168 | 0.494048 | 1.000000 |
| recent utility last 5s | 20 | 0.102 | 0.480329 | 0.567000 | -0.086672 | 0 | 20 | 0.150000 | 1.000000 |
| flash effect present | 196 | 1.000 | 0.539870 | 0.629032 | -0.089162 | 0 | 196 | 0.459184 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.5s` - `58.0s`, rows `98`
- `63.0s` - `97.5s`, rows `70`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `67.5`, LSTM `0.4484`, XGBoost `0.6178`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `67.0`, LSTM `0.4407`, XGBoost `0.5879`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.4496`, XGBoost `0.5924`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.5`, LSTM `0.4512`, XGBoost `0.5912`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `74.5`, LSTM `0.4500`, XGBoost `0.5892`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `78.0`, LSTM `0.4535`, XGBoost `0.5924`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.5`, LSTM `0.4509`, XGBoost `0.5879`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.4819`, XGBoost `0.6178`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `73.5`, LSTM `0.4537`, XGBoost `0.5892`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `74.0`, LSTM `0.4548`, XGBoost `0.5892`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `1`
