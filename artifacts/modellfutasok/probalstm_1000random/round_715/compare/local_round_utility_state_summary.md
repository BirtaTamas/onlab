# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-the-mongolz-vs-m80-bo3-e7FibL-GpwhFRhM0kGS5r4/the-mongolz-vs-m80-m3-inferno.csv`
- round_num: `14`
- rows: `229`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 229 | 1.000 | 0.472667 | 0.476263 | -0.003597 | 82 | 147 | 0.471616 | 0.886463 |
| active/recent utility | 229 | 1.000 | 0.472667 | 0.476263 | -0.003597 | 82 | 147 | 0.471616 | 0.886463 |
| strong utility action | 44 | 0.192 | 0.496610 | 0.490320 | 0.006290 | 18 | 26 | 0.613636 | 0.909091 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 44 | 0.192 | 0.496610 | 0.490320 | 0.006290 | 18 | 26 | 0.613636 | 0.909091 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 229 | 1.000 | 0.472667 | 0.476263 | -0.003597 | 82 | 147 | 0.471616 | 0.886463 |

## Active Smoke/Inferno Intervals

- `44.0s` - `65.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `58.0`, LSTM `0.4481`, XGBoost `0.4955`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.5157`, XGBoost `0.4837`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.5157`, XGBoost `0.4837`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.5155`, XGBoost `0.4837`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.5153`, XGBoost `0.4837`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.5134`, XGBoost `0.4842`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.5127`, XGBoost `0.4842`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.5104`, XGBoost `0.4837`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.5096`, XGBoost `0.4842`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.5094`, XGBoost `0.4842`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
