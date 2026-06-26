# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-heroic-vs-flyquest-bo3-ErQHzvBcWPHiA-H04IjPMf/heroic-vs-flyquest-m2-anubis.csv`
- round_num: `22`
- rows: `199`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 199 | 1.000 | 0.133050 | 0.187442 | -0.054392 | 199 | 0 | 1.000000 | 0.809045 |
| active/recent utility | 199 | 1.000 | 0.133050 | 0.187442 | -0.054392 | 199 | 0 | 1.000000 | 0.809045 |
| strong utility action | 158 | 0.794 | 0.143970 | 0.208623 | -0.064653 | 158 | 0 | 1.000000 | 0.797468 |
| utility damage | 23 | 0.116 | 0.119366 | 0.248081 | -0.128715 | 23 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 138 | 0.693 | 0.130859 | 0.202172 | -0.071313 | 138 | 0 | 1.000000 | 0.789855 |
| recent utility last 5s | 20 | 0.101 | 0.234435 | 0.253135 | -0.018700 | 20 | 0 | 1.000000 | 0.850000 |
| flash effect present | 199 | 1.000 | 0.133050 | 0.187442 | -0.054392 | 199 | 0 | 1.000000 | 0.809045 |

## Active Smoke/Inferno Intervals

- `9.0s` - `75.0s`, rows `133`
- `97.0s` - `99.0s`, rows `5`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `23.0`, LSTM `0.4697`, XGBoost `0.7125`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `5.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.4894`, XGBoost `0.7112`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `5.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.0419`, XGBoost `0.2360`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `6.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.0439`, XGBoost `0.2365`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `9.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.0467`, XGBoost `0.2355`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `2.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.0445`, XGBoost `0.2312`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.0449`, XGBoost `0.2301`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.0544`, XGBoost `0.2335`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `10.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.0459`, XGBoost `0.2208`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.0611`, XGBoost `0.2357`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `10.0`, recent_utility `0`
