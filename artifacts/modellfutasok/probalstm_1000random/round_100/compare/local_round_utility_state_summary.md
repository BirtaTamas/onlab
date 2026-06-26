# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-imperial-vs-metizport-bo3-yMtoBsoZq-jiQ0fSUscH7u/imperial-vs-metizport-m2-dust2.csv`
- round_num: `3`
- rows: `171`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 171 | 1.000 | 0.005629 | 0.014505 | -0.008876 | 170 | 1 | 1.000000 | 1.000000 |
| active/recent utility | 171 | 1.000 | 0.005629 | 0.014505 | -0.008876 | 170 | 1 | 1.000000 | 1.000000 |
| strong utility action | 102 | 0.596 | 0.006644 | 0.017739 | -0.011095 | 102 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 102 | 0.596 | 0.006644 | 0.017739 | -0.011095 | 102 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 171 | 1.000 | 0.005629 | 0.014505 | -0.008876 | 170 | 1 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `4.5s` - `11.0s`, rows `14`
- `14.0s` - `57.5s`, rows `88`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `20.5`, LSTM `0.0034`, XGBoost `0.0324`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.0039`, XGBoost `0.0322`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.0040`, XGBoost `0.0321`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.0046`, XGBoost `0.0326`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.0050`, XGBoost `0.0326`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.0054`, XGBoost `0.0330`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.0040`, XGBoost `0.0315`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.0054`, XGBoost `0.0326`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.0055`, XGBoost `0.0326`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.0055`, XGBoost `0.0326`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
