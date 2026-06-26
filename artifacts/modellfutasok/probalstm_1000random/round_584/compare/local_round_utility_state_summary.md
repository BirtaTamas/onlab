# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2_finals/blast-bounty-2025-season-2-finals-the-mongolz-vs-aurora-bo3-BL3Rcvf733R8cy7TtLY5HE/the-mongolz-vs-aurora-m1-dust2.csv`
- round_num: `5`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.801181 | 0.736605 | 0.064576 | 229 | 1 | 0.995652 | 0.604348 |
| active/recent utility | 230 | 1.000 | 0.801181 | 0.736605 | 0.064576 | 229 | 1 | 0.995652 | 0.604348 |
| strong utility action | 162 | 0.704 | 0.740308 | 0.665732 | 0.074576 | 161 | 1 | 0.993827 | 0.444444 |
| utility damage | 30 | 0.130 | 0.685094 | 0.623763 | 0.061331 | 30 | 0 | 1.000000 | 0.333333 |
| active smoke/inferno | 154 | 0.670 | 0.747345 | 0.675537 | 0.071808 | 153 | 1 | 0.993506 | 0.467532 |
| recent utility last 5s | 10 | 0.043 | 0.604177 | 0.475363 | 0.128814 | 10 | 0 | 1.000000 | 0.000000 |
| flash effect present | 230 | 1.000 | 0.801181 | 0.736605 | 0.064576 | 229 | 1 | 0.995652 | 0.604348 |

## Active Smoke/Inferno Intervals

- `4.5s` - `81.0s`, rows `154`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `31.5`, LSTM `0.6473`, XGBoost `0.4827`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.6435`, XGBoost `0.4827`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.6403`, XGBoost `0.4797`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.6396`, XGBoost `0.4827`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.6366`, XGBoost `0.4827`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.6364`, XGBoost `0.4827`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.6341`, XGBoost `0.4828`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.6317`, XGBoost `0.4805`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.6298`, XGBoost `0.4808`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.6263`, XGBoost `0.4808`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `0.0`, recent_utility `0`
