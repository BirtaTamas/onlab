# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-saw-vs-big-bo3-Eh5yMCium2D2NNwnLk7jHb/saw-vs-big-m1-ancient.csv`
- round_num: `2`
- rows: `223`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 223 | 1.000 | 0.146985 | 0.161216 | -0.014231 | 172 | 51 | 1.000000 | 0.995516 |
| active/recent utility | 223 | 1.000 | 0.146985 | 0.161216 | -0.014231 | 172 | 51 | 1.000000 | 0.995516 |
| strong utility action | 128 | 0.574 | 0.232583 | 0.251272 | -0.018689 | 80 | 48 | 1.000000 | 0.992188 |
| utility damage | 10 | 0.045 | 0.273507 | 0.308160 | -0.034653 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 128 | 0.574 | 0.232583 | 0.251272 | -0.018689 | 80 | 48 | 1.000000 | 0.992188 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 223 | 1.000 | 0.146985 | 0.161216 | -0.014231 | 172 | 51 | 1.000000 | 0.995516 |

## Active Smoke/Inferno Intervals

- `6.5s` - `70.0s`, rows `128`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `54.0`, LSTM `0.3094`, XGBoost `0.5683`, closer `lstm`, smoke `5`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.1407`, XGBoost `0.3527`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `54.5`, LSTM `0.1155`, XGBoost `0.2785`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.1296`, XGBoost `0.2754`, closer `lstm`, smoke `5`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.1391`, XGBoost `0.2810`, closer `lstm`, smoke `5`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.1340`, XGBoost `0.2738`, closer `lstm`, smoke `5`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `6.5`, LSTM `0.1387`, XGBoost `0.2756`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.1503`, XGBoost `0.2772`, closer `lstm`, smoke `5`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `7.0`, LSTM `0.1525`, XGBoost `0.2756`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `7.5`, LSTM `0.1530`, XGBoost `0.2761`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
