# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-nemiga-bo3-HBPh0RFmxqP1tE9QMaq3nA/heroic-vs-nemiga-m2-mirage.csv`
- round_num: `4`
- rows: `156`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 156 | 1.000 | 0.049985 | 0.116175 | -0.066190 | 156 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 156 | 1.000 | 0.049985 | 0.116175 | -0.066190 | 156 | 0 | 1.000000 | 1.000000 |
| strong utility action | 101 | 0.647 | 0.069060 | 0.162151 | -0.093091 | 101 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 91 | 0.583 | 0.067742 | 0.153007 | -0.085265 | 91 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.064 | 0.081057 | 0.245360 | -0.164304 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 156 | 1.000 | 0.049985 | 0.116175 | -0.066190 | 156 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `53.0s`, rows `91`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `24.0`, LSTM `0.0630`, XGBoost `0.2616`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.0722`, XGBoost `0.2616`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.0756`, XGBoost `0.2590`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.0807`, XGBoost `0.2616`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `2.0`, LSTM `0.0642`, XGBoost `0.2446`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `2.5`, LSTM `0.0654`, XGBoost `0.2446`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `3.0`, LSTM `0.0669`, XGBoost `0.2446`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `1.5`, LSTM `0.0677`, XGBoost `0.2446`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `23.5`, LSTM `0.0853`, XGBoost `0.2616`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.0894`, XGBoost `0.2616`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
