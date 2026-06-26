# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-eternal-fire-vs-flyquest-bo3-bOv4otMGdpLsO1VdhzI_AV/eternal-fire-vs-flyquest-m1-inferno.csv`
- round_num: `4`
- rows: `132`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 132 | 1.000 | 0.789606 | 0.792519 | -0.002913 | 48 | 84 | 1.000000 | 1.000000 |
| active/recent utility | 132 | 1.000 | 0.789606 | 0.792519 | -0.002913 | 48 | 84 | 1.000000 | 1.000000 |
| strong utility action | 123 | 0.932 | 0.792037 | 0.796831 | -0.004794 | 39 | 84 | 1.000000 | 1.000000 |
| utility damage | 11 | 0.083 | 0.719013 | 0.736296 | -0.017284 | 0 | 11 | 1.000000 | 1.000000 |
| active smoke/inferno | 112 | 0.848 | 0.800087 | 0.802933 | -0.002846 | 38 | 74 | 1.000000 | 1.000000 |
| recent utility last 5s | 11 | 0.083 | 0.710078 | 0.734706 | -0.024628 | 1 | 10 | 1.000000 | 1.000000 |
| flash effect present | 132 | 1.000 | 0.789606 | 0.792519 | -0.002913 | 48 | 84 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.0s` - `65.5s`, rows `112`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `49.0`, LSTM `0.8432`, XGBoost `0.7295`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.8367`, XGBoost `0.7274`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.8329`, XGBoost `0.7271`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.8516`, XGBoost `0.7546`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.8249`, XGBoost `0.7295`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.8155`, XGBoost `0.7265`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.8179`, XGBoost `0.7295`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.6523`, XGBoost `0.7340`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.8166`, XGBoost `0.7352`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.8520`, XGBoost `0.7748`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
