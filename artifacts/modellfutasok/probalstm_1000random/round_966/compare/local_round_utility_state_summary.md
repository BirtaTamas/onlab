# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-nrg-vs-fluxo-bo3-aFv0UX6WO0txoeY8N630nT/nrg-vs-fluxo-m1-nuke.csv`
- round_num: `2`
- rows: `189`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 189 | 1.000 | 0.443824 | 0.544259 | -0.100435 | 168 | 21 | 0.423280 | 0.179894 |
| active/recent utility | 189 | 1.000 | 0.443824 | 0.544259 | -0.100435 | 168 | 21 | 0.423280 | 0.179894 |
| strong utility action | 134 | 0.709 | 0.493692 | 0.562088 | -0.068396 | 113 | 21 | 0.305970 | 0.119403 |
| utility damage | 10 | 0.053 | 0.553042 | 0.613737 | -0.060695 | 10 | 0 | 0.000000 | 0.000000 |
| active smoke/inferno | 134 | 0.709 | 0.493692 | 0.562088 | -0.068396 | 113 | 21 | 0.305970 | 0.119403 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 189 | 1.000 | 0.443824 | 0.544259 | -0.100435 | 168 | 21 | 0.423280 | 0.179894 |

## Active Smoke/Inferno Intervals

- `8.0s` - `69.0s`, rows `123`
- `82.5s` - `87.5s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `83.0`, LSTM `0.1675`, XGBoost `0.5881`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `83.5`, LSTM `0.1865`, XGBoost `0.5918`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `86.5`, LSTM `0.1535`, XGBoost `0.5508`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `22.0`, recent_utility `0`
- seconds `86.0`, LSTM `0.1535`, XGBoost `0.5505`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `22.0`, recent_utility `0`
- seconds `85.5`, LSTM `0.1584`, XGBoost `0.5505`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `22.0`, recent_utility `0`
- seconds `87.0`, LSTM `0.1600`, XGBoost `0.5458`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `22.0`, recent_utility `0`
- seconds `87.5`, LSTM `0.1366`, XGBoost `0.5113`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `22.0`, recent_utility `0`
- seconds `82.5`, LSTM `0.2167`, XGBoost `0.5887`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `85.0`, LSTM `0.1878`, XGBoost `0.5505`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `22.0`, recent_utility `0`
- seconds `84.0`, LSTM `0.2530`, XGBoost `0.5984`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
