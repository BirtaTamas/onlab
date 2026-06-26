# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2_finals/blast-bounty-2025-season-2-finals-vitality-vs-the-mongolz-bo3-JVS9HKMAkaZTRHkoiRSMP6/vitality-vs-the-mongolz-m1-mirage.csv`
- round_num: `6`
- rows: `223`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 223 | 1.000 | 0.471229 | 0.517892 | -0.046663 | 178 | 45 | 0.726457 | 0.735426 |
| active/recent utility | 223 | 1.000 | 0.471229 | 0.517892 | -0.046663 | 178 | 45 | 0.726457 | 0.735426 |
| strong utility action | 161 | 0.722 | 0.478607 | 0.538502 | -0.059894 | 134 | 27 | 0.695652 | 0.689441 |
| utility damage | 13 | 0.058 | 0.391899 | 0.476088 | -0.084189 | 12 | 1 | 0.769231 | 0.769231 |
| active smoke/inferno | 150 | 0.673 | 0.482785 | 0.544244 | -0.061458 | 123 | 27 | 0.673333 | 0.666667 |
| recent utility last 5s | 11 | 0.049 | 0.421638 | 0.460205 | -0.038567 | 11 | 0 | 1.000000 | 1.000000 |
| flash effect present | 223 | 1.000 | 0.471229 | 0.517892 | -0.046663 | 178 | 45 | 0.726457 | 0.735426 |

## Active Smoke/Inferno Intervals

- `7.0s` - `33.0s`, rows `53`
- `52.5s` - `78.5s`, rows `53`
- `83.0s` - `104.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `78.0`, LSTM `0.2363`, XGBoost `0.5159`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.2975`, XGBoost `0.5151`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.5`, LSTM `0.2170`, XGBoost `0.4256`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `74.0`, LSTM `0.3826`, XGBoost `0.5864`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `75.0`, LSTM `0.4952`, XGBoost `0.6962`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `58.5`, LSTM `0.2669`, XGBoost `0.4581`, closer `lstm`, smoke `4`, inferno `3`, utility_damage `32.0`, recent_utility `0`
- seconds `58.0`, LSTM `0.2684`, XGBoost `0.4581`, closer `lstm`, smoke `4`, inferno `3`, utility_damage `32.0`, recent_utility `0`
- seconds `76.5`, LSTM `0.0893`, XGBoost `0.2669`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `78.5`, LSTM `0.3362`, XGBoost `0.5133`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `76.0`, LSTM `0.0829`, XGBoost `0.2586`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
