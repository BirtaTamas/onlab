# Local Round Utility State Analysis

- csv_path: `processed_full\esl_pro_league_season_21\esl-pro-league-season-21-vitality-vs-the-mongolz-bo3-7VmOOQFfF_Xgx4vOG4cYIY\vitality-vs-the-mongolz-m1-anubis.csv`
- round_num: `12`
- rows: `176`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 176 | 1.000 | 0.234598 | 0.297838 | -0.063240 | 155 | 21 | 0.812500 | 0.806818 |
| active/recent utility | 176 | 1.000 | 0.234598 | 0.297838 | -0.063240 | 155 | 21 | 0.812500 | 0.806818 |
| strong utility action | 147 | 0.835 | 0.247416 | 0.307893 | -0.060477 | 127 | 20 | 0.775510 | 0.768707 |
| utility damage | 8 | 0.045 | 0.285360 | 0.317593 | -0.032234 | 6 | 2 | 1.000000 | 1.000000 |
| active smoke/inferno | 147 | 0.835 | 0.247416 | 0.307893 | -0.060477 | 127 | 20 | 0.775510 | 0.768707 |
| recent utility last 5s | 8 | 0.045 | 0.285360 | 0.317593 | -0.032234 | 6 | 2 | 1.000000 | 1.000000 |
| flash effect present | 176 | 1.000 | 0.234598 | 0.297838 | -0.063240 | 155 | 21 | 0.812500 | 0.806818 |

## Active Smoke/Inferno Intervals

- `7.5s` - `63.5s`, rows `113`
- `71.0s` - `87.5s`, rows `34`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `43.0`, LSTM `0.0483`, XGBoost `0.3105`, closer `lstm`, smoke `3`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.0773`, XGBoost `0.3382`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.0561`, XGBoost `0.3105`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.0946`, XGBoost `0.3384`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.1228`, XGBoost `0.3384`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.0244`, XGBoost `0.2244`, closer `lstm`, smoke `3`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.0304`, XGBoost `0.2244`, closer `lstm`, smoke `3`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.0318`, XGBoost `0.2231`, closer `lstm`, smoke `3`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.1554`, XGBoost `0.3415`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.0229`, XGBoost `0.2041`, closer `lstm`, smoke `3`, inferno `3`, utility_damage `0.0`, recent_utility `0`
