# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-falcons-vs-natus-vincere-bo3-z3OpWwYDPa33wwfDY8_B1Q/falcons-vs-natus-vincere-m1-nuke.csv`
- round_num: `7`
- rows: `158`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 158 | 1.000 | 0.436680 | 0.492824 | -0.056143 | 143 | 15 | 0.348101 | 0.348101 |
| active/recent utility | 158 | 1.000 | 0.436680 | 0.492824 | -0.056143 | 143 | 15 | 0.348101 | 0.348101 |
| strong utility action | 125 | 0.791 | 0.424063 | 0.486644 | -0.062581 | 111 | 14 | 0.360000 | 0.360000 |
| utility damage | 20 | 0.127 | 0.618741 | 0.701127 | -0.082386 | 20 | 0 | 0.000000 | 0.000000 |
| active smoke/inferno | 125 | 0.791 | 0.424063 | 0.486644 | -0.062581 | 111 | 14 | 0.360000 | 0.360000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 158 | 1.000 | 0.436680 | 0.492824 | -0.056143 | 143 | 15 | 0.348101 | 0.348101 |

## Active Smoke/Inferno Intervals

- `8.0s` - `47.5s`, rows `80`
- `52.5s` - `74.5s`, rows `45`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `54.0`, LSTM `0.2158`, XGBoost `0.3580`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.2266`, XGBoost `0.3560`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.2534`, XGBoost `0.3813`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.2411`, XGBoost `0.3545`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.2450`, XGBoost `0.3560`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.2458`, XGBoost `0.3554`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.5`, LSTM `0.2461`, XGBoost `0.3554`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.5813`, XGBoost `0.6890`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.5820`, XGBoost `0.6890`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.6074`, XGBoost `0.7135`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
