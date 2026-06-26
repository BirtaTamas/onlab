# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-aurora-vs-faze-bo3-ZgdBOa3Yi0KCkwa_Ap1ef3/aurora-vs-faze-m2-train.csv`
- round_num: `10`
- rows: `154`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 154 | 1.000 | 0.484574 | 0.643874 | -0.159300 | 154 | 0 | 0.305195 | 0.155844 |
| active/recent utility | 154 | 1.000 | 0.484574 | 0.643874 | -0.159300 | 154 | 0 | 0.305195 | 0.155844 |
| strong utility action | 138 | 0.896 | 0.518160 | 0.685092 | -0.166932 | 138 | 0 | 0.224638 | 0.108696 |
| utility damage | 32 | 0.208 | 0.454889 | 0.579664 | -0.124774 | 32 | 0 | 0.406250 | 0.406250 |
| active smoke/inferno | 125 | 0.812 | 0.535773 | 0.707343 | -0.171570 | 125 | 0 | 0.144000 | 0.096000 |
| recent utility last 5s | 10 | 0.065 | 0.433615 | 0.564601 | -0.130987 | 10 | 0 | 1.000000 | 0.000000 |
| flash effect present | 154 | 1.000 | 0.484574 | 0.643874 | -0.159300 | 154 | 0 | 0.305195 | 0.155844 |

## Active Smoke/Inferno Intervals

- `8.5s` - `70.5s`, rows `125`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `29.0`, LSTM `0.5374`, XGBoost `0.7482`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.5354`, XGBoost `0.7460`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.5390`, XGBoost `0.7485`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.5387`, XGBoost `0.7471`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.5390`, XGBoost `0.7471`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.5410`, XGBoost `0.7485`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.5409`, XGBoost `0.7472`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.5418`, XGBoost `0.7479`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.5574`, XGBoost `0.7635`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.5367`, XGBoost `0.7406`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
