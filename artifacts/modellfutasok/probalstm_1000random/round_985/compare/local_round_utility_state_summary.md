# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-liquid-bo3-9v1WXdmzbeO2q7iD5Nu_mP/mouz-vs-liquid-m2-nuke.csv`
- round_num: `6`
- rows: `125`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 125 | 1.000 | 0.365179 | 0.367522 | -0.002343 | 56 | 69 | 0.760000 | 0.848000 |
| active/recent utility | 125 | 1.000 | 0.365179 | 0.367522 | -0.002343 | 56 | 69 | 0.760000 | 0.848000 |
| strong utility action | 119 | 0.952 | 0.361899 | 0.366231 | -0.004332 | 56 | 63 | 0.747899 | 0.840336 |
| utility damage | 10 | 0.080 | 0.506658 | 0.386454 | 0.120204 | 0 | 10 | 0.300000 | 1.000000 |
| active smoke/inferno | 109 | 0.872 | 0.352434 | 0.363880 | -0.011446 | 56 | 53 | 0.733945 | 0.825688 |
| recent utility last 5s | 10 | 0.080 | 0.465063 | 0.391858 | 0.073205 | 0 | 10 | 0.900000 | 1.000000 |
| flash effect present | 125 | 1.000 | 0.365179 | 0.367522 | -0.002343 | 56 | 69 | 0.760000 | 0.848000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `62.0s`, rows `109`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `54.5`, LSTM `0.0328`, XGBoost `0.2179`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `57.5`, LSTM `0.0330`, XGBoost `0.2150`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.0366`, XGBoost `0.2179`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.0374`, XGBoost `0.2179`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.0376`, XGBoost `0.2179`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `57.0`, LSTM `0.0378`, XGBoost `0.2179`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.0394`, XGBoost `0.2179`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.0396`, XGBoost `0.2179`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `56.5`, LSTM `0.0404`, XGBoost `0.2179`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `58.0`, LSTM `0.0412`, XGBoost `0.2150`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
