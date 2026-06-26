# Local Round Utility State Analysis

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-the-huns-vs-ninja-bo3-8zmdVWrC356tnVH1OFLf2Y/the-huns-vs-ninja-m1-ancient.csv`
- round_num: `16`
- rows: `163`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 163 | 1.000 | 0.266602 | 0.363814 | -0.097212 | 143 | 20 | 0.742331 | 0.699387 |
| active/recent utility | 163 | 1.000 | 0.266602 | 0.363814 | -0.097212 | 143 | 20 | 0.742331 | 0.699387 |
| strong utility action | 147 | 0.902 | 0.279476 | 0.371577 | -0.092101 | 128 | 19 | 0.714286 | 0.666667 |
| utility damage | 14 | 0.086 | 0.590759 | 0.577038 | 0.013721 | 4 | 10 | 0.071429 | 0.214286 |
| active smoke/inferno | 137 | 0.840 | 0.271443 | 0.363614 | -0.092171 | 118 | 19 | 0.693431 | 0.642336 |
| recent utility last 5s | 10 | 0.061 | 0.389532 | 0.480676 | -0.091144 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 163 | 1.000 | 0.266602 | 0.363814 | -0.097212 | 143 | 20 | 0.742331 | 0.699387 |

## Active Smoke/Inferno Intervals

- `6.5s` - `52.5s`, rows `93`
- `54.5s` - `76.0s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `74.0`, LSTM `0.0426`, XGBoost `0.2842`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `19.0`, recent_utility `0`
- seconds `74.5`, LSTM `0.0449`, XGBoost `0.2836`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `19.0`, recent_utility `0`
- seconds `75.5`, LSTM `0.0537`, XGBoost `0.2855`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `19.0`, recent_utility `0`
- seconds `75.0`, LSTM `0.0544`, XGBoost `0.2836`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `19.0`, recent_utility `0`
- seconds `76.0`, LSTM `0.0634`, XGBoost `0.2850`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `19.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.0199`, XGBoost `0.2393`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.0219`, XGBoost `0.2393`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.0373`, XGBoost `0.2525`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.0396`, XGBoost `0.2522`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.0262`, XGBoost `0.2381`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
