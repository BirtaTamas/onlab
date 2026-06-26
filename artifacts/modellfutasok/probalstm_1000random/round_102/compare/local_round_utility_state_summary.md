# Local Round Utility State Analysis

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-tyloo-vs-rare-atom-bo3-8GB1HWZtKOlh9_707n2A62/tyloo-vs-rare-atom-m2-inferno.csv`
- round_num: `9`
- rows: `119`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 119 | 1.000 | 0.555657 | 0.483291 | 0.072366 | 97 | 22 | 0.512605 | 0.470588 |
| active/recent utility | 119 | 1.000 | 0.555657 | 0.483291 | 0.072366 | 97 | 22 | 0.512605 | 0.470588 |
| strong utility action | 113 | 0.950 | 0.547561 | 0.472009 | 0.075552 | 95 | 18 | 0.486726 | 0.442478 |
| utility damage | 30 | 0.252 | 0.578250 | 0.538826 | 0.039425 | 23 | 7 | 0.666667 | 0.633333 |
| active smoke/inferno | 104 | 0.874 | 0.543449 | 0.466241 | 0.077208 | 86 | 18 | 0.442308 | 0.394231 |
| recent utility last 5s | 21 | 0.176 | 0.524929 | 0.454414 | 0.070515 | 21 | 0 | 0.523810 | 0.523810 |
| flash effect present | 119 | 1.000 | 0.555657 | 0.483291 | 0.072366 | 97 | 22 | 0.512605 | 0.470588 |

## Active Smoke/Inferno Intervals

- `7.5s` - `59.0s`, rows `104`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `49.0`, LSTM `0.4904`, XGBoost `0.2269`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.4881`, XGBoost `0.2269`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.5050`, XGBoost `0.2473`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.5016`, XGBoost `0.2473`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.4905`, XGBoost `0.2496`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.4879`, XGBoost `0.2473`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.4849`, XGBoost `0.2532`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.4967`, XGBoost `0.2754`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.4732`, XGBoost `0.2531`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.4710`, XGBoost `0.2532`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
