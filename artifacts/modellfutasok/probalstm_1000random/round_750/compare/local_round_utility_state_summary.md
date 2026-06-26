# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-vitality-vs-gamerlegion-bo3-HfAhqHTEhpe_HlObeToa76/vitality-vs-gamerlegion-m2-dust2.csv`
- round_num: `11`
- rows: `230`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.292886 | 0.424558 | -0.131673 | 216 | 14 | 0.904348 | 0.800000 |
| active/recent utility | 230 | 1.000 | 0.292886 | 0.424558 | -0.131673 | 216 | 14 | 0.904348 | 0.800000 |
| strong utility action | 220 | 0.957 | 0.298611 | 0.432896 | -0.134285 | 206 | 14 | 0.900000 | 0.795455 |
| utility damage | 20 | 0.087 | 0.611479 | 0.683851 | -0.072372 | 10 | 10 | 0.500000 | 0.450000 |
| active smoke/inferno | 203 | 0.883 | 0.294402 | 0.427626 | -0.133223 | 189 | 14 | 0.891626 | 0.812808 |
| recent utility last 5s | 20 | 0.087 | 0.471258 | 0.593413 | -0.122155 | 15 | 5 | 0.750000 | 0.150000 |
| flash effect present | 230 | 1.000 | 0.292886 | 0.424558 | -0.131673 | 216 | 14 | 0.904348 | 0.800000 |

## Active Smoke/Inferno Intervals

- `10.0s` - `86.0s`, rows `153`
- `90.0s` - `114.5s`, rows `50`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `67.0`, LSTM `0.1465`, XGBoost `0.5272`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `67.5`, LSTM `0.1668`, XGBoost `0.5238`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.0`, LSTM `0.1821`, XGBoost `0.5167`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `68.0`, LSTM `0.1923`, XGBoost `0.5269`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.2050`, XGBoost `0.5260`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.5`, LSTM `0.2182`, XGBoost `0.5240`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `69.0`, LSTM `0.2254`, XGBoost `0.5260`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.2369`, XGBoost `0.5192`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.0`, LSTM `0.2525`, XGBoost `0.5202`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `65.5`, LSTM `0.2570`, XGBoost `0.5167`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `1`
