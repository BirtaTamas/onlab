# Local Round Utility State Analysis

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-lynn-vision-vs-tyloo-bo3-tXRL8tbpb2Kb27VbUgWfK1/lynn-vision-vs-tyloo-m2-ancient.csv`
- round_num: `9`
- rows: `100`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 100 | 1.000 | 0.565050 | 0.502318 | 0.062732 | 76 | 24 | 0.770000 | 0.730000 |
| active/recent utility | 100 | 1.000 | 0.565050 | 0.502318 | 0.062732 | 76 | 24 | 0.770000 | 0.730000 |
| strong utility action | 79 | 0.790 | 0.592766 | 0.508152 | 0.084614 | 64 | 15 | 0.810127 | 0.721519 |
| utility damage | 28 | 0.280 | 0.571375 | 0.522254 | 0.049121 | 23 | 5 | 0.857143 | 0.857143 |
| active smoke/inferno | 68 | 0.680 | 0.571897 | 0.463199 | 0.108698 | 64 | 4 | 0.808824 | 0.691176 |
| recent utility last 5s | 10 | 0.100 | 0.784410 | 0.846524 | -0.062114 | 0 | 10 | 0.900000 | 1.000000 |
| flash effect present | 100 | 1.000 | 0.565050 | 0.502318 | 0.062732 | 76 | 24 | 0.770000 | 0.730000 |

## Active Smoke/Inferno Intervals

- `6.0s` - `39.5s`, rows `68`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `30.5`, LSTM `0.5565`, XGBoost `0.2835`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.4879`, XGBoost `0.2199`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.4825`, XGBoost `0.2162`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.4804`, XGBoost `0.2151`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.5485`, XGBoost `0.2833`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.4778`, XGBoost `0.2142`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.5412`, XGBoost `0.2823`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.5350`, XGBoost `0.2823`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.5301`, XGBoost `0.2823`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.5279`, XGBoost `0.2823`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
