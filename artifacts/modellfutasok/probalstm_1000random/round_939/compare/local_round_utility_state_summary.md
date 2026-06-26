# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-the-mongolz-bo3-7VmOOQFfF_Xgx4vOG4cYIY/vitality-vs-the-mongolz-m3-inferno.csv`
- round_num: `7`
- rows: `153`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 153 | 1.000 | 0.067693 | 0.122540 | -0.054847 | 137 | 16 | 1.000000 | 1.000000 |
| active/recent utility | 153 | 1.000 | 0.067693 | 0.122540 | -0.054847 | 137 | 16 | 1.000000 | 1.000000 |
| strong utility action | 91 | 0.595 | 0.070244 | 0.119927 | -0.049683 | 75 | 16 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 79 | 0.516 | 0.061079 | 0.097245 | -0.036166 | 63 | 16 | 1.000000 | 1.000000 |
| recent utility last 5s | 20 | 0.131 | 0.098669 | 0.200912 | -0.102243 | 20 | 0 | 1.000000 | 1.000000 |
| flash effect present | 153 | 1.000 | 0.067693 | 0.122540 | -0.054847 | 137 | 16 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.5s` - `17.0s`, rows `14`
- `18.5s` - `25.0s`, rows `14`
- `51.0s` - `76.0s`, rows `51`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `1.0`, LSTM `0.0976`, XGBoost `0.3008`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `15.0`, LSTM `0.2008`, XGBoost `0.3961`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `1.5`, LSTM `0.1093`, XGBoost `0.3008`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `2.0`, LSTM `0.1153`, XGBoost `0.3008`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `2.5`, LSTM `0.1217`, XGBoost `0.2951`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `3.0`, LSTM `0.1288`, XGBoost `0.2951`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `11.0`, LSTM `0.1355`, XGBoost `0.2929`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.1400`, XGBoost `0.2973`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `3.5`, LSTM `0.1385`, XGBoost `0.2951`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `10.5`, LSTM `0.1385`, XGBoost `0.2924`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
