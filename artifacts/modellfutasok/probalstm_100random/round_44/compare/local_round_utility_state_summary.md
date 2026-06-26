# Local Round Utility State Analysis

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-lynn-vision-vs-tyloo-bo3-tXRL8tbpb2Kb27VbUgWfK1/lynn-vision-vs-tyloo-m2-ancient.csv`
- round_num: `10`
- rows: `187`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 187 | 1.000 | 0.836170 | 0.837637 | -0.001467 | 82 | 105 | 1.000000 | 1.000000 |
| active/recent utility | 187 | 1.000 | 0.836170 | 0.837637 | -0.001467 | 82 | 105 | 1.000000 | 1.000000 |
| strong utility action | 161 | 0.861 | 0.838525 | 0.838899 | -0.000373 | 75 | 86 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.053 | 0.771804 | 0.739812 | 0.031992 | 9 | 1 | 1.000000 | 1.000000 |
| active smoke/inferno | 151 | 0.807 | 0.843821 | 0.845720 | -0.001900 | 65 | 86 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.053 | 0.758565 | 0.735892 | 0.022673 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 187 | 1.000 | 0.836170 | 0.837637 | -0.001467 | 82 | 105 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `55.5s`, rows `99`
- `64.0s` - `70.5s`, rows `14`
- `74.5s` - `93.0s`, rows `38`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `76.5`, LSTM `0.7757`, XGBoost `0.8431`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.0`, LSTM `0.7771`, XGBoost `0.8431`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `75.0`, LSTM `0.7802`, XGBoost `0.8431`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.7943`, XGBoost `0.7331`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.7948`, XGBoost `0.7341`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `29.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.7924`, XGBoost `0.7331`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `76.0`, LSTM `0.7856`, XGBoost `0.8431`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.7909`, XGBoost `0.7341`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `29.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.7903`, XGBoost `0.7341`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `29.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.7893`, XGBoost `0.7341`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `29.0`, recent_utility `0`
