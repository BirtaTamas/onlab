# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-liquid-bo3-9v1WXdmzbeO2q7iD5Nu_mP/mouz-vs-liquid-m2-nuke.csv`
- round_num: `7`
- rows: `104`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 104 | 1.000 | 0.013545 | 0.020332 | -0.006787 | 67 | 37 | 1.000000 | 1.000000 |
| active/recent utility | 104 | 1.000 | 0.013545 | 0.020332 | -0.006787 | 67 | 37 | 1.000000 | 1.000000 |
| strong utility action | 68 | 0.654 | 0.015646 | 0.025263 | -0.009617 | 46 | 22 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 58 | 0.558 | 0.016206 | 0.021945 | -0.005739 | 36 | 22 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.096 | 0.012398 | 0.044511 | -0.032113 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 104 | 1.000 | 0.013545 | 0.020332 | -0.006787 | 67 | 37 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `37.0s`, rows `58`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `3.0`, LSTM `0.0107`, XGBoost `0.0441`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `4.5`, LSTM `0.0110`, XGBoost `0.0443`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `3.5`, LSTM `0.0108`, XGBoost `0.0441`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `4.0`, LSTM `0.0109`, XGBoost `0.0441`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `1.5`, LSTM `0.0122`, XGBoost `0.0451`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `2.5`, LSTM `0.0114`, XGBoost `0.0441`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `1.0`, LSTM `0.0132`, XGBoost `0.0453`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `2.0`, LSTM `0.0121`, XGBoost `0.0441`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `5.0`, LSTM `0.0136`, XGBoost `0.0444`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `9.5`, LSTM `0.0148`, XGBoost `0.0444`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
