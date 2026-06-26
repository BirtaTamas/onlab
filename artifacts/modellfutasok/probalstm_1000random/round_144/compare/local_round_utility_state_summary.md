# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-furia-vs-fluxo-bo3-cy88FeSpEinhT8XDRxQGHo/furia-vs-fluxo-m2-mirage.csv`
- round_num: `10`
- rows: `235`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 235 | 1.000 | 0.173287 | 0.192347 | -0.019060 | 159 | 76 | 0.765957 | 0.778723 |
| active/recent utility | 235 | 1.000 | 0.173287 | 0.192347 | -0.019060 | 159 | 76 | 0.765957 | 0.778723 |
| strong utility action | 169 | 0.719 | 0.202654 | 0.219890 | -0.017236 | 105 | 64 | 0.745562 | 0.763314 |
| utility damage | 21 | 0.089 | 0.314058 | 0.333640 | -0.019582 | 10 | 11 | 0.523810 | 0.523810 |
| active smoke/inferno | 165 | 0.702 | 0.207312 | 0.222308 | -0.014996 | 101 | 64 | 0.739394 | 0.757576 |
| recent utility last 5s | 10 | 0.043 | 0.013757 | 0.122839 | -0.109082 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 235 | 1.000 | 0.173287 | 0.192347 | -0.019060 | 159 | 76 | 0.765957 | 0.778723 |

## Active Smoke/Inferno Intervals

- `6.0s` - `66.0s`, rows `121`
- `81.0s` - `102.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `49.5`, LSTM `0.1865`, XGBoost `0.3380`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.2127`, XGBoost `0.3367`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.0133`, XGBoost `0.1330`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.0141`, XGBoost `0.1336`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.3152`, XGBoost `0.4337`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.0172`, XGBoost `0.1351`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `61.0`, LSTM `0.0124`, XGBoost `0.1301`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.0140`, XGBoost `0.1301`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.1676`, XGBoost `0.2829`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.0`, LSTM `0.0152`, XGBoost `0.1303`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
