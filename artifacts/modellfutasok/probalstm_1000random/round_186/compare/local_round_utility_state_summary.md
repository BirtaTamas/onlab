# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-heroic-bo3-VpF2znQtwzecEgVsCr-4Wn/astralis-vs-heroic-m2-inferno.csv`
- round_num: `7`
- rows: `148`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 148 | 1.000 | 0.674852 | 0.701628 | -0.026776 | 55 | 93 | 0.804054 | 0.682432 |
| active/recent utility | 148 | 1.000 | 0.674852 | 0.701628 | -0.026776 | 55 | 93 | 0.804054 | 0.682432 |
| strong utility action | 134 | 0.905 | 0.670831 | 0.697886 | -0.027055 | 48 | 86 | 0.805970 | 0.679104 |
| utility damage | 42 | 0.284 | 0.703901 | 0.757458 | -0.053556 | 12 | 30 | 0.928571 | 0.880952 |
| active smoke/inferno | 120 | 0.811 | 0.690020 | 0.722332 | -0.032312 | 35 | 85 | 0.816667 | 0.758333 |
| recent utility last 5s | 14 | 0.095 | 0.506353 | 0.488349 | 0.018004 | 13 | 1 | 0.714286 | 0.000000 |
| flash effect present | 148 | 1.000 | 0.674852 | 0.701628 | -0.026776 | 55 | 93 | 0.804054 | 0.682432 |

## Active Smoke/Inferno Intervals

- `10.5s` - `70.0s`, rows `120`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `49.5`, LSTM `0.5604`, XGBoost `0.8031`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `10.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.5679`, XGBoost `0.7905`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `10.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.5800`, XGBoost `0.7983`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `10.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.5992`, XGBoost `0.8031`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `10.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.5860`, XGBoost `0.7866`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `10.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.6038`, XGBoost `0.8011`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `10.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.6009`, XGBoost `0.7889`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `10.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.7971`, XGBoost `0.9800`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `10.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.6141`, XGBoost `0.7853`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `10.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.4552`, XGBoost `0.3048`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
