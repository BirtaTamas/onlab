# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-g2-vs-fluxo-bo3-IhqycqXYyOA3DyfY0xuGyX/g2-vs-fluxo-m2-inferno.csv`
- round_num: `3`
- rows: `125`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 125 | 1.000 | 0.131147 | 0.187331 | -0.056184 | 115 | 10 | 1.000000 | 1.000000 |
| active/recent utility | 125 | 1.000 | 0.131147 | 0.187331 | -0.056184 | 115 | 10 | 1.000000 | 1.000000 |
| strong utility action | 116 | 0.928 | 0.129622 | 0.183177 | -0.053555 | 106 | 10 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.080 | 0.123773 | 0.145652 | -0.021878 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 109 | 0.872 | 0.132633 | 0.184908 | -0.052275 | 99 | 10 | 1.000000 | 1.000000 |
| recent utility last 5s | 23 | 0.184 | 0.116464 | 0.158341 | -0.041877 | 20 | 3 | 1.000000 | 1.000000 |
| flash effect present | 125 | 1.000 | 0.131147 | 0.187331 | -0.056184 | 115 | 10 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `60.5s`, rows `109`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `47.0`, LSTM `0.1665`, XGBoost `0.3787`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.1946`, XGBoost `0.3801`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.5`, LSTM `0.2391`, XGBoost `0.4079`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.0`, LSTM `0.2452`, XGBoost `0.4079`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.2178`, XGBoost `0.3794`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.0`, LSTM `0.2473`, XGBoost `0.4079`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.2359`, XGBoost `0.3894`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.2555`, XGBoost `0.4079`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `57.0`, LSTM `0.2569`, XGBoost `0.4079`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.2595`, XGBoost `0.4079`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
