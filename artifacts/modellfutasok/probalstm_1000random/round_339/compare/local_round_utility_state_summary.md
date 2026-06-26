# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-heroic-vs-3dmax-bo3-OVT4ch_FfOW2E26liKqT_k/heroic-vs-3dmax-m2-inferno.csv`
- round_num: `15`
- rows: `153`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 153 | 1.000 | 0.207085 | 0.295935 | -0.088850 | 143 | 10 | 0.993464 | 0.973856 |
| active/recent utility | 153 | 1.000 | 0.207085 | 0.295935 | -0.088850 | 143 | 10 | 0.993464 | 0.973856 |
| strong utility action | 115 | 0.752 | 0.214373 | 0.298565 | -0.084192 | 105 | 10 | 0.991304 | 0.965217 |
| utility damage | 13 | 0.085 | 0.324472 | 0.430951 | -0.106479 | 13 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 115 | 0.752 | 0.214373 | 0.298565 | -0.084192 | 105 | 10 | 0.991304 | 0.965217 |
| recent utility last 5s | 10 | 0.065 | 0.099945 | 0.250560 | -0.150615 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 153 | 1.000 | 0.207085 | 0.295935 | -0.088850 | 143 | 10 | 0.993464 | 0.973856 |

## Active Smoke/Inferno Intervals

- `10.0s` - `18.0s`, rows `17`
- `22.0s` - `70.5s`, rows `98`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `48.5`, LSTM `0.0615`, XGBoost `0.2567`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.2371`, XGBoost `0.4261`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.2345`, XGBoost `0.4212`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.0699`, XGBoost `0.2545`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `35.0`, LSTM `0.2420`, XGBoost `0.4261`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.0742`, XGBoost `0.2545`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `45.0`, LSTM `0.0731`, XGBoost `0.2532`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `44.5`, LSTM `0.0771`, XGBoost `0.2532`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `49.0`, LSTM `0.0840`, XGBoost `0.2581`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.0816`, XGBoost `0.2545`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `1`
