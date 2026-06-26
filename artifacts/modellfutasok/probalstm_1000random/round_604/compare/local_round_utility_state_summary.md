# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-gamerlegion-vs-pain-bo3-zcuZjSa9VUSMkJoK5k8I3c/gamerlegion-vs-pain-m3-mirage.csv`
- round_num: `3`
- rows: `271`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 271 | 1.000 | 0.334788 | 0.542815 | -0.208027 | 266 | 5 | 0.867159 | 0.309963 |
| active/recent utility | 271 | 1.000 | 0.334788 | 0.542815 | -0.208027 | 266 | 5 | 0.867159 | 0.309963 |
| strong utility action | 76 | 0.280 | 0.479565 | 0.612350 | -0.132785 | 76 | 0 | 0.657895 | 0.105263 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 76 | 0.280 | 0.479565 | 0.612350 | -0.132785 | 76 | 0 | 0.657895 | 0.105263 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 271 | 1.000 | 0.334788 | 0.542815 | -0.208027 | 266 | 5 | 0.867159 | 0.309963 |

## Active Smoke/Inferno Intervals

- `6.0s` - `43.5s`, rows `76`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `24.5`, LSTM `0.2837`, XGBoost `0.5727`, closer `lstm`, smoke `7`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.4530`, XGBoost `0.7199`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.2332`, XGBoost `0.4981`, closer `lstm`, smoke `7`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.4584`, XGBoost `0.7199`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.4605`, XGBoost `0.7199`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.4338`, XGBoost `0.6900`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.4415`, XGBoost `0.6900`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.4737`, XGBoost `0.7199`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.4774`, XGBoost `0.7199`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.2574`, XGBoost `0.4955`, closer `lstm`, smoke `7`, inferno `0`, utility_damage `1.0`, recent_utility `0`
