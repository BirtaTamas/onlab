# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-nemiga-bo3-HBPh0RFmxqP1tE9QMaq3nA/heroic-vs-nemiga-m2-mirage.csv`
- round_num: `14`
- rows: `164`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 164 | 1.000 | 0.469088 | 0.499491 | -0.030403 | 51 | 113 | 0.432927 | 0.432927 |
| active/recent utility | 164 | 1.000 | 0.469088 | 0.499491 | -0.030403 | 51 | 113 | 0.432927 | 0.432927 |
| strong utility action | 124 | 0.756 | 0.442373 | 0.471467 | -0.029093 | 36 | 88 | 0.395161 | 0.395161 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 124 | 0.756 | 0.442373 | 0.471467 | -0.029093 | 36 | 88 | 0.395161 | 0.395161 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 164 | 1.000 | 0.469088 | 0.499491 | -0.030403 | 51 | 113 | 0.432927 | 0.432927 |

## Active Smoke/Inferno Intervals

- `9.0s` - `70.5s`, rows `124`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `15.5`, LSTM `0.0992`, XGBoost `0.2719`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.1262`, XGBoost `0.2725`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.1269`, XGBoost `0.2719`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.1333`, XGBoost `0.2704`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.1388`, XGBoost `0.2721`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.1383`, XGBoost `0.2703`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.1403`, XGBoost `0.2705`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.1405`, XGBoost `0.2703`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.1483`, XGBoost `0.2719`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.1500`, XGBoost `0.2721`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
