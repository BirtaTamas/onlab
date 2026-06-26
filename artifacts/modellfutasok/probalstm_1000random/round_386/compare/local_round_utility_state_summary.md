# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-furia-vs-saw-bo3-hxORpk_jCtMpGRLo1Voi3p/furia-vs-saw-m2-dust2.csv`
- round_num: `14`
- rows: `118`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 118 | 1.000 | 0.016601 | 0.017579 | -0.000978 | 80 | 38 | 1.000000 | 1.000000 |
| active/recent utility | 118 | 1.000 | 0.016601 | 0.017579 | -0.000978 | 80 | 38 | 1.000000 | 1.000000 |
| strong utility action | 92 | 0.780 | 0.016621 | 0.016322 | 0.000300 | 57 | 35 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 92 | 0.780 | 0.016621 | 0.016322 | 0.000300 | 57 | 35 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 118 | 1.000 | 0.016601 | 0.017579 | -0.000978 | 80 | 38 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.0s` - `32.5s`, rows `48`
- `36.0s` - `57.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `27.0`, LSTM `0.0376`, XGBoost `0.0203`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `19.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.0371`, XGBoost `0.0227`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.0360`, XGBoost `0.0219`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `6.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.0157`, XGBoost `0.0265`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.0327`, XGBoost `0.0219`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.0338`, XGBoost `0.0231`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.0334`, XGBoost `0.0227`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.0333`, XGBoost `0.0227`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.0339`, XGBoost `0.0235`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.0353`, XGBoost `0.0252`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
