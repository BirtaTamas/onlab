# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-legacy-vs-nrg-bo3-_uO_eo-VIGwp_pYoaUs9Le/legacy-vs-nrg-m3-dust2.csv`
- round_num: `11`
- rows: `122`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 122 | 1.000 | 0.453326 | 0.434787 | 0.018538 | 57 | 65 | 0.516393 | 0.557377 |
| active/recent utility | 122 | 1.000 | 0.453326 | 0.434787 | 0.018538 | 57 | 65 | 0.516393 | 0.557377 |
| strong utility action | 97 | 0.795 | 0.482714 | 0.455161 | 0.027553 | 42 | 55 | 0.463918 | 0.515464 |
| utility damage | 20 | 0.164 | 0.471618 | 0.476258 | -0.004640 | 11 | 9 | 0.600000 | 0.650000 |
| active smoke/inferno | 97 | 0.795 | 0.482714 | 0.455161 | 0.027553 | 42 | 55 | 0.463918 | 0.515464 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 122 | 1.000 | 0.453326 | 0.434787 | 0.018538 | 57 | 65 | 0.516393 | 0.557377 |

## Active Smoke/Inferno Intervals

- `3.5s` - `51.5s`, rows `97`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `34.5`, LSTM `0.4216`, XGBoost `0.1964`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.4231`, XGBoost `0.2099`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.4088`, XGBoost `0.2029`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.5305`, XGBoost `0.3251`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.5290`, XGBoost `0.3251`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.4178`, XGBoost `0.2158`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.4211`, XGBoost `0.2211`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.5244`, XGBoost `0.3251`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.4261`, XGBoost `0.2273`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.5485`, XGBoost `0.3500`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
