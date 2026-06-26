# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-heroic-bo3-VpF2znQtwzecEgVsCr-4Wn/astralis-vs-heroic-m2-inferno.csv`
- round_num: `10`
- rows: `257`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 257 | 1.000 | 0.685938 | 0.692068 | -0.006130 | 118 | 139 | 0.988327 | 0.910506 |
| active/recent utility | 257 | 1.000 | 0.685938 | 0.692068 | -0.006130 | 118 | 139 | 0.988327 | 0.910506 |
| strong utility action | 217 | 0.844 | 0.685484 | 0.694690 | -0.009206 | 96 | 121 | 0.986175 | 0.990783 |
| utility damage | 26 | 0.101 | 0.578959 | 0.548932 | 0.030027 | 17 | 9 | 1.000000 | 1.000000 |
| active smoke/inferno | 217 | 0.844 | 0.685484 | 0.694690 | -0.009206 | 96 | 121 | 0.986175 | 0.990783 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 257 | 1.000 | 0.685938 | 0.692068 | -0.006130 | 118 | 139 | 0.988327 | 0.910506 |

## Active Smoke/Inferno Intervals

- `10.5s` - `118.5s`, rows `217`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `91.0`, LSTM `0.6779`, XGBoost `0.8606`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `91.5`, LSTM `0.6876`, XGBoost `0.8556`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `92.5`, LSTM `0.6777`, XGBoost `0.8405`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `92.0`, LSTM `0.6904`, XGBoost `0.8505`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `94.0`, LSTM `0.6931`, XGBoost `0.8478`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `90.5`, LSTM `0.7224`, XGBoost `0.8658`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `93.5`, LSTM `0.7097`, XGBoost `0.8467`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `88.5`, LSTM `0.7362`, XGBoost `0.8621`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `89.5`, LSTM `0.7358`, XGBoost `0.8616`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `107.5`, LSTM `0.6442`, XGBoost `0.7662`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
