# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-m80-vs-rooster-bo3-GFAv4Fg83aXYKbsY0nLkP_/m80-vs-rooster-m2-inferno.csv`
- round_num: `16`
- rows: `212`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 212 | 1.000 | 0.495537 | 0.482276 | 0.013261 | 76 | 136 | 0.466981 | 0.580189 |
| active/recent utility | 212 | 1.000 | 0.495537 | 0.482276 | 0.013261 | 76 | 136 | 0.466981 | 0.580189 |
| strong utility action | 184 | 0.868 | 0.480523 | 0.470624 | 0.009899 | 76 | 108 | 0.494565 | 0.559783 |
| utility damage | 21 | 0.099 | 0.667095 | 0.584308 | 0.082788 | 0 | 21 | 0.095238 | 0.238095 |
| active smoke/inferno | 183 | 0.863 | 0.479839 | 0.470484 | 0.009355 | 76 | 107 | 0.497268 | 0.557377 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 212 | 1.000 | 0.495537 | 0.482276 | 0.013261 | 76 | 136 | 0.466981 | 0.580189 |

## Active Smoke/Inferno Intervals

- `10.5s` - `59.5s`, rows `99`
- `64.0s` - `105.5s`, rows `84`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `76.5`, LSTM `0.3255`, XGBoost `0.4476`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.3526`, XGBoost `0.4701`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.6228`, XGBoost `0.5111`, closer `xgboost`, smoke `1`, inferno `3`, utility_damage `14.0`, recent_utility `0`
- seconds `75.5`, LSTM `0.3351`, XGBoost `0.4459`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.6183`, XGBoost `0.5088`, closer `xgboost`, smoke `1`, inferno `3`, utility_damage `14.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.6056`, XGBoost `0.4961`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `14.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.6081`, XGBoost `0.4991`, closer `xgboost`, smoke `0`, inferno `2`, utility_damage `14.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.8334`, XGBoost `0.7256`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `76.0`, LSTM `0.3389`, XGBoost `0.4459`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `75.0`, LSTM `0.3306`, XGBoost `0.4367`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
