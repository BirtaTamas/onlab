# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-nemiga-bo3-HBPh0RFmxqP1tE9QMaq3nA/heroic-vs-nemiga-m2-mirage.csv`
- round_num: `5`
- rows: `182`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 182 | 1.000 | 0.704668 | 0.744990 | -0.040322 | 31 | 151 | 0.747253 | 0.912088 |
| active/recent utility | 182 | 1.000 | 0.704668 | 0.744990 | -0.040322 | 31 | 151 | 0.747253 | 0.912088 |
| strong utility action | 85 | 0.467 | 0.559654 | 0.594651 | -0.034997 | 31 | 54 | 0.611765 | 0.811765 |
| utility damage | 10 | 0.055 | 0.544415 | 0.524693 | 0.019721 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 85 | 0.467 | 0.559654 | 0.594651 | -0.034997 | 31 | 54 | 0.611765 | 0.811765 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 182 | 1.000 | 0.704668 | 0.744990 | -0.040322 | 31 | 151 | 0.747253 | 0.912088 |

## Active Smoke/Inferno Intervals

- `6.5s` - `48.5s`, rows `85`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `12.5`, LSTM `0.3948`, XGBoost `0.5389`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.4075`, XGBoost `0.5377`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.4080`, XGBoost `0.5343`, closer `xgboost`, smoke `1`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.1780`, XGBoost `0.3029`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.3989`, XGBoost `0.5235`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.1788`, XGBoost `0.3022`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.4110`, XGBoost `0.5343`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.4239`, XGBoost `0.5410`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.1897`, XGBoost `0.3022`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.4254`, XGBoost `0.5343`, closer `xgboost`, smoke `1`, inferno `3`, utility_damage `0.0`, recent_utility `0`
