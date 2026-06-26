# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-liquid-bo3-9v1WXdmzbeO2q7iD5Nu_mP/mouz-vs-liquid-m2-nuke.csv`
- round_num: `13`
- rows: `192`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 192 | 1.000 | 0.372958 | 0.409963 | -0.037005 | 35 | 157 | 0.520833 | 0.531250 |
| active/recent utility | 192 | 1.000 | 0.372958 | 0.409963 | -0.037005 | 35 | 157 | 0.520833 | 0.531250 |
| strong utility action | 58 | 0.302 | 0.540224 | 0.585409 | -0.045185 | 2 | 56 | 1.000000 | 0.965517 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 58 | 0.302 | 0.540224 | 0.585409 | -0.045185 | 2 | 56 | 1.000000 | 0.965517 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 192 | 1.000 | 0.372958 | 0.409963 | -0.037005 | 35 | 157 | 0.520833 | 0.531250 |

## Active Smoke/Inferno Intervals

- `6.5s` - `35.0s`, rows `58`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `25.5`, LSTM `0.5837`, XGBoost `0.7600`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.5828`, XGBoost `0.7566`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.5907`, XGBoost `0.7600`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.5957`, XGBoost `0.7612`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.5986`, XGBoost `0.7566`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.6055`, XGBoost `0.7609`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.6093`, XGBoost `0.7609`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.6106`, XGBoost `0.7602`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.5840`, XGBoost `0.7289`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `19.0`, LSTM `0.5852`, XGBoost `0.7289`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
