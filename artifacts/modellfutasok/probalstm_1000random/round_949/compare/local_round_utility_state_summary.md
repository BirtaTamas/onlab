# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-heroic-bo3-VpF2znQtwzecEgVsCr-4Wn/astralis-vs-heroic-m2-inferno.csv`
- round_num: `14`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.855388 | 0.917478 | -0.062090 | 16 | 214 | 1.000000 | 1.000000 |
| active/recent utility | 230 | 1.000 | 0.855388 | 0.917478 | -0.062090 | 16 | 214 | 1.000000 | 1.000000 |
| strong utility action | 139 | 0.604 | 0.874252 | 0.927821 | -0.053569 | 11 | 128 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 139 | 0.604 | 0.874252 | 0.927821 | -0.053569 | 11 | 128 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 230 | 1.000 | 0.855388 | 0.917478 | -0.062090 | 16 | 214 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.5s` - `38.5s`, rows `57`
- `55.5s` - `96.0s`, rows `82`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `82.5`, LSTM `0.6085`, XGBoost `0.8087`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `83.0`, LSTM `0.6311`, XGBoost `0.8067`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `84.0`, LSTM `0.6341`, XGBoost `0.8063`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `82.0`, LSTM `0.6392`, XGBoost `0.8090`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `83.5`, LSTM `0.6400`, XGBoost `0.8039`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `84.5`, LSTM `0.6414`, XGBoost `0.8052`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `85.0`, LSTM `0.6469`, XGBoost `0.8052`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `85.5`, LSTM `0.6798`, XGBoost `0.8052`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `81.5`, LSTM `0.6930`, XGBoost `0.8098`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `86.0`, LSTM `0.7757`, XGBoost `0.8920`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
