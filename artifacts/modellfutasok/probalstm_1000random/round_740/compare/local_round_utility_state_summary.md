# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-housebets-bo3-NgyLHfqCvYO4WZnaqhUlfi/heroic-vs-housebets-m1-dust2.csv`
- round_num: `6`
- rows: `258`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 258 | 1.000 | 0.865480 | 0.885987 | -0.020507 | 77 | 181 | 1.000000 | 1.000000 |
| active/recent utility | 258 | 1.000 | 0.865480 | 0.885987 | -0.020507 | 77 | 181 | 1.000000 | 1.000000 |
| strong utility action | 199 | 0.771 | 0.840659 | 0.869003 | -0.028345 | 37 | 162 | 1.000000 | 1.000000 |
| utility damage | 12 | 0.047 | 0.774746 | 0.770001 | 0.004745 | 8 | 4 | 1.000000 | 1.000000 |
| active smoke/inferno | 192 | 0.744 | 0.844378 | 0.874674 | -0.030296 | 30 | 162 | 1.000000 | 1.000000 |
| recent utility last 5s | 13 | 0.050 | 0.726220 | 0.709190 | 0.017030 | 11 | 2 | 1.000000 | 1.000000 |
| flash effect present | 258 | 1.000 | 0.865480 | 0.885987 | -0.020507 | 77 | 181 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `4.5s` - `76.0s`, rows `144`
- `95.5s` - `119.0s`, rows `48`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `110.0`, LSTM `0.5677`, XGBoost `0.8239`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `109.5`, LSTM `0.5744`, XGBoost `0.8279`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `110.5`, LSTM `0.5891`, XGBoost `0.8273`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `109.0`, LSTM `0.5896`, XGBoost `0.8134`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `111.0`, LSTM `0.6102`, XGBoost `0.8273`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `108.5`, LSTM `0.5846`, XGBoost `0.8006`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `112.0`, LSTM `0.6241`, XGBoost `0.8248`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `107.5`, LSTM `0.6366`, XGBoost `0.8300`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `111.5`, LSTM `0.6338`, XGBoost `0.8206`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `112.5`, LSTM `0.6290`, XGBoost `0.8143`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
