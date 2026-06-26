# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2_finals/blast-bounty-2025-season-2-finals-the-mongolz-vs-aurora-bo3-BL3Rcvf733R8cy7TtLY5HE/the-mongolz-vs-aurora-m1-dust2.csv`
- round_num: `22`
- rows: `255`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 255 | 1.000 | 0.337605 | 0.305763 | 0.031842 | 115 | 140 | 0.458824 | 0.458824 |
| active/recent utility | 255 | 1.000 | 0.337605 | 0.305763 | 0.031842 | 115 | 140 | 0.458824 | 0.458824 |
| strong utility action | 155 | 0.608 | 0.487045 | 0.441246 | 0.045800 | 32 | 123 | 0.219355 | 0.219355 |
| utility damage | 25 | 0.098 | 0.434692 | 0.406726 | 0.027966 | 8 | 17 | 0.360000 | 0.360000 |
| active smoke/inferno | 153 | 0.600 | 0.485777 | 0.439881 | 0.045896 | 32 | 121 | 0.222222 | 0.222222 |
| recent utility last 5s | 10 | 0.039 | 0.005920 | 0.028420 | -0.022500 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 255 | 1.000 | 0.337605 | 0.305763 | 0.031842 | 115 | 140 | 0.458824 | 0.458824 |

## Active Smoke/Inferno Intervals

- `9.5s` - `85.5s`, rows `153`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `74.0`, LSTM `0.2042`, XGBoost `0.0925`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `2.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.6418`, XGBoost `0.5387`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.6378`, XGBoost `0.5399`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.6358`, XGBoost `0.5385`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.6312`, XGBoost `0.5348`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.6317`, XGBoost `0.5359`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.6304`, XGBoost `0.5348`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.6300`, XGBoost `0.5348`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.6299`, XGBoost `0.5348`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.6346`, XGBoost `0.5399`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
