# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-natus-vincere-vs-spirit-bo3-cW0x-KCT4cbPLaZUAvb08Z/natus-vincere-vs-spirit-m2-ancient.csv`
- round_num: `13`
- rows: `206`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 206 | 1.000 | 0.752345 | 0.772068 | -0.019723 | 21 | 185 | 0.825243 | 0.941748 |
| active/recent utility | 206 | 1.000 | 0.752345 | 0.772068 | -0.019723 | 21 | 185 | 0.825243 | 0.941748 |
| strong utility action | 82 | 0.398 | 0.877149 | 0.904918 | -0.027769 | 7 | 75 | 0.853659 | 0.853659 |
| utility damage | 10 | 0.049 | 0.860108 | 0.951884 | -0.091776 | 0 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 82 | 0.398 | 0.877149 | 0.904918 | -0.027769 | 7 | 75 | 0.853659 | 0.853659 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 206 | 1.000 | 0.752345 | 0.772068 | -0.019723 | 21 | 185 | 0.825243 | 0.941748 |

## Active Smoke/Inferno Intervals

- `40.0s` - `80.5s`, rows `82`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `45.5`, LSTM `0.2998`, XGBoost `0.4732`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.5275`, XGBoost `0.6984`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.5262`, XGBoost `0.6957`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.5603`, XGBoost `0.7257`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `6.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.7854`, XGBoost `0.9399`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `6.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.7884`, XGBoost `0.9375`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `6.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.8815`, XGBoost `0.9843`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `6.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.8867`, XGBoost `0.9843`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `6.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.8975`, XGBoost `0.9845`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `6.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.3833`, XGBoost `0.4666`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
