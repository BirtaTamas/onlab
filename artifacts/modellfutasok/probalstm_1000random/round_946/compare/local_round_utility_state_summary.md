# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-vitality-bo5-RwgqrXEuhDJTxQHhSIn72X/mouz-vs-vitality-m2-nuke.csv`
- round_num: `2`
- rows: `176`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 176 | 1.000 | 0.923584 | 0.979879 | -0.056296 | 0 | 176 | 1.000000 | 1.000000 |
| active/recent utility | 176 | 1.000 | 0.923584 | 0.979879 | -0.056296 | 0 | 176 | 1.000000 | 1.000000 |
| strong utility action | 44 | 0.250 | 0.925019 | 0.981023 | -0.056004 | 0 | 44 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 44 | 0.250 | 0.925019 | 0.981023 | -0.056004 | 0 | 44 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 176 | 1.000 | 0.923584 | 0.979879 | -0.056296 | 0 | 176 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.0s` - `30.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `25.5`, LSTM `0.9100`, XGBoost `0.9804`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.9123`, XGBoost `0.9801`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.9164`, XGBoost `0.9812`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.9169`, XGBoost `0.9812`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.9175`, XGBoost `0.9811`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.9177`, XGBoost `0.9812`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.9177`, XGBoost `0.9803`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.9179`, XGBoost `0.9804`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.9190`, XGBoost `0.9812`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.9198`, XGBoost `0.9812`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
