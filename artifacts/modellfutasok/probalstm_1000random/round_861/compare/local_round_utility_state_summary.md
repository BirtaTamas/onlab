# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-vitality-vs-hotu-bo3-g2oB3RySVGugmKq6yJcHo9/vitality-vs-hotu-m2-dust2.csv`
- round_num: `15`
- rows: `138`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 138 | 1.000 | 0.424543 | 0.443259 | -0.018716 | 74 | 64 | 0.579710 | 0.130435 |
| active/recent utility | 138 | 1.000 | 0.424543 | 0.443259 | -0.018716 | 74 | 64 | 0.579710 | 0.130435 |
| strong utility action | 124 | 0.899 | 0.406494 | 0.439383 | -0.032889 | 60 | 64 | 0.532258 | 0.145161 |
| utility damage | 10 | 0.072 | 0.375894 | 0.355897 | 0.019997 | 6 | 4 | 0.500000 | 0.000000 |
| active smoke/inferno | 121 | 0.877 | 0.403743 | 0.438392 | -0.034649 | 57 | 64 | 0.520661 | 0.148760 |
| recent utility last 5s | 10 | 0.072 | 0.515643 | 0.489252 | 0.026391 | 10 | 0 | 1.000000 | 0.200000 |
| flash effect present | 138 | 1.000 | 0.424543 | 0.443259 | -0.018716 | 74 | 64 | 0.579710 | 0.130435 |

## Active Smoke/Inferno Intervals

- `8.5s` - `68.5s`, rows `121`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `64.5`, LSTM `0.3449`, XGBoost `0.7018`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `65.0`, LSTM `0.3771`, XGBoost `0.6733`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.0`, LSTM `0.4047`, XGBoost `0.6985`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `65.5`, LSTM `0.6852`, XGBoost `0.9343`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.4577`, XGBoost `0.6985`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.4861`, XGBoost `0.6960`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.4923`, XGBoost `0.6985`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `66.0`, LSTM `0.7319`, XGBoost `0.9348`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.0`, LSTM `0.5117`, XGBoost `0.6985`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `66.5`, LSTM `0.7529`, XGBoost `0.9348`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
