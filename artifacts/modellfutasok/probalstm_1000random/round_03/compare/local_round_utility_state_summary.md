# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-flyquest-vs-mibr-bo3-qPrK-wzQgATa8KQ5HjYeOS/flyquest-vs-mibr-m1-nuke.csv`
- round_num: `12`
- rows: `190`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 190 | 1.000 | 0.691920 | 0.734196 | -0.042276 | 21 | 169 | 1.000000 | 1.000000 |
| active/recent utility | 190 | 1.000 | 0.691920 | 0.734196 | -0.042276 | 21 | 169 | 1.000000 | 1.000000 |
| strong utility action | 128 | 0.674 | 0.683917 | 0.736934 | -0.053017 | 6 | 122 | 1.000000 | 1.000000 |
| utility damage | 23 | 0.121 | 0.627877 | 0.664676 | -0.036800 | 5 | 18 | 1.000000 | 1.000000 |
| active smoke/inferno | 128 | 0.674 | 0.683917 | 0.736934 | -0.053017 | 6 | 122 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 190 | 1.000 | 0.691920 | 0.734196 | -0.042276 | 21 | 169 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.0s` - `39.5s`, rows `62`
- `55.0s` - `87.5s`, rows `66`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `75.5`, LSTM `0.7306`, XGBoost `0.8392`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `75.0`, LSTM `0.7315`, XGBoost `0.8391`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.7396`, XGBoost `0.8422`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `74.0`, LSTM `0.7424`, XGBoost `0.8443`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `74.5`, LSTM `0.7279`, XGBoost `0.8288`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `76.0`, LSTM `0.8496`, XGBoost `0.9495`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.5`, LSTM `0.7474`, XGBoost `0.8441`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `73.0`, LSTM `0.7489`, XGBoost `0.8441`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.5897`, XGBoost `0.6820`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.5930`, XGBoost `0.6820`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `1.0`, recent_utility `0`
