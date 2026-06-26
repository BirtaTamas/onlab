# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-spirit-vs-vitality-bo3-KtWhzrlsNkWCCS0U9BIlr3/spirit-vs-vitality-m2-nuke.csv`
- round_num: `23`
- rows: `175`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 175 | 1.000 | 0.109121 | 0.086237 | 0.022884 | 120 | 55 | 1.000000 | 1.000000 |
| active/recent utility | 175 | 1.000 | 0.109121 | 0.086237 | 0.022884 | 120 | 55 | 1.000000 | 1.000000 |
| strong utility action | 162 | 0.926 | 0.112206 | 0.085298 | 0.026907 | 108 | 54 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 150 | 0.857 | 0.110347 | 0.082070 | 0.028277 | 104 | 46 | 1.000000 | 1.000000 |
| recent utility last 5s | 12 | 0.069 | 0.135443 | 0.125655 | 0.009788 | 4 | 8 | 1.000000 | 1.000000 |
| flash effect present | 175 | 1.000 | 0.109121 | 0.086237 | 0.022884 | 120 | 55 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.5s` - `33.5s`, rows `47`
- `34.5s` - `85.5s`, rows `103`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `22.0`, LSTM `0.3611`, XGBoost `0.1160`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.3560`, XGBoost `0.1147`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.3534`, XGBoost `0.1141`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.3525`, XGBoost `0.1168`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.3504`, XGBoost `0.1175`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.3395`, XGBoost `0.1109`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.3428`, XGBoost `0.1168`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.3303`, XGBoost `0.1109`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.3230`, XGBoost `0.1101`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.3233`, XGBoost `0.1164`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
