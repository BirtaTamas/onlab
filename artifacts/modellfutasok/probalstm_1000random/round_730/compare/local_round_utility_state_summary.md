# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-spirit-vs-vitality-bo3-KtWhzrlsNkWCCS0U9BIlr3/spirit-vs-vitality-m2-nuke.csv`
- round_num: `13`
- rows: `153`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 153 | 1.000 | 0.637268 | 0.746970 | -0.109701 | 140 | 13 | 0.032680 | 0.039216 |
| active/recent utility | 153 | 1.000 | 0.637268 | 0.746970 | -0.109701 | 140 | 13 | 0.032680 | 0.039216 |
| strong utility action | 60 | 0.392 | 0.619589 | 0.808459 | -0.188870 | 59 | 1 | 0.000000 | 0.016667 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 60 | 0.392 | 0.619589 | 0.808459 | -0.188870 | 59 | 1 | 0.000000 | 0.016667 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 153 | 1.000 | 0.637268 | 0.746970 | -0.109701 | 140 | 13 | 0.032680 | 0.039216 |

## Active Smoke/Inferno Intervals

- `11.0s` - `40.5s`, rows `60`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `30.5`, LSTM `0.5669`, XGBoost `0.8534`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.5760`, XGBoost `0.8521`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.5997`, XGBoost `0.8646`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `2.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.5849`, XGBoost `0.8361`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.5859`, XGBoost `0.8345`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `19.0`, LSTM `0.5877`, XGBoost `0.8347`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.5922`, XGBoost `0.8373`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.5901`, XGBoost `0.8343`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.6205`, XGBoost `0.8646`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.5913`, XGBoost `0.8341`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
