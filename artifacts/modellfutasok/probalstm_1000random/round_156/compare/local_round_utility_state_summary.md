# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-spirit-vs-vitality-bo3-KtWhzrlsNkWCCS0U9BIlr3/spirit-vs-vitality-m2-nuke.csv`
- round_num: `10`
- rows: `258`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 258 | 1.000 | 0.284268 | 0.342675 | -0.058407 | 235 | 23 | 0.655039 | 0.465116 |
| active/recent utility | 258 | 1.000 | 0.284268 | 0.342675 | -0.058407 | 235 | 23 | 0.655039 | 0.465116 |
| strong utility action | 174 | 0.674 | 0.361670 | 0.435405 | -0.073735 | 154 | 20 | 0.517241 | 0.321839 |
| utility damage | 10 | 0.039 | 0.516020 | 0.552948 | -0.036928 | 10 | 0 | 0.000000 | 0.000000 |
| active smoke/inferno | 174 | 0.674 | 0.361670 | 0.435405 | -0.073735 | 154 | 20 | 0.517241 | 0.321839 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 258 | 1.000 | 0.284268 | 0.342675 | -0.058407 | 235 | 23 | 0.655039 | 0.465116 |

## Active Smoke/Inferno Intervals

- `9.0s` - `64.5s`, rows `112`
- `66.5s` - `97.0s`, rows `62`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `78.0`, LSTM `0.0154`, XGBoost `0.2711`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `73.5`, LSTM `0.0700`, XGBoost `0.3200`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.0`, LSTM `0.0896`, XGBoost `0.3321`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `74.5`, LSTM `0.0296`, XGBoost `0.2709`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `75.0`, LSTM `0.0330`, XGBoost `0.2708`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `75.5`, LSTM `0.0333`, XGBoost `0.2708`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `74.0`, LSTM `0.0417`, XGBoost `0.2788`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.0982`, XGBoost `0.3325`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `82.5`, LSTM `0.0225`, XGBoost `0.2529`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `83.0`, LSTM `0.0273`, XGBoost `0.2566`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
