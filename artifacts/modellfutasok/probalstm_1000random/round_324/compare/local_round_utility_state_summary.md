# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-betboom-vs-legacy-anubis-nLMamLTYoRhlv2MuS6sSiC/betboom-vs-legacy-anubis.csv`
- round_num: `9`
- rows: `261`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 261 | 1.000 | 0.382184 | 0.524086 | -0.141901 | 261 | 0 | 0.977011 | 0.272031 |
| active/recent utility | 261 | 1.000 | 0.382184 | 0.524086 | -0.141901 | 261 | 0 | 0.977011 | 0.272031 |
| strong utility action | 209 | 0.801 | 0.398017 | 0.532672 | -0.134655 | 209 | 0 | 0.976077 | 0.239234 |
| utility damage | 17 | 0.065 | 0.439451 | 0.496614 | -0.057163 | 17 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 206 | 0.789 | 0.398046 | 0.532621 | -0.134575 | 206 | 0 | 0.975728 | 0.242718 |
| recent utility last 5s | 10 | 0.038 | 0.405327 | 0.536339 | -0.131012 | 10 | 0 | 1.000000 | 0.000000 |
| flash effect present | 261 | 1.000 | 0.382184 | 0.524086 | -0.141901 | 261 | 0 | 0.977011 | 0.272031 |

## Active Smoke/Inferno Intervals

- `7.5s` - `55.0s`, rows `96`
- `59.0s` - `113.5s`, rows `110`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `99.0`, LSTM `0.2124`, XGBoost `0.6283`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `98.5`, LSTM `0.2252`, XGBoost `0.6043`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `98.0`, LSTM `0.2365`, XGBoost `0.5990`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `92.5`, LSTM `0.3637`, XGBoost `0.7005`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `99.5`, LSTM `0.3092`, XGBoost `0.6288`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `97.5`, LSTM `0.2805`, XGBoost `0.5946`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `96.5`, LSTM `0.3658`, XGBoost `0.6797`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `100.0`, LSTM `0.3370`, XGBoost `0.6448`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `92.0`, LSTM `0.4157`, XGBoost `0.7013`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `103.5`, LSTM `0.2599`, XGBoost `0.5454`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
