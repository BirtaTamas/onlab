# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-betboom-vs-legacy-anubis-nLMamLTYoRhlv2MuS6sSiC/betboom-vs-legacy-anubis.csv`
- round_num: `5`
- rows: `282`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 282 | 1.000 | 0.323132 | 0.401599 | -0.078468 | 276 | 6 | 0.719858 | 0.446809 |
| active/recent utility | 282 | 1.000 | 0.323132 | 0.401599 | -0.078468 | 276 | 6 | 0.719858 | 0.446809 |
| strong utility action | 196 | 0.695 | 0.403105 | 0.496778 | -0.093673 | 190 | 6 | 0.607143 | 0.326531 |
| utility damage | 20 | 0.071 | 0.559003 | 0.741877 | -0.182874 | 20 | 0 | 0.000000 | 0.000000 |
| active smoke/inferno | 187 | 0.663 | 0.401863 | 0.495857 | -0.093994 | 181 | 6 | 0.588235 | 0.331551 |
| recent utility last 5s | 10 | 0.035 | 0.430855 | 0.514199 | -0.083344 | 10 | 0 | 1.000000 | 0.300000 |
| flash effect present | 282 | 1.000 | 0.323132 | 0.401599 | -0.078468 | 276 | 6 | 0.719858 | 0.446809 |

## Active Smoke/Inferno Intervals

- `9.5s` - `80.5s`, rows `143`
- `90.0s` - `111.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `10.5`, LSTM `0.4982`, XGBoost `0.7780`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.5212`, XGBoost `0.7824`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.4476`, XGBoost `0.7003`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `78.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.5237`, XGBoost `0.7740`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.4496`, XGBoost `0.6984`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `78.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.5403`, XGBoost `0.7884`, closer `lstm`, smoke `2`, inferno `6`, utility_damage `20.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.5413`, XGBoost `0.7880`, closer `lstm`, smoke `2`, inferno `6`, utility_damage `20.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.5478`, XGBoost `0.7884`, closer `lstm`, smoke `2`, inferno `6`, utility_damage `20.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.5484`, XGBoost `0.7880`, closer `lstm`, smoke `2`, inferno `5`, utility_damage `20.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.4688`, XGBoost `0.6984`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `78.0`, recent_utility `0`
