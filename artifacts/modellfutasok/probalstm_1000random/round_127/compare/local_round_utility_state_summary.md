# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-flyquest-vs-legacy-bo3-FlEa8e0vdBrf1ft_mNbThh/flyquest-vs-legacy-m2-nuke.csv`
- round_num: `8`
- rows: `214`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 214 | 1.000 | 0.086900 | 0.156880 | -0.069981 | 204 | 10 | 1.000000 | 0.995327 |
| active/recent utility | 214 | 1.000 | 0.086900 | 0.156880 | -0.069981 | 204 | 10 | 1.000000 | 0.995327 |
| strong utility action | 180 | 0.841 | 0.087553 | 0.146944 | -0.059391 | 170 | 10 | 1.000000 | 0.994444 |
| utility damage | 10 | 0.047 | 0.208027 | 0.208665 | -0.000638 | 7 | 3 | 1.000000 | 1.000000 |
| active smoke/inferno | 180 | 0.841 | 0.087553 | 0.146944 | -0.059391 | 170 | 10 | 1.000000 | 0.994444 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 214 | 1.000 | 0.086900 | 0.156880 | -0.069981 | 204 | 10 | 1.000000 | 0.995327 |

## Active Smoke/Inferno Intervals

- `8.5s` - `46.0s`, rows `76`
- `55.0s` - `106.5s`, rows `104`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `9.5`, LSTM `0.1167`, XGBoost `0.3692`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.1221`, XGBoost `0.3668`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.1216`, XGBoost `0.3616`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.1280`, XGBoost `0.3671`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.1272`, XGBoost `0.3565`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.1570`, XGBoost `0.3668`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `93.0`, LSTM `0.0628`, XGBoost `0.2491`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `92.5`, LSTM `0.0628`, XGBoost `0.2461`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `92.0`, LSTM `0.0752`, XGBoost `0.2513`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `106.0`, LSTM `0.0698`, XGBoost `0.2420`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
