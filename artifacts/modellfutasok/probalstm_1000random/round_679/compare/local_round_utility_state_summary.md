# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-liquid-vs-furia-bo3-oYHD2J45okzf6eapD2F9CM/liquid-vs-furia-m1-mirage.csv`
- round_num: `13`
- rows: `136`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 136 | 1.000 | 0.259468 | 0.417137 | -0.157669 | 133 | 3 | 0.808824 | 0.669118 |
| active/recent utility | 136 | 1.000 | 0.259468 | 0.417137 | -0.157669 | 133 | 3 | 0.808824 | 0.669118 |
| strong utility action | 58 | 0.426 | 0.339754 | 0.470802 | -0.131048 | 57 | 1 | 0.844828 | 0.724138 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 58 | 0.426 | 0.339754 | 0.470802 | -0.131048 | 57 | 1 | 0.844828 | 0.724138 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 136 | 1.000 | 0.259468 | 0.417137 | -0.157669 | 133 | 3 | 0.808824 | 0.669118 |

## Active Smoke/Inferno Intervals

- `8.5s` - `31.5s`, rows `47`
- `54.5s` - `59.5s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `31.5`, LSTM `0.1851`, XGBoost `0.4511`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.2310`, XGBoost `0.4955`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.2435`, XGBoost `0.4955`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.2033`, XGBoost `0.4518`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `57.5`, LSTM `0.0597`, XGBoost `0.2853`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `57.0`, LSTM `0.0633`, XGBoost `0.2884`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `56.5`, LSTM `0.0674`, XGBoost `0.2884`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `59.0`, LSTM `0.0671`, XGBoost `0.2822`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.0759`, XGBoost `0.2884`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.0775`, XGBoost `0.2884`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
