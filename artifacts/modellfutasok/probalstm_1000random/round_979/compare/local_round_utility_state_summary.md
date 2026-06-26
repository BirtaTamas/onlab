# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-virtuspro-vs-legacy-ancient-7ivruObh5LTTVaCYe9h-YO/virtus-pro-vs-legacy-ancient.csv`
- round_num: `21`
- rows: `254`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 254 | 1.000 | 0.325586 | 0.312932 | 0.012654 | 122 | 132 | 0.421260 | 0.421260 |
| active/recent utility | 254 | 1.000 | 0.325586 | 0.312932 | 0.012654 | 122 | 132 | 0.421260 | 0.421260 |
| strong utility action | 172 | 0.677 | 0.452773 | 0.433413 | 0.019359 | 45 | 127 | 0.197674 | 0.197674 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 162 | 0.638 | 0.446100 | 0.426958 | 0.019142 | 45 | 117 | 0.209877 | 0.209877 |
| recent utility last 5s | 20 | 0.079 | 0.285083 | 0.292616 | -0.007533 | 10 | 10 | 0.500000 | 0.500000 |
| flash effect present | 254 | 1.000 | 0.325586 | 0.312932 | 0.012654 | 122 | 132 | 0.421260 | 0.421260 |

## Active Smoke/Inferno Intervals

- `7.0s` - `64.5s`, rows `116`
- `67.5s` - `90.0s`, rows `46`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `75.5`, LSTM `0.1455`, XGBoost `0.2664`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `76.0`, LSTM `0.1288`, XGBoost `0.2310`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `75.0`, LSTM `0.1804`, XGBoost `0.2666`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.5754`, XGBoost `0.5054`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.5731`, XGBoost `0.5090`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.5803`, XGBoost `0.5165`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.5725`, XGBoost `0.5090`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.5722`, XGBoost `0.5090`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.5793`, XGBoost `0.5165`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.5772`, XGBoost `0.5163`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
