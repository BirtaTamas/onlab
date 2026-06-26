# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-furia-vs-vitality-bo3-3MYCYJWfx_8le7ueost7BH/furia-vs-vitality-m1-nuke.csv`
- round_num: `21`
- rows: `284`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 284 | 1.000 | 0.302851 | 0.291230 | 0.011621 | 142 | 142 | 0.619718 | 0.732394 |
| active/recent utility | 284 | 1.000 | 0.302851 | 0.291230 | 0.011621 | 142 | 142 | 0.619718 | 0.732394 |
| strong utility action | 180 | 0.634 | 0.369143 | 0.358117 | 0.011027 | 106 | 74 | 0.488889 | 0.666667 |
| utility damage | 19 | 0.067 | 0.571890 | 0.577468 | -0.005578 | 9 | 10 | 0.000000 | 0.000000 |
| active smoke/inferno | 180 | 0.634 | 0.369143 | 0.358117 | 0.011027 | 106 | 74 | 0.488889 | 0.666667 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 284 | 1.000 | 0.302851 | 0.291230 | 0.011621 | 142 | 142 | 0.619718 | 0.732394 |

## Active Smoke/Inferno Intervals

- `8.0s` - `51.0s`, rows `87`
- `61.5s` - `107.5s`, rows `93`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `76.5`, LSTM `0.4665`, XGBoost `0.2332`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.0`, LSTM `0.3734`, XGBoost `0.2332`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.3528`, XGBoost `0.2367`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.5145`, XGBoost `0.4020`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.5005`, XGBoost `0.4014`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.5332`, XGBoost `0.4383`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.5324`, XGBoost `0.4383`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.0`, LSTM `0.4948`, XGBoost `0.4020`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.6378`, XGBoost `0.7290`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.3277`, XGBoost `0.2367`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
