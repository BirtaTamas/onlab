# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-furia-vs-pain-bo3-BGpRMXEt8xpbRAS7KbpPH6/furia-vs-pain-m2-overpass.csv`
- round_num: `12`
- rows: `189`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 189 | 1.000 | 0.581562 | 0.658995 | -0.077433 | 2 | 187 | 0.444444 | 0.968254 |
| active/recent utility | 189 | 1.000 | 0.581562 | 0.658995 | -0.077433 | 2 | 187 | 0.444444 | 0.968254 |
| strong utility action | 123 | 0.651 | 0.618524 | 0.685141 | -0.066617 | 2 | 121 | 0.536585 | 0.951220 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 123 | 0.651 | 0.618524 | 0.685141 | -0.066617 | 2 | 121 | 0.536585 | 0.951220 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 189 | 1.000 | 0.581562 | 0.658995 | -0.077433 | 2 | 187 | 0.444444 | 0.968254 |

## Active Smoke/Inferno Intervals

- `8.5s` - `39.0s`, rows `62`
- `64.0s` - `94.0s`, rows `61`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `11.0`, LSTM `0.4012`, XGBoost `0.5463`, closer `xgboost`, smoke `2`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.4350`, XGBoost `0.5613`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.4387`, XGBoost `0.5613`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.4407`, XGBoost `0.5613`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.4279`, XGBoost `0.5471`, closer `xgboost`, smoke `2`, inferno `5`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.4430`, XGBoost `0.5613`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.4440`, XGBoost `0.5613`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.4240`, XGBoost `0.5394`, closer `xgboost`, smoke `2`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.4469`, XGBoost `0.5613`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.4372`, XGBoost `0.5471`, closer `xgboost`, smoke `2`, inferno `5`, utility_damage `0.0`, recent_utility `0`
