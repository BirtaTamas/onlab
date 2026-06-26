# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-heroic-vs-nrg-dust2-QDtqFlW1Z9UhZpBNOAavnd/heroic-vs-nrg-dust2.csv`
- round_num: `6`
- rows: `204`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 204 | 1.000 | 0.323455 | 0.396059 | -0.072604 | 200 | 4 | 0.612745 | 0.612745 |
| active/recent utility | 204 | 1.000 | 0.323455 | 0.396059 | -0.072604 | 200 | 4 | 0.612745 | 0.612745 |
| strong utility action | 150 | 0.735 | 0.404989 | 0.497237 | -0.092248 | 147 | 3 | 0.513333 | 0.513333 |
| utility damage | 10 | 0.049 | 0.172650 | 0.347193 | -0.174543 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 140 | 0.686 | 0.384133 | 0.480295 | -0.096162 | 138 | 2 | 0.550000 | 0.550000 |
| recent utility last 5s | 21 | 0.103 | 0.712374 | 0.774892 | -0.062517 | 20 | 1 | 0.000000 | 0.000000 |
| flash effect present | 204 | 1.000 | 0.323455 | 0.396059 | -0.072604 | 200 | 4 | 0.612745 | 0.612745 |

## Active Smoke/Inferno Intervals

- `8.0s` - `72.5s`, rows `130`
- `97.0s` - `101.5s`, rows `10`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `65.5`, LSTM `0.1109`, XGBoost `0.3440`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.0`, LSTM `0.1100`, XGBoost `0.3322`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.5`, LSTM `0.1134`, XGBoost `0.3337`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.1305`, XGBoost `0.3463`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `15.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.1310`, XGBoost `0.3467`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `15.0`, recent_utility `0`
- seconds `67.0`, LSTM `0.1161`, XGBoost `0.3259`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.5`, LSTM `0.1422`, XGBoost `0.3500`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.1368`, XGBoost `0.3441`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `15.0`, recent_utility `0`
- seconds `67.5`, LSTM `0.0793`, XGBoost `0.2766`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.5`, LSTM `0.1456`, XGBoost `0.3423`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
