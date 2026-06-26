# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2_finals/blast-bounty-2025-season-2-finals-the-mongolz-vs-aurora-bo3-BL3Rcvf733R8cy7TtLY5HE/the-mongolz-vs-aurora-m1-dust2.csv`
- round_num: `3`
- rows: `192`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 192 | 1.000 | 0.024335 | 0.039407 | -0.015071 | 180 | 12 | 1.000000 | 1.000000 |
| active/recent utility | 192 | 1.000 | 0.024335 | 0.039407 | -0.015071 | 180 | 12 | 1.000000 | 1.000000 |
| strong utility action | 116 | 0.604 | 0.031421 | 0.048889 | -0.017468 | 109 | 7 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 106 | 0.552 | 0.032833 | 0.045609 | -0.012775 | 99 | 7 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.052 | 0.016448 | 0.083656 | -0.067208 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 192 | 1.000 | 0.024335 | 0.039407 | -0.015071 | 180 | 12 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `13.5s` - `44.0s`, rows `62`
- `64.0s` - `85.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `1.0`, LSTM `0.0134`, XGBoost `0.0913`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `2.5`, LSTM `0.0186`, XGBoost `0.0960`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `1.5`, LSTM `0.0134`, XGBoost `0.0906`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `0.5`, LSTM `0.0157`, XGBoost `0.0913`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `2.0`, LSTM `0.0165`, XGBoost `0.0906`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `3.0`, LSTM `0.0209`, XGBoost `0.0927`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `4.0`, LSTM `0.0144`, XGBoost `0.0721`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `4.5`, LSTM `0.0159`, XGBoost `0.0716`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `5.0`, LSTM `0.0181`, XGBoost `0.0710`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `3.5`, LSTM `0.0175`, XGBoost `0.0692`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
