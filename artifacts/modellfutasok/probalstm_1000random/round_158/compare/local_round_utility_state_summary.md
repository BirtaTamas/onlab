# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-legacy-vs-vitality-bo3-43WNFDazpfbmBN3Sj5hWmP/vitality-vs-legacy-m2-dust2.csv`
- round_num: `3`
- rows: `198`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 198 | 1.000 | 0.163570 | 0.358000 | -0.194430 | 193 | 5 | 1.000000 | 1.000000 |
| active/recent utility | 198 | 1.000 | 0.163570 | 0.358000 | -0.194430 | 193 | 5 | 1.000000 | 1.000000 |
| strong utility action | 169 | 0.854 | 0.142210 | 0.352396 | -0.210186 | 169 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 169 | 0.854 | 0.142210 | 0.352396 | -0.210186 | 169 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.051 | 0.087981 | 0.356321 | -0.268340 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 198 | 1.000 | 0.163570 | 0.358000 | -0.194430 | 193 | 5 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.5s` - `61.5s`, rows `105`
- `67.0s` - `98.5s`, rows `64`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `76.0`, LSTM `0.0619`, XGBoost `0.3762`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `75.0`, LSTM `0.0662`, XGBoost `0.3737`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `75.5`, LSTM `0.0659`, XGBoost `0.3703`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `81.0`, LSTM `0.0833`, XGBoost `0.3866`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `74.0`, LSTM `0.0554`, XGBoost `0.3477`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `38.0`, LSTM `0.0605`, XGBoost `0.3510`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `80.0`, LSTM `0.0861`, XGBoost `0.3763`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `74.5`, LSTM `0.0879`, XGBoost `0.3751`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `73.5`, LSTM `0.0616`, XGBoost `0.3477`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `76.5`, LSTM `0.0934`, XGBoost `0.3793`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
