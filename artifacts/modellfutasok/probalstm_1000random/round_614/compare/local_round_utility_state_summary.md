# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-furia-vs-vitality-bo3-3MYCYJWfx_8le7ueost7BH/furia-vs-vitality-m1-nuke.csv`
- round_num: `1`
- rows: `166`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 166 | 1.000 | 0.343093 | 0.404462 | -0.061368 | 166 | 0 | 0.373494 | 0.355422 |
| active/recent utility | 142 | 0.855 | 0.314280 | 0.374711 | -0.060431 | 142 | 0 | 0.436620 | 0.415493 |
| strong utility action | 117 | 0.705 | 0.380347 | 0.448675 | -0.068328 | 117 | 0 | 0.316239 | 0.290598 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 117 | 0.705 | 0.380347 | 0.448675 | -0.068328 | 117 | 0 | 0.316239 | 0.290598 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 66 | 0.398 | 0.080358 | 0.136161 | -0.055804 | 66 | 0 | 0.939394 | 0.893939 |

## Active Smoke/Inferno Intervals

- `10.5s` - `46.5s`, rows `73`
- `48.5s` - `70.0s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `55.5`, LSTM `0.4440`, XGBoost `0.6426`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.1286`, XGBoost `0.2879`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.5`, LSTM `0.1554`, XGBoost `0.2742`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.1613`, XGBoost `0.2769`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.1653`, XGBoost `0.2741`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.1768`, XGBoost `0.2731`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `62.0`, LSTM `0.0382`, XGBoost `0.1317`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `67.0`, LSTM `0.0163`, XGBoost `0.1020`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.5`, LSTM `0.0171`, XGBoost `0.1020`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.5`, LSTM `0.0171`, XGBoost `0.1020`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
