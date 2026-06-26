# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-heroic-vs-aurora-bo3-0XrprgXu_t-aBJHUPpJYb4/heroic-vs-aurora-m1-overpass.csv`
- round_num: `3`
- rows: `198`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 198 | 1.000 | 0.839653 | 0.938155 | -0.098502 | 0 | 198 | 0.979798 | 1.000000 |
| active/recent utility | 198 | 1.000 | 0.839653 | 0.938155 | -0.098502 | 0 | 198 | 0.979798 | 1.000000 |
| strong utility action | 164 | 0.828 | 0.830780 | 0.935736 | -0.104956 | 0 | 164 | 0.975610 | 1.000000 |
| utility damage | 10 | 0.051 | 0.695030 | 0.917279 | -0.222249 | 0 | 10 | 0.700000 | 1.000000 |
| active smoke/inferno | 164 | 0.828 | 0.830780 | 0.935736 | -0.104956 | 0 | 164 | 0.975610 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 198 | 1.000 | 0.839653 | 0.938155 | -0.098502 | 0 | 198 | 0.979798 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `32.0s`, rows `48`
- `34.5s` - `60.5s`, rows `53`
- `67.5s` - `98.5s`, rows `63`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `87.5`, LSTM `0.4514`, XGBoost `0.8623`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `17.0`, recent_utility `0`
- seconds `87.0`, LSTM `0.4642`, XGBoost `0.8626`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `17.0`, recent_utility `0`
- seconds `86.5`, LSTM `0.4749`, XGBoost `0.8500`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `86.0`, LSTM `0.5153`, XGBoost `0.8658`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `85.0`, LSTM `0.5364`, XGBoost `0.8598`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `88.0`, LSTM `0.4641`, XGBoost `0.7865`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `17.0`, recent_utility `0`
- seconds `85.5`, LSTM `0.5409`, XGBoost `0.8624`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `84.5`, LSTM `0.5485`, XGBoost `0.8590`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `89.0`, LSTM `0.6159`, XGBoost `0.9216`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `17.0`, recent_utility `0`
- seconds `83.0`, LSTM `0.5882`, XGBoost `0.8904`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
