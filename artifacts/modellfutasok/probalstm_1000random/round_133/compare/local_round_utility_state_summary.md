# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-nrg-vs-vitality-bo3-7fVRmFXct71SEzAlz8IKvC/nrg-vs-vitality-m1-mirage.csv`
- round_num: `18`
- rows: `174`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 174 | 1.000 | 0.268993 | 0.333792 | -0.064800 | 146 | 28 | 0.913793 | 0.609195 |
| active/recent utility | 174 | 1.000 | 0.268993 | 0.333792 | -0.064800 | 146 | 28 | 0.913793 | 0.609195 |
| strong utility action | 162 | 0.931 | 0.252687 | 0.318316 | -0.065630 | 134 | 28 | 0.907407 | 0.654321 |
| utility damage | 17 | 0.098 | 0.451920 | 0.519843 | -0.067923 | 17 | 0 | 1.000000 | 0.058824 |
| active smoke/inferno | 162 | 0.931 | 0.252687 | 0.318316 | -0.065630 | 134 | 28 | 0.907407 | 0.654321 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 174 | 1.000 | 0.268993 | 0.333792 | -0.064800 | 146 | 28 | 0.913793 | 0.609195 |

## Active Smoke/Inferno Intervals

- `6.0s` - `86.5s`, rows `162`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `57.5`, LSTM `0.1627`, XGBoost `0.4022`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.1641`, XGBoost `0.4022`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.1684`, XGBoost `0.4022`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `57.0`, LSTM `0.1721`, XGBoost `0.4022`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.1728`, XGBoost `0.4022`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `54.5`, LSTM `0.1732`, XGBoost `0.4022`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.1763`, XGBoost `0.4022`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.1800`, XGBoost `0.4022`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.1861`, XGBoost `0.4022`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `56.5`, LSTM `0.1879`, XGBoost `0.4022`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
