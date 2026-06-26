# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-big-vs-furia-bo3-8LyYppfzx0M6KmNUlhRuUi/big-vs-furia-m1-inferno.csv`
- round_num: `10`
- rows: `262`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 262 | 1.000 | 0.680931 | 0.701521 | -0.020590 | 92 | 170 | 0.927481 | 0.912214 |
| active/recent utility | 262 | 1.000 | 0.680931 | 0.701521 | -0.020590 | 92 | 170 | 0.927481 | 0.912214 |
| strong utility action | 210 | 0.802 | 0.671682 | 0.685915 | -0.014233 | 78 | 132 | 0.909524 | 0.890476 |
| utility damage | 45 | 0.172 | 0.736356 | 0.734130 | 0.002226 | 22 | 23 | 1.000000 | 1.000000 |
| active smoke/inferno | 200 | 0.763 | 0.658504 | 0.672049 | -0.013545 | 78 | 122 | 0.905000 | 0.885000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 262 | 1.000 | 0.680931 | 0.701521 | -0.020590 | 92 | 170 | 0.927481 | 0.912214 |

## Active Smoke/Inferno Intervals

- `6.5s` - `57.5s`, rows `103`
- `68.5s` - `116.5s`, rows `97`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `105.5`, LSTM `0.4979`, XGBoost `0.1717`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `106.0`, LSTM `0.4573`, XGBoost `0.1749`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `105.0`, LSTM `0.6303`, XGBoost `0.4263`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `106.5`, LSTM `0.3715`, XGBoost `0.1749`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `107.0`, LSTM `0.3653`, XGBoost `0.1814`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `111.0`, LSTM `0.2641`, XGBoost `0.0943`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `114.0`, LSTM `0.4983`, XGBoost `0.3328`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `110.5`, LSTM `0.2593`, XGBoost `0.0943`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `111.5`, LSTM `0.2506`, XGBoost `0.0943`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `102.0`, LSTM `0.6743`, XGBoost `0.5213`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
