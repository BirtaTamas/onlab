# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-astralis-vs-natus-vincere-bo3-4-6Sb81TUo41h9OxcK0xKz/astralis-vs-natus-vincere-m3-nuke.csv`
- round_num: `5`
- rows: `146`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 146 | 1.000 | 0.117674 | 0.196342 | -0.078668 | 124 | 22 | 1.000000 | 1.000000 |
| active/recent utility | 146 | 1.000 | 0.117674 | 0.196342 | -0.078668 | 124 | 22 | 1.000000 | 1.000000 |
| strong utility action | 82 | 0.562 | 0.161370 | 0.258392 | -0.097022 | 61 | 21 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 82 | 0.562 | 0.161370 | 0.258392 | -0.097022 | 61 | 21 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 146 | 1.000 | 0.117674 | 0.196342 | -0.078668 | 124 | 22 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.5s` - `50.0s`, rows `82`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `10.5`, LSTM `0.1544`, XGBoost `0.3586`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.1621`, XGBoost `0.3655`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.1574`, XGBoost `0.3601`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.1659`, XGBoost `0.3651`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.1660`, XGBoost `0.3639`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.1619`, XGBoost `0.3586`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.1709`, XGBoost `0.3651`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.1716`, XGBoost `0.3644`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.1724`, XGBoost `0.3651`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.1799`, XGBoost `0.3655`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
