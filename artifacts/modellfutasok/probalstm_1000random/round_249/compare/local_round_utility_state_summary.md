# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-og-vs-nrg-bo3-GH6ZBFOA9sfdeCxgnhHN9f/og-vs-nrg-m2-nuke.csv`
- round_num: `8`
- rows: `214`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 214 | 1.000 | 0.121639 | 0.152889 | -0.031249 | 194 | 20 | 0.981308 | 0.789720 |
| active/recent utility | 214 | 1.000 | 0.121639 | 0.152889 | -0.031249 | 194 | 20 | 0.981308 | 0.789720 |
| strong utility action | 118 | 0.551 | 0.157235 | 0.196540 | -0.039305 | 98 | 20 | 0.966102 | 0.754237 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 118 | 0.551 | 0.157235 | 0.196540 | -0.039305 | 98 | 20 | 0.966102 | 0.754237 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 214 | 1.000 | 0.121639 | 0.152889 | -0.031249 | 194 | 20 | 0.981308 | 0.789720 |

## Active Smoke/Inferno Intervals

- `8.0s` - `57.0s`, rows `99`
- `58.5s` - `65.0s`, rows `14`
- `104.5s` - `106.5s`, rows `5`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `25.0`, LSTM `0.2310`, XGBoost `0.4184`, closer `lstm`, smoke `6`, inferno `2`, utility_damage `30.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.2769`, XGBoost `0.4173`, closer `lstm`, smoke `6`, inferno `2`, utility_damage `27.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.2887`, XGBoost `0.4188`, closer `lstm`, smoke `6`, inferno `2`, utility_damage `24.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.4506`, XGBoost `0.5720`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.4504`, XGBoost `0.5715`, closer `lstm`, smoke `1`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.4580`, XGBoost `0.5715`, closer `lstm`, smoke `2`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.4573`, XGBoost `0.5708`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.4602`, XGBoost `0.5717`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.4606`, XGBoost `0.5708`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.4609`, XGBoost `0.5708`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
