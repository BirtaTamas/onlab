# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-tyloo-vs-metizport-mirage-uJE2h4ym3PvBPopNN8-YOA/tyloo-vs-metizport-mirage.csv`
- round_num: `3`
- rows: `171`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 171 | 1.000 | 0.010700 | 0.052491 | -0.041791 | 171 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 171 | 1.000 | 0.010700 | 0.052491 | -0.041791 | 171 | 0 | 1.000000 | 1.000000 |
| strong utility action | 111 | 0.649 | 0.013250 | 0.058259 | -0.045008 | 111 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 111 | 0.649 | 0.013250 | 0.058259 | -0.045008 | 111 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 171 | 1.000 | 0.010700 | 0.052491 | -0.041791 | 171 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.5s` - `64.5s`, rows `111`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `28.0`, LSTM `0.0209`, XGBoost `0.1334`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.0227`, XGBoost `0.1334`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.0273`, XGBoost `0.1334`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.0299`, XGBoost `0.1334`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.0303`, XGBoost `0.1316`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.0281`, XGBoost `0.1291`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.0264`, XGBoost `0.1271`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.0318`, XGBoost `0.1308`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.0360`, XGBoost `0.1348`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.0289`, XGBoost `0.1271`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
