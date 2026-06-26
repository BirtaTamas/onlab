# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-flyquest-vs-spirit-bo3-NmwBJVzYbgyZgcQrbNESHr/flyquest-vs-spirit-m1-anubis.csv`
- round_num: `14`
- rows: `130`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 130 | 1.000 | 0.022692 | 0.029995 | -0.007303 | 108 | 22 | 1.000000 | 1.000000 |
| active/recent utility | 130 | 1.000 | 0.022692 | 0.029995 | -0.007303 | 108 | 22 | 1.000000 | 1.000000 |
| strong utility action | 106 | 0.815 | 0.022746 | 0.031799 | -0.009053 | 97 | 9 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 96 | 0.738 | 0.023096 | 0.032518 | -0.009422 | 87 | 9 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.077 | 0.019389 | 0.024900 | -0.005510 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 130 | 1.000 | 0.022692 | 0.029995 | -0.007303 | 108 | 22 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `14.5s` - `36.0s`, rows `44`
- `39.0s` - `64.5s`, rows `52`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `64.0`, LSTM `0.0277`, XGBoost `0.1019`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.0400`, XGBoost `0.1031`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `62.0`, LSTM `0.0473`, XGBoost `0.1096`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `64.5`, LSTM `0.0262`, XGBoost `0.0874`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.0441`, XGBoost `0.1036`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `61.5`, LSTM `0.0675`, XGBoost `0.1109`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `57.5`, LSTM `0.1743`, XGBoost `0.1343`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.0408`, XGBoost `0.0770`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `57.0`, LSTM `0.0455`, XGBoost `0.0792`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `58.0`, LSTM `0.1013`, XGBoost `0.0723`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
