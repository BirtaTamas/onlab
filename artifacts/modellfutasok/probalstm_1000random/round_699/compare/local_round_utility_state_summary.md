# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-nrg-vs-vitality-bo3-7fVRmFXct71SEzAlz8IKvC/nrg-vs-vitality-m1-mirage.csv`
- round_num: `5`
- rows: `151`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 151 | 1.000 | 0.034744 | 0.073362 | -0.038618 | 150 | 1 | 1.000000 | 1.000000 |
| active/recent utility | 151 | 1.000 | 0.034744 | 0.073362 | -0.038618 | 150 | 1 | 1.000000 | 1.000000 |
| strong utility action | 99 | 0.656 | 0.043239 | 0.079159 | -0.035920 | 98 | 1 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 99 | 0.656 | 0.043239 | 0.079159 | -0.035920 | 98 | 1 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 151 | 1.000 | 0.034744 | 0.073362 | -0.038618 | 150 | 1 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.5s` - `59.5s`, rows `99`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `10.5`, LSTM `0.0677`, XGBoost `0.1608`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.0701`, XGBoost `0.1614`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.0728`, XGBoost `0.1608`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.0825`, XGBoost `0.1627`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.0830`, XGBoost `0.1627`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.0818`, XGBoost `0.1613`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.0837`, XGBoost `0.1623`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.0850`, XGBoost `0.1623`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.0848`, XGBoost `0.1614`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.0860`, XGBoost `0.1623`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
