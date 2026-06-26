# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-imperial-vs-legacy-bo3-GRvbnL5Q4zT_JzAd-0AXgo/imperial-vs-legacy-m3-mirage.csv`
- round_num: `12`
- rows: `143`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 143 | 1.000 | 0.091815 | 0.150240 | -0.058425 | 142 | 1 | 1.000000 | 0.874126 |
| active/recent utility | 143 | 1.000 | 0.091815 | 0.150240 | -0.058425 | 142 | 1 | 1.000000 | 0.874126 |
| strong utility action | 108 | 0.755 | 0.064811 | 0.109892 | -0.045081 | 107 | 1 | 1.000000 | 0.944444 |
| utility damage | 10 | 0.070 | 0.279361 | 0.338790 | -0.059429 | 9 | 1 | 1.000000 | 0.700000 |
| active smoke/inferno | 108 | 0.755 | 0.064811 | 0.109892 | -0.045081 | 107 | 1 | 1.000000 | 0.944444 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 143 | 1.000 | 0.091815 | 0.150240 | -0.058425 | 142 | 1 | 1.000000 | 0.874126 |

## Active Smoke/Inferno Intervals

- `6.0s` - `38.5s`, rows `66`
- `50.5s` - `71.0s`, rows `42`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `17.5`, LSTM `0.1193`, XGBoost `0.2781`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.1220`, XGBoost `0.2713`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.1446`, XGBoost `0.2754`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.0266`, XGBoost `0.1452`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.0283`, XGBoost `0.1452`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.0806`, XGBoost `0.1971`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.0306`, XGBoost `0.1452`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.0324`, XGBoost `0.1452`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.4751`, XGBoost `0.5842`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `13.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.1608`, XGBoost `0.2684`, closer `lstm`, smoke `2`, inferno `3`, utility_damage `0.0`, recent_utility `0`
