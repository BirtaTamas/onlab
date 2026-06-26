# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-faze-vs-virtuspro-bo3-YK5CkfojMMEdQ3U0_CVA2l/faze-vs-virtuspro-m1-anubis.csv`
- round_num: `9`
- rows: `177`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 177 | 1.000 | 0.396049 | 0.466716 | -0.070667 | 169 | 8 | 0.779661 | 0.774011 |
| active/recent utility | 177 | 1.000 | 0.396049 | 0.466716 | -0.070667 | 169 | 8 | 0.779661 | 0.774011 |
| strong utility action | 163 | 0.921 | 0.406998 | 0.478768 | -0.071770 | 156 | 7 | 0.760736 | 0.754601 |
| utility damage | 11 | 0.062 | 0.784211 | 0.843941 | -0.059730 | 10 | 1 | 0.000000 | 0.000000 |
| active smoke/inferno | 151 | 0.853 | 0.413530 | 0.486866 | -0.073336 | 144 | 7 | 0.754967 | 0.748344 |
| recent utility last 5s | 10 | 0.056 | 0.270828 | 0.327524 | -0.056696 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 177 | 1.000 | 0.396049 | 0.466716 | -0.070667 | 169 | 8 | 0.779661 | 0.774011 |

## Active Smoke/Inferno Intervals

- `12.0s` - `87.0s`, rows `151`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `18.0`, LSTM `0.1701`, XGBoost `0.3248`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.0`, LSTM `0.6001`, XGBoost `0.7419`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `10.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.1923`, XGBoost `0.3267`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.1923`, XGBoost `0.3267`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.2014`, XGBoost `0.3317`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.3210`, XGBoost `0.4494`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `10.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.2003`, XGBoost `0.3272`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.2008`, XGBoost `0.3277`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.2007`, XGBoost `0.3264`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.2017`, XGBoost `0.3272`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
