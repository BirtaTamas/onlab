# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-faze-vs-virtuspro-bo3-YK5CkfojMMEdQ3U0_CVA2l/faze-vs-virtuspro-m1-anubis.csv`
- round_num: `3`
- rows: `227`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 227 | 1.000 | 0.072160 | 0.153041 | -0.080881 | 227 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 227 | 1.000 | 0.072160 | 0.153041 | -0.080881 | 227 | 0 | 1.000000 | 1.000000 |
| strong utility action | 195 | 0.859 | 0.071269 | 0.152120 | -0.080851 | 195 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 185 | 0.815 | 0.069729 | 0.145077 | -0.075348 | 185 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.044 | 0.099768 | 0.282421 | -0.182654 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 227 | 1.000 | 0.072160 | 0.153041 | -0.080881 | 227 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `14.0s` - `84.0s`, rows `141`
- `89.0s` - `110.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `1.5`, LSTM `0.0716`, XGBoost `0.2941`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `2.0`, LSTM `0.0773`, XGBoost `0.2932`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `2.5`, LSTM `0.0778`, XGBoost `0.2932`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `3.0`, LSTM `0.0782`, XGBoost `0.2815`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `3.5`, LSTM `0.0935`, XGBoost `0.2815`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `4.5`, LSTM `0.1057`, XGBoost `0.2824`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `4.0`, LSTM `0.1102`, XGBoost `0.2815`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `23.0`, LSTM `0.1030`, XGBoost `0.2686`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.1042`, XGBoost `0.2686`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `5.0`, LSTM `0.1156`, XGBoost `0.2778`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
