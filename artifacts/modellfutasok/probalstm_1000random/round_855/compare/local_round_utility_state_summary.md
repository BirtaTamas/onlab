# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-imperial-vs-legacy-bo3-GRvbnL5Q4zT_JzAd-0AXgo/imperial-vs-legacy-m2-dust2.csv`
- round_num: `11`
- rows: `214`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 214 | 1.000 | 0.141621 | 0.258031 | -0.116410 | 214 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 214 | 1.000 | 0.141621 | 0.258031 | -0.116410 | 214 | 0 | 1.000000 | 1.000000 |
| strong utility action | 143 | 0.668 | 0.159731 | 0.293012 | -0.133280 | 143 | 0 | 1.000000 | 1.000000 |
| utility damage | 16 | 0.075 | 0.029212 | 0.205374 | -0.176162 | 16 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 143 | 0.668 | 0.159731 | 0.293012 | -0.133280 | 143 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 214 | 1.000 | 0.141621 | 0.258031 | -0.116410 | 214 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `72.5s`, rows `129`
- `98.5s` - `105.0s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `103.0`, LSTM `0.0452`, XGBoost `0.3470`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `105.0`, LSTM `0.0504`, XGBoost `0.3464`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `104.0`, LSTM `0.0457`, XGBoost `0.3372`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `103.5`, LSTM `0.0409`, XGBoost `0.3297`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `104.5`, LSTM `0.0491`, XGBoost `0.3364`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.0396`, XGBoost `0.3196`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `125.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.0372`, XGBoost `0.3054`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `125.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.0341`, XGBoost `0.2879`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `72.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.0384`, XGBoost `0.2869`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `72.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.0454`, XGBoost `0.2869`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `72.0`, recent_utility `0`
