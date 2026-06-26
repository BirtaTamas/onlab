# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-tyloo-vs-nrg-anubis-OygKONihup8TZ7k3ClDb0W/tyloo-vs-nrg-anubis.csv`
- round_num: `7`
- rows: `186`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 186 | 1.000 | 0.562006 | 0.568569 | -0.006563 | 78 | 108 | 0.747312 | 0.747312 |
| active/recent utility | 186 | 1.000 | 0.562006 | 0.568569 | -0.006563 | 78 | 108 | 0.747312 | 0.747312 |
| strong utility action | 158 | 0.849 | 0.526069 | 0.532522 | -0.006453 | 67 | 91 | 0.702532 | 0.702532 |
| utility damage | 22 | 0.118 | 0.703762 | 0.706777 | -0.003016 | 10 | 12 | 1.000000 | 1.000000 |
| active smoke/inferno | 158 | 0.849 | 0.526069 | 0.532522 | -0.006453 | 67 | 91 | 0.702532 | 0.702532 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 186 | 1.000 | 0.562006 | 0.568569 | -0.006563 | 78 | 108 | 0.747312 | 0.747312 |

## Active Smoke/Inferno Intervals

- `7.0s` - `85.5s`, rows `158`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `81.0`, LSTM `0.6949`, XGBoost `0.5414`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.5499`, XGBoost `0.6553`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `61.5`, LSTM `0.0132`, XGBoost `0.1110`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.0`, LSTM `0.0186`, XGBoost `0.1113`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `81.5`, LSTM `0.4540`, XGBoost `0.3657`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `57.5`, LSTM `0.3168`, XGBoost `0.4037`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `82.0`, LSTM `0.4506`, XGBoost `0.3678`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.0`, LSTM `0.0353`, XGBoost `0.1111`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.6436`, XGBoost `0.7188`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.6450`, XGBoost `0.7194`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
