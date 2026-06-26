# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-tyloo-vs-nrg-anubis-OygKONihup8TZ7k3ClDb0W/tyloo-vs-nrg-anubis.csv`
- round_num: `15`
- rows: `248`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.304776 | 0.157999 | 0.431194 | 0.560484 | 0.304776 |
| xgboost | 0.265086 | 0.119079 | 0.352973 | 0.963710 | 0.265086 |

## Closer Per Tick

- lstm: `104`
- xgboost: `144`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
