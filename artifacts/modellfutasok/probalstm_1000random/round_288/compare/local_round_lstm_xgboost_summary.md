# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-tyloo-vs-nrg-anubis-OygKONihup8TZ7k3ClDb0W/tyloo-vs-nrg-anubis.csv`
- round_num: `8`
- rows: `214`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.432194 | 0.206373 | 0.591504 | 0.724299 | 0.567806 |
| xgboost | 0.449953 | 0.226206 | 0.633638 | 0.785047 | 0.550047 |

## Closer Per Tick

- lstm: `111`
- xgboost: `103`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
