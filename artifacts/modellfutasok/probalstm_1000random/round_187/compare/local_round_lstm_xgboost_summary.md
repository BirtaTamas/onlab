# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-tyloo-vs-nrg-anubis-OygKONihup8TZ7k3ClDb0W/tyloo-vs-nrg-anubis.csv`
- round_num: `9`
- rows: `193`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.425087 | 0.208360 | 0.602082 | 0.829016 | 0.574913 |
| xgboost | 0.472362 | 0.268792 | 0.754521 | 0.637306 | 0.527638 |

## Closer Per Tick

- lstm: `146`
- xgboost: `47`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
