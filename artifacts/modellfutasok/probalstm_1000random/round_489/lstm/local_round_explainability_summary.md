# Local Round Explainability

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-ninjas-in-pyjamas-vs-natus-vincere-bo3-NyGFAYn592Hu0YfljTVG0N/ninjas-in-pyjamas-vs-natus-vincere-m2-inferno.csv`
- round_num: `1`

## Largest probability jumps

- tick `4826`, seconds `55.50`, LSTM `0.2041`, delta `-0.2865`
- tick `4602`, seconds `52.00`, LSTM `0.4075`, delta `+0.2515`
- tick `5754`, seconds `70.00`, LSTM `0.2780`, delta `+0.2246`
- tick `4442`, seconds `49.50`, LSTM `0.2588`, delta `-0.1999`
- tick `7034`, seconds `90.00`, LSTM `0.0496`, delta `-0.1533`
- tick `6106`, seconds `75.50`, LSTM `0.3576`, delta `+0.0949`
- tick `4858`, seconds `56.00`, LSTM `0.1239`, delta `-0.0802`
- tick `4634`, seconds `52.50`, LSTM `0.4751`, delta `+0.0676`
- tick `6138`, seconds `76.00`, LSTM `0.2973`, delta `-0.0603`
- tick `5946`, seconds `73.00`, LSTM `0.3332`, delta `-0.0503`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.006803`, |coef| `0.006803`
- `lag_00__damage_diff_last_5s`: coefficient `0.005256`, |coef| `0.005256`
- `lag_00__CT_kills_last_3s`: coefficient `0.004515`, |coef| `0.004515`
- `lag_14__CT_place_ARCH`: coefficient `0.004159`, |coef| `0.004159`
- `lag_08__CT_place_ARCH`: coefficient `-0.004061`, |coef| `0.004061`
- `lag_00__T_kills_last_3s`: coefficient `-0.004000`, |coef| `0.004000`
- `lag_12__T1__is_walking`: coefficient `0.003483`, |coef| `0.003483`
- `lag_00__CT_place_APARTMENTS`: coefficient `0.002986`, |coef| `0.002986`
- `lag_12__CT5__is_walking`: coefficient `0.002754`, |coef| `0.002754`
- `lag_09__CT_place_BALCONY`: coefficient `-0.002743`, |coef| `0.002743`
- `lag_00__T_damage_last_5s`: coefficient `-0.002684`, |coef| `0.002684`
- `lag_00__CT_damage_last_5s`: coefficient `0.002639`, |coef| `0.002639`
- `lag_12__CT4__is_walking`: coefficient `0.002526`, |coef| `0.002526`
- `lag_04__T4__duck_amount`: coefficient `-0.002437`, |coef| `0.002437`
- `lag_00__alive_diff`: coefficient `0.002434`, |coef| `0.002434`

## Top 10 utility ridge features

- `lag_06__T5__smoke`: coefficient `-0.002026` (lowers CT win probability)
- `lag_09__T1__smoke`: coefficient `-0.001948` (lowers CT win probability)
- `lag_01__T1__molly`: coefficient `-0.001885` (lowers CT win probability)
- `lag_06__CT3__flash_duration`: coefficient `0.001700` (raises CT win probability)
- `lag_11__T4__flash_duration`: coefficient `-0.001555` (lowers CT win probability)
- `lag_06__CT5__flash_duration`: coefficient `0.001545` (raises CT win probability)
- `lag_12__T4__flash_duration`: coefficient `-0.001537` (lowers CT win probability)
- `lag_00__CT5__flash_duration`: coefficient `0.001536` (raises CT win probability)
- `lag_06__CT_flash_duration_sum`: coefficient `0.001484` (raises CT win probability)
- `lag_06__T_B_site_active_smokes`: coefficient `0.001449` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.006803` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.005256` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.004515` (raises CT win probability)
- `lag_14__CT_place_ARCH`: coefficient `0.004159` (raises CT win probability)
- `lag_08__CT_place_ARCH`: coefficient `-0.004061` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.004000` (lowers CT win probability)
- `lag_12__T1__is_walking`: coefficient `0.003483` (raises CT win probability)
- `lag_00__CT_place_APARTMENTS`: coefficient `0.002986` (raises CT win probability)
- `lag_12__CT5__is_walking`: coefficient `0.002754` (raises CT win probability)
- `lag_09__CT_place_BALCONY`: coefficient `-0.002743` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `4826`, seconds `55.50`, LSTM delta `-0.2865`

Top all feature movements:
- `lag_09__CT_place_BALCONY`: contribution `-0.017605`
- `lag_08__CT_place_ARCH`: contribution `-0.016571`
- `lag_00__kill_diff_last_3s`: contribution `-0.016374`
- `lag_06__CT_place_BALCONY`: contribution `-0.013921`
- `lag_00__T_kills_last_3s`: contribution `-0.012671`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `4602`, seconds `52.00`, LSTM delta `+0.2515`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.016374`
- `lag_02__CT_place_BALCONY`: contribution `+0.013386`
- `lag_00__CT_kills_last_3s`: contribution `+0.013035`
- `lag_00__damage_diff_last_5s`: contribution `+0.009842`
- `lag_04__T4__duck_amount`: contribution `+0.008326`

Top utility-only movements:
- `lag_11__T4__flash_duration`: contribution `+0.004922`

### tick `5754`, seconds `70.00`, LSTM delta `+0.2246`

Top all feature movements:
- `lag_14__CT_place_ARCH`: contribution `+0.033944`
- `lag_08__CT_place_ARCH`: contribution `+0.033142`
- `lag_00__kill_diff_last_3s`: contribution `+0.016374`
- `lag_00__CT_kills_last_3s`: contribution `+0.013035`
- `lag_00__damage_diff_last_5s`: contribution `+0.011858`

Top utility-only movements:
- `lag_06__T5__smoke`: contribution `+0.004391`
- `lag_09__T1__smoke`: contribution `+0.004204`
- `lag_01__T1__molly`: contribution `+0.004174`

### tick `4442`, seconds `49.50`, LSTM delta `-0.1999`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.016374`
- `lag_00__T_kills_last_3s`: contribution `-0.012671`
- `lag_00__CT_place_APARTMENTS`: contribution `-0.011471`
- `lag_00__damage_diff_last_5s`: contribution `-0.009842`
- `lag_12__T1__is_walking`: contribution `-0.007948`

Top utility-only movements:
- `lag_12__T4__flash_duration`: contribution `-0.004864`
- `lag_06__T4__flash_duration`: contribution `-0.004250`

### tick `7034`, seconds `90.00`, LSTM delta `-0.1533`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.016374`
- `lag_00__T_kills_last_3s`: contribution `-0.012671`
- `lag_08__T1__duck_amount`: contribution `-0.006676`
- `lag_10__T1__duck_amount`: contribution `-0.006660`
- `lag_00__damage_diff_last_5s`: contribution `-0.006522`

Top utility-only movements:
- No utility movement among the top local contributors.
