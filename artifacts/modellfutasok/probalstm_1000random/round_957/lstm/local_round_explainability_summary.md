# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-3dmax-vs-vitality-nuke-h8drweGjLe5Dwjfuh5VfUb/3dmax-vs-vitality-nuke.csv`
- round_num: `7`

## Largest probability jumps

- tick `51158`, seconds `63.50`, LSTM `0.7677`, delta `+0.2843`
- tick `50454`, seconds `52.50`, LSTM `0.6856`, delta `+0.2458`
- tick `50422`, seconds `52.00`, LSTM `0.4398`, delta `+0.1602`
- tick `50486`, seconds `53.00`, LSTM `0.5303`, delta `-0.1553`
- tick `49878`, seconds `43.50`, LSTM `0.3456`, delta `-0.0991`
- tick `51318`, seconds `66.00`, LSTM `0.8856`, delta `+0.0853`
- tick `50294`, seconds `50.00`, LSTM `0.2668`, delta `-0.0649`
- tick `49974`, seconds `45.00`, LSTM `0.2497`, delta `-0.0510`
- tick `50070`, seconds `46.50`, LSTM `0.2779`, delta `+0.0427`
- tick `51126`, seconds `63.00`, LSTM `0.4835`, delta `-0.0406`

## Top 15 local ridge features

- `lag_10__CT_place_LOCKERROOM`: coefficient `-0.002874`, |coef| `0.002874`
- `lag_00__kill_diff_last_3s`: coefficient `0.002684`, |coef| `0.002684`
- `lag_04__CT1__is_scoped`: coefficient `-0.002526`, |coef| `0.002526`
- `lag_00__CT_kills_last_3s`: coefficient `0.002189`, |coef| `0.002189`
- `lag_00__damage_diff_last_5s`: coefficient `0.002086`, |coef| `0.002086`
- `lag_03__T_place_MINI`: coefficient `-0.002086`, |coef| `0.002086`
- `lag_14__CT4__duck_amount`: coefficient `-0.002035`, |coef| `0.002035`
- `lag_04__T_place_HUT`: coefficient `0.002016`, |coef| `0.002016`
- `lag_00__CT_place_ADMIN`: coefficient `-0.001854`, |coef| `0.001854`
- `lag_00__CT_damage_last_5s`: coefficient `0.001851`, |coef| `0.001851`
- `lag_04__T_place_SILO`: coefficient `-0.001744`, |coef| `0.001744`
- `lag_07__T2__duck_amount`: coefficient `-0.001742`, |coef| `0.001742`
- `lag_06__T_place_MINI`: coefficient `0.001715`, |coef| `0.001715`
- `lag_00__CT_place_GARAGE`: coefficient `0.001627`, |coef| `0.001627`
- `lag_04__CT_place_ADMIN`: coefficient `0.001624`, |coef| `0.001624`

## Top 10 utility ridge features

- `lag_05__CT1__flash_duration`: coefficient `0.001556` (raises CT win probability)
- `lag_00__T2__flash_duration`: coefficient `-0.001315` (lowers CT win probability)
- `lag_01__T2__flash_duration`: coefficient `-0.001277` (lowers CT win probability)
- `lag_07__CT5__flash_duration`: coefficient `0.001189` (raises CT win probability)
- `lag_00__CT3__flash_duration`: coefficient `-0.001038` (lowers CT win probability)
- `lag_05__CT_flash_duration_sum`: coefficient `0.000918` (raises CT win probability)
- `lag_02__T2__flash_duration`: coefficient `0.000885` (raises CT win probability)
- `lag_00__T5__flash_duration`: coefficient `-0.000839` (lowers CT win probability)
- `lag_00__T_flash_duration_sum`: coefficient `-0.000831` (lowers CT win probability)
- `lag_05__CT3__flash_duration`: coefficient `0.000812` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_10__CT_place_LOCKERROOM`: coefficient `-0.002874` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002684` (raises CT win probability)
- `lag_04__CT1__is_scoped`: coefficient `-0.002526` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002189` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002086` (raises CT win probability)
- `lag_03__T_place_MINI`: coefficient `-0.002086` (lowers CT win probability)
- `lag_14__CT4__duck_amount`: coefficient `-0.002035` (lowers CT win probability)
- `lag_04__T_place_HUT`: coefficient `0.002016` (raises CT win probability)
- `lag_00__CT_place_ADMIN`: coefficient `-0.001854` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001851` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `51158`, seconds `63.50`, LSTM delta `+0.2843`

Top all feature movements:
- `lag_10__CT_place_LOCKERROOM`: contribution `+0.035781`
- `lag_03__T_place_MINI`: contribution `+0.029019`
- `lag_06__T_place_MINI`: contribution `+0.023859`
- `lag_04__T_place_HUT`: contribution `+0.018794`
- `lag_08__T_place_MINI`: contribution `+0.017903`

Top utility-only movements:
- `lag_07__CT5__flash_duration`: contribution `+0.006157`
- `lag_06__CT1__flash_duration`: contribution `+0.002539`

### tick `50454`, seconds `52.50`, LSTM delta `+0.2458`

Top all feature movements:
- `lag_00__CT_place_ADMIN`: contribution `+0.012881`
- `lag_04__T_place_SILO`: contribution `+0.011849`
- `lag_04__CT_place_ADMIN`: contribution `+0.011282`
- `lag_05__CT1__flash_duration`: contribution `+0.008288`
- `lag_08__CT_place_MINI`: contribution `+0.007423`

Top utility-only movements:
- `lag_05__CT1__flash_duration`: contribution `+0.008288`
- `lag_01__T2__flash_duration`: contribution `+0.005752`
- `lag_05__CT_flash_duration_sum`: contribution `+0.003532`
- `lag_00__CT3__flash_duration`: contribution `+0.003378`

### tick `50422`, seconds `52.00`, LSTM delta `+0.1602`

Top all feature movements:
- `lag_04__CT1__is_scoped`: contribution `+0.010817`
- `lag_03__T_place_SILO`: contribution `+0.009147`
- `lag_03__CT_place_ADMIN`: contribution `+0.008146`
- `lag_14__CT4__duck_amount`: contribution `+0.007475`
- `lag_00__kill_diff_last_3s`: contribution `+0.006460`

Top utility-only movements:
- `lag_00__T2__flash_duration`: contribution `+0.005922`
- `lag_04__T2__flash_duration`: contribution `+0.002382`

### tick `50486`, seconds `53.00`, LSTM delta `-0.1553`

Top all feature movements:
- `lag_04__CT1__is_scoped`: contribution `-0.010817`
- `lag_14__CT4__duck_amount`: contribution `-0.007475`
- `lag_01__CT_place_ADMIN`: contribution `-0.007215`
- `lag_09__CT_place_MINI`: contribution `-0.007084`
- `lag_00__CT_shots_fired_sum`: contribution `-0.006834`

Top utility-only movements:
- `lag_02__T2__flash_duration`: contribution `-0.003984`
- `lag_06__CT1__flash_duration`: contribution `+0.002842`

### tick `49878`, seconds `43.50`, LSTM delta `-0.0991`

Top all feature movements:
- `lag_00__CT_place_GARAGE`: contribution `-0.011694`
- `lag_00__kill_diff_last_3s`: contribution `-0.006460`
- `lag_01__CT_place_MINI`: contribution `-0.005673`
- `lag_01__T5__duck_amount`: contribution `-0.005241`
- `lag_14__T4__is_scoped`: contribution `-0.005064`

Top utility-only movements:
- No utility movement among the top local contributors.
