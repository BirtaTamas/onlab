# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-gamerlegion-vs-complexity-bo3-A8nOd44IyEYHGVOxrkExMv/gamerlegion-vs-complexity-m1-inferno.csv`
- round_num: `9`

## Largest probability jumps

- tick `67647`, seconds `17.00`, LSTM `0.7416`, delta `+0.1374`
- tick `69567`, seconds `47.00`, LSTM `0.9367`, delta `+0.1007`
- tick `66591`, seconds `0.50`, LSTM `0.5720`, delta `+0.0420`
- tick `68127`, seconds `24.50`, LSTM `0.8155`, delta `+0.0372`
- tick `68095`, seconds `24.00`, LSTM `0.7783`, delta `+0.0349`
- tick `69215`, seconds `41.50`, LSTM `0.8250`, delta `+0.0309`
- tick `67615`, seconds `16.50`, LSTM `0.6042`, delta `+0.0269`
- tick `69471`, seconds `45.50`, LSTM `0.8400`, delta `+0.0251`
- tick `66943`, seconds `6.00`, LSTM `0.5826`, delta `-0.0244`
- tick `67775`, seconds `19.00`, LSTM `0.7476`, delta `+0.0227`

## Top 15 local ridge features

- `lag_00__T_place_ARCH`: coefficient `0.001783`, |coef| `0.001783`
- `lag_11__CT_place_BALCONY`: coefficient `-0.001057`, |coef| `0.001057`
- `lag_11__CT_place_PIT`: coefficient `0.001049`, |coef| `0.001049`
- `lag_13__CT_place_BALCONY`: coefficient `0.001015`, |coef| `0.001015`
- `lag_00__CT_kills_last_3s`: coefficient `0.000962`, |coef| `0.000962`
- `lag_00__CT_damage_last_5s`: coefficient `0.000928`, |coef| `0.000928`
- `lag_00__CT3__is_walking`: coefficient `-0.000861`, |coef| `0.000861`
- `lag_00__kill_diff_last_3s`: coefficient `0.000855`, |coef| `0.000855`
- `lag_09__CT_place_PIT`: coefficient `-0.000849`, |coef| `0.000849`
- `lag_14__CT1__flash_duration`: coefficient `0.000846`, |coef| `0.000846`
- `lag_00__T2__alive`: coefficient `-0.000836`, |coef| `0.000836`
- `lag_00__T2__hp`: coefficient `-0.000826`, |coef| `0.000826`
- `lag_03__CT5__duck_amount`: coefficient `0.000815`, |coef| `0.000815`
- `lag_00__T2__duck_amount`: coefficient `-0.000787`, |coef| `0.000787`
- `lag_00__damage_diff_last_5s`: coefficient `0.000774`, |coef| `0.000774`

## Top 10 utility ridge features

- `lag_14__CT1__flash_duration`: coefficient `0.000846` (raises CT win probability)
- `lag_00__CT_mollies_last_5s`: coefficient `0.000660` (raises CT win probability)
- `lag_00__T5__flash`: coefficient `-0.000636` (lowers CT win probability)
- `lag_00__T5__utility_total`: coefficient `-0.000606` (lowers CT win probability)
- `lag_01__CT1__flash_duration`: coefficient `-0.000543` (lowers CT win probability)
- `lag_00__CT1__flash_duration`: coefficient `0.000540` (raises CT win probability)
- `lag_10__CT1__flash_duration`: coefficient `0.000535` (raises CT win probability)
- `lag_00__T2__flash`: coefficient `-0.000513` (lowers CT win probability)
- `lag_12__CT1__flash_duration`: coefficient `-0.000512` (lowers CT win probability)
- `lag_03__CT_B_site_active_smokes`: coefficient `-0.000505` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_ARCH`: coefficient `0.001783` (raises CT win probability)
- `lag_11__CT_place_BALCONY`: coefficient `-0.001057` (lowers CT win probability)
- `lag_11__CT_place_PIT`: coefficient `0.001049` (raises CT win probability)
- `lag_13__CT_place_BALCONY`: coefficient `0.001015` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000962` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.000928` (raises CT win probability)
- `lag_00__CT3__is_walking`: coefficient `-0.000861` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000855` (raises CT win probability)
- `lag_09__CT_place_PIT`: coefficient `-0.000849` (lowers CT win probability)
- `lag_00__T2__alive`: coefficient `-0.000836` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `67647`, seconds `17.00`, LSTM delta `+0.1374`

Top all feature movements:
- `lag_13__CT_place_BALCONY`: contribution `+0.006514`
- `lag_10__CT_place_BALCONY`: contribution `+0.004488`
- `lag_00__CT1__flash_duration`: contribution `+0.004163`
- `lag_12__CT1__is_scoped`: contribution `+0.002886`
- `lag_10__CT_place_APARTMENTS`: contribution `+0.002851`

Top utility-only movements:
- `lag_00__CT1__flash_duration`: contribution `+0.004163`
- `lag_00__T5__flash`: contribution `+0.001806`
- `lag_03__CT_A_site_active_infernos`: contribution `+0.001610`

### tick `69567`, seconds `47.00`, LSTM delta `+0.1007`

Top all feature movements:
- `lag_00__T_place_ARCH`: contribution `+0.016591`
- `lag_11__CT_place_BALCONY`: contribution `+0.006786`
- `lag_11__CT_place_PIT`: contribution `+0.004516`
- `lag_09__CT_place_PIT`: contribution `+0.003655`
- `lag_03__CT5__duck_amount`: contribution `+0.003077`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `66591`, seconds `0.50`, LSTM delta `+0.0420`

Top all feature movements:
- `lag_00__CT_mollies_last_5s`: contribution `+0.021876`
- `lag_00__CT_smokes_last_5s`: contribution `+0.005947`
- `lag_00__CT_flashes_last_5s`: contribution `+0.002406`
- `lag_00__CT_velocity_mean`: contribution `+0.000777`
- `lag_01__T1__has_bomb`: contribution `+0.000712`

Top utility-only movements:
- `lag_00__CT_mollies_last_5s`: contribution `+0.021876`
- `lag_00__CT_smokes_last_5s`: contribution `+0.005947`
- `lag_00__CT_flashes_last_5s`: contribution `+0.002406`
- `lag_01__CT1__utility_total`: contribution `+0.000506`
- `lag_01__CT1__flash`: contribution `+0.000470`

### tick `68127`, seconds `24.50`, LSTM delta `+0.0372`

Top all feature movements:
- `lag_01__CT1__flash_duration`: contribution `+0.004187`
- `lag_15__CT1__flash_duration`: contribution `+0.002189`
- `lag_00__CT3__is_walking`: contribution `+0.002054`
- `lag_09__CT1__is_scoped`: contribution `+0.001927`
- `lag_00__CT1__duck_amount`: contribution `+0.001898`

Top utility-only movements:
- `lag_01__CT1__flash_duration`: contribution `+0.004187`
- `lag_15__CT1__flash_duration`: contribution `+0.002189`

### tick `68095`, seconds `24.00`, LSTM delta `+0.0349`

Top all feature movements:
- `lag_14__CT1__flash_duration`: contribution `+0.006519`
- `lag_00__CT1__flash_duration`: contribution `-0.004163`
- `lag_10__CT1__duck_amount`: contribution `-0.002774`
- `lag_08__CT1__is_scoped`: contribution `+0.002098`
- `lag_00__CT3__is_walking`: contribution `-0.002054`

Top utility-only movements:
- `lag_14__CT1__flash_duration`: contribution `+0.006519`
- `lag_00__CT1__flash_duration`: contribution `-0.004163`
- `lag_14__CT_flash_duration_sum`: contribution `+0.001255`
- `lag_14__CT_utility_damage_last_5s`: contribution `+0.001222`
