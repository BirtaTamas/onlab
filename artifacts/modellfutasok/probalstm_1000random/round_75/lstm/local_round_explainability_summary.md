# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-the-mongolz-vs-m80-bo3-e7FibL-GpwhFRhM0kGS5r4/the-mongolz-vs-m80-m3-inferno.csv`
- round_num: `5`

## Largest probability jumps

- tick `25584`, seconds `42.00`, LSTM `0.8307`, delta `+0.1660`
- tick `25424`, seconds `39.50`, LSTM `0.7407`, delta `-0.1202`
- tick `25392`, seconds `39.00`, LSTM `0.8609`, delta `+0.1071`
- tick `25456`, seconds `40.00`, LSTM `0.6806`, delta `-0.0601`
- tick `25776`, seconds `45.00`, LSTM `0.9372`, delta `+0.0534`
- tick `24592`, seconds `26.50`, LSTM `0.7149`, delta `-0.0445`
- tick `23280`, seconds `6.00`, LSTM `0.7986`, delta `+0.0294`
- tick `25552`, seconds `41.50`, LSTM `0.6647`, delta `-0.0285`
- tick `25168`, seconds `35.50`, LSTM `0.7246`, delta `-0.0254`
- tick `25072`, seconds `34.00`, LSTM `0.7596`, delta `+0.0243`

## Top 15 local ridge features

- `lag_00__CT_place_BALCONY`: coefficient `-0.001213`, |coef| `0.001213`
- `lag_08__CT_place_BALCONY`: coefficient `-0.001163`, |coef| `0.001163`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001064`, |coef| `0.001064`
- `lag_03__CT_place_QUAD`: coefficient `0.001026`, |coef| `0.001026`
- `lag_02__CT_place_BALCONY`: coefficient `-0.000989`, |coef| `0.000989`
- `lag_04__CT_place_BALCONY`: coefficient `0.000926`, |coef| `0.000926`
- `lag_06__CT1__is_scoped`: coefficient `-0.000783`, |coef| `0.000783`
- `lag_14__CT_place_LIBRARY`: coefficient `-0.000752`, |coef| `0.000752`
- `lag_11__T1__duck_amount`: coefficient `0.000751`, |coef| `0.000751`
- `lag_15__CT2__duck_amount`: coefficient `-0.000750`, |coef| `0.000750`
- `lag_01__T_flashed_players`: coefficient `0.000744`, |coef| `0.000744`
- `lag_04__CT_place_APARTMENTS`: coefficient `-0.000733`, |coef| `0.000733`
- `lag_10__CT2__duck_amount`: coefficient `0.000712`, |coef| `0.000712`
- `lag_07__T_flashed_players`: coefficient `0.000700`, |coef| `0.000700`
- `lag_00__damage_diff_last_5s`: coefficient `0.000696`, |coef| `0.000696`

## Top 10 utility ridge features

- `lag_03__T_utility_damage_last_5s`: coefficient `-0.000693` (lowers CT win probability)
- `lag_08__T_utility_damage_last_5s`: coefficient `0.000638` (raises CT win probability)
- `lag_03__utility_damage_diff_last_5s`: coefficient `0.000577` (raises CT win probability)
- `lag_04__T_utility_damage_last_5s`: coefficient `-0.000536` (lowers CT win probability)
- `lag_03__T5__flash_duration`: coefficient `-0.000534` (lowers CT win probability)
- `lag_02__CT_B_site_active_infernos`: coefficient `0.000487` (raises CT win probability)
- `lag_04__utility_damage_diff_last_5s`: coefficient `0.000429` (raises CT win probability)
- `lag_02__T_utility_damage_last_5s`: coefficient `0.000419` (raises CT win probability)
- `lag_00__T_flashes_last_5s`: coefficient `-0.000417` (lowers CT win probability)
- `lag_00__T1__smoke`: coefficient `-0.000402` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_BALCONY`: coefficient `-0.001213` (lowers CT win probability)
- `lag_08__CT_place_BALCONY`: coefficient `-0.001163` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001064` (raises CT win probability)
- `lag_03__CT_place_QUAD`: coefficient `0.001026` (raises CT win probability)
- `lag_02__CT_place_BALCONY`: coefficient `-0.000989` (lowers CT win probability)
- `lag_04__CT_place_BALCONY`: coefficient `0.000926` (raises CT win probability)
- `lag_06__CT1__is_scoped`: coefficient `-0.000783` (lowers CT win probability)
- `lag_14__CT_place_LIBRARY`: coefficient `-0.000752` (lowers CT win probability)
- `lag_11__T1__duck_amount`: coefficient `0.000751` (raises CT win probability)
- `lag_15__CT2__duck_amount`: coefficient `-0.000750` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `25584`, seconds `42.00`, LSTM delta `+0.1660`

Top all feature movements:
- `lag_03__CT_place_QUAD`: contribution `+0.008083`
- `lag_08__CT_place_BALCONY`: contribution `+0.007466`
- `lag_02__CT_place_BALCONY`: contribution `+0.006349`
- `lag_04__CT_place_BALCONY`: contribution `+0.005943`
- `lag_14__CT_place_LIBRARY`: contribution `+0.004825`

Top utility-only movements:
- `lag_08__T_utility_damage_last_5s`: contribution `+0.003278`
- `lag_02__CT_B_site_active_infernos`: contribution `+0.001674`

### tick `25424`, seconds `39.50`, LSTM delta `-0.1202`

Top all feature movements:
- `lag_08__CT_place_BALCONY`: contribution `-0.007466`
- `lag_00__CT_shots_fired_sum`: contribution `-0.005174`
- `lag_03__T_utility_damage_last_5s`: contribution `-0.003562`
- `lag_06__CT1__is_scoped`: contribution `-0.003354`
- `lag_09__CT_place_LIBRARY`: contribution `-0.003278`

Top utility-only movements:
- `lag_03__T_utility_damage_last_5s`: contribution `-0.003562`
- `lag_03__utility_damage_diff_last_5s`: contribution `-0.001877`

### tick `25392`, seconds `39.00`, LSTM delta `+0.1071`

Top all feature movements:
- `lag_02__CT_place_BALCONY`: contribution `+0.006349`
- `lag_01__T_flashed_players`: contribution `+0.004310`
- `lag_00__CT_shots_fired_sum`: contribution `+0.003696`
- `lag_10__CT2__duck_amount`: contribution `+0.002711`
- `lag_14__CT2__duck_amount`: contribution `+0.002293`

Top utility-only movements:
- `lag_02__T_utility_damage_last_5s`: contribution `+0.002153`

### tick `25456`, seconds `40.00`, LSTM delta `-0.0601`

Top all feature movements:
- `lag_00__CT_place_BALCONY`: contribution `-0.007785`
- `lag_04__CT_place_BALCONY`: contribution `-0.005943`
- `lag_03__T_flashed_players`: contribution `-0.003813`
- `lag_09__CT_place_BALCONY`: contribution `-0.003451`
- `lag_04__CT_place_APARTMENTS`: contribution `-0.002816`

Top utility-only movements:
- `lag_04__T_utility_damage_last_5s`: contribution `-0.002755`
- `lag_04__utility_damage_diff_last_5s`: contribution `-0.001394`
- `lag_03__T5__flash_duration`: contribution `-0.001335`

### tick `25776`, seconds `45.00`, LSTM delta `+0.0534`

Top all feature movements:
- `lag_08__CT_place_BALCONY`: contribution `+0.007466`
- `lag_04__T_utility_damage_last_5s`: contribution `+0.002755`
- `lag_09__CT_place_QUAD`: contribution `+0.002745`
- `lag_05__CT_shots_fired_sum`: contribution `+0.001995`
- `lag_02__CT_place_QUAD`: contribution `+0.001891`

Top utility-only movements:
- `lag_04__T_utility_damage_last_5s`: contribution `+0.002755`
- `lag_04__utility_damage_diff_last_5s`: contribution `+0.001704`
- `lag_14__T_utility_damage_last_5s`: contribution `+0.001042`
- `lag_03__utility_damage_diff_last_5s`: contribution `+0.000834`
