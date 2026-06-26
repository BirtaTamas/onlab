# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-legacy-vs-lynn-vision-bo3-80tf5tBYONxHYQuFp0AoSQ/legacy-vs-lynn-vision-m1-dust2.csv`
- round_num: `8`

## Largest probability jumps

- tick `68621`, seconds `83.00`, LSTM `0.0335`, delta `-0.0767`
- tick `68557`, seconds `82.00`, LSTM `0.1015`, delta `+0.0579`
- tick `63341`, seconds `0.50`, LSTM `0.1120`, delta `-0.0512`
- tick `64237`, seconds `14.50`, LSTM `0.1301`, delta `-0.0484`
- tick `64077`, seconds `12.00`, LSTM `0.1502`, delta `-0.0442`
- tick `64557`, seconds `19.50`, LSTM `0.1725`, delta `+0.0332`
- tick `67565`, seconds `66.50`, LSTM `0.1362`, delta `-0.0313`
- tick `63789`, seconds `7.50`, LSTM `0.1372`, delta `+0.0305`
- tick `66285`, seconds `46.50`, LSTM `0.1374`, delta `-0.0272`
- tick `64845`, seconds `24.00`, LSTM `0.1578`, delta `-0.0256`

## Top 15 local ridge features

- `lag_00__CT2__is_scoped`: coefficient `0.001532`, |coef| `0.001532`
- `lag_00__T5__is_scoped`: coefficient `0.000953`, |coef| `0.000953`
- `lag_00__CT_he_last_5s`: coefficient `-0.000931`, |coef| `0.000931`
- `lag_00__T_velocity_mean`: coefficient `-0.000654`, |coef| `0.000654`
- `lag_11__CT_he_last_5s`: coefficient `-0.000594`, |coef| `0.000594`
- `lag_00__CT_scoped_count`: coefficient `0.000581`, |coef| `0.000581`
- `lag_00__CT_velocity_mean`: coefficient `-0.000577`, |coef| `0.000577`
- `lag_00__T_scoped_count`: coefficient `0.000576`, |coef| `0.000576`
- `lag_08__CT2__duck_amount`: coefficient `-0.000573`, |coef| `0.000573`
- `lag_00__CT2__duck_amount`: coefficient `-0.000569`, |coef| `0.000569`
- `lag_04__T_place_SHORTSTAIRS`: coefficient `-0.000559`, |coef| `0.000559`
- `lag_00__T2__duck_amount`: coefficient `0.000541`, |coef| `0.000541`
- `lag_01__T_place_EXTENDEDA`: coefficient `-0.000521`, |coef| `0.000521`
- `lag_10__CT_he_last_5s`: coefficient `-0.000502`, |coef| `0.000502`
- `lag_04__CT_flashed_players`: coefficient `0.000463`, |coef| `0.000463`

## Top 10 utility ridge features

- `lag_00__CT_he_last_5s`: coefficient `-0.000931` (lowers CT win probability)
- `lag_11__CT_he_last_5s`: coefficient `-0.000594` (lowers CT win probability)
- `lag_10__CT_he_last_5s`: coefficient `-0.000502` (lowers CT win probability)
- `lag_01__CT_he_last_5s`: coefficient `-0.000376` (lowers CT win probability)
- `lag_15__CT_he_last_5s`: coefficient `-0.000355` (lowers CT win probability)
- `lag_04__CT1__flash_duration`: coefficient `0.000347` (raises CT win probability)
- `lag_04__CT3__flash_duration`: coefficient `0.000317` (raises CT win probability)
- `lag_06__CT1__flash_duration`: coefficient `-0.000315` (lowers CT win probability)
- `lag_13__CT_he_last_5s`: coefficient `0.000308` (raises CT win probability)
- `lag_06__CT_flash_duration_sum`: coefficient `-0.000291` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT2__is_scoped`: coefficient `0.001532` (raises CT win probability)
- `lag_00__T5__is_scoped`: coefficient `0.000953` (raises CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000654` (lowers CT win probability)
- `lag_00__CT_scoped_count`: coefficient `0.000581` (raises CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.000577` (lowers CT win probability)
- `lag_00__T_scoped_count`: coefficient `0.000576` (raises CT win probability)
- `lag_08__CT2__duck_amount`: coefficient `-0.000573` (lowers CT win probability)
- `lag_00__CT2__duck_amount`: coefficient `-0.000569` (lowers CT win probability)
- `lag_04__T_place_SHORTSTAIRS`: coefficient `-0.000559` (lowers CT win probability)
- `lag_00__T2__duck_amount`: coefficient `0.000541` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `68621`, seconds `83.00`, LSTM delta `-0.0767`

Top all feature movements:
- `lag_00__T5__is_scoped`: contribution `-0.004545`
- `lag_06__CT_flashed_players`: contribution `-0.002780`
- `lag_01__T_place_EXTENDEDA`: contribution `-0.002581`
- `lag_04__T_place_SHORTSTAIRS`: contribution `-0.002347`
- `lag_00__CT2__duck_amount`: contribution `-0.002167`

Top utility-only movements:
- `lag_06__CT1__flash_duration`: contribution `-0.002071`
- `lag_06__CT_flash_duration_sum`: contribution `-0.001284`

### tick `68557`, seconds `82.00`, LSTM delta `+0.0579`

Top all feature movements:
- `lag_04__CT_flashed_players`: contribution `+0.003042`
- `lag_10__CT_place_EXTENDEDA`: contribution `+0.002376`
- `lag_04__CT1__flash_duration`: contribution `+0.002282`
- `lag_00__CT2__duck_amount`: contribution `+0.002167`
- `lag_10__T5__is_scoped`: contribution `+0.002140`

Top utility-only movements:
- `lag_04__CT1__flash_duration`: contribution `+0.002282`
- `lag_02__CT5__flash_duration`: contribution `+0.001196`
- `lag_04__CT_flash_duration_sum`: contribution `+0.001033`

### tick `63341`, seconds `0.50`, LSTM delta `-0.0512`

Top all feature movements:
- `lag_00__CT_he_last_5s`: contribution `-0.017090`
- `lag_00__T_velocity_mean`: contribution `-0.002245`
- `lag_00__CT_velocity_mean`: contribution `-0.001977`
- `lag_01__CT_place_CTSPAWN`: contribution `-0.001572`
- `lag_01__T_place_TSPAWN`: contribution `-0.001482`

Top utility-only movements:
- `lag_00__CT_he_last_5s`: contribution `-0.017090`
- `lag_01__T2__utility_total`: contribution `-0.000352`
- `lag_01__T_smoke_inv`: contribution `-0.000332`
- `lag_01__T2__flash`: contribution `-0.000316`
- `lag_01__utility_inv_diff`: contribution `-0.000289`

### tick `64237`, seconds `14.50`, LSTM delta `-0.0484`

Top all feature movements:
- `lag_00__CT2__is_scoped`: contribution `-0.009377`
- `lag_00__CT_place_ARAMP`: contribution `-0.002749`
- `lag_12__CT_flashed_players`: contribution `-0.002028`
- `lag_04__CT3__flash_duration`: contribution `-0.001628`
- `lag_04__T3__duck_amount`: contribution `-0.001599`

Top utility-only movements:
- `lag_04__CT3__flash_duration`: contribution `-0.001628`
- `lag_12__CT3__flash_duration`: contribution `-0.001131`
- `lag_12__CT_flash_duration_sum`: contribution `-0.000767`

### tick `64077`, seconds `12.00`, LSTM delta `-0.0442`

Top all feature movements:
- `lag_00__CT2__is_scoped`: contribution `-0.009377`
- `lag_13__CT_he_last_5s`: contribution `-0.005643`
- `lag_00__T5__is_scoped`: contribution `-0.004545`
- `lag_00__T2__duck_amount`: contribution `-0.002067`
- `lag_00__T_scoped_count`: contribution `-0.001658`

Top utility-only movements:
- `lag_13__CT_he_last_5s`: contribution `-0.005643`
- `lag_07__CT3__flash_duration`: contribution `-0.000623`
