# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-virtuspro-vs-og-inferno-UyQlNJx_rptvvsTtINI5j3/virtus-pro-vs-og-inferno.csv`
- round_num: `3`

## Largest probability jumps

- tick `24217`, seconds `103.50`, LSTM `0.7965`, delta `+0.2907`
- tick `24249`, seconds `104.00`, LSTM `0.5067`, delta `-0.2898`
- tick `24345`, seconds `105.50`, LSTM `0.7875`, delta `+0.2344`
- tick `24569`, seconds `109.00`, LSTM `0.9312`, delta `+0.1024`
- tick `24281`, seconds `104.50`, LSTM `0.5533`, delta `+0.0466`
- tick `24153`, seconds `102.50`, LSTM `0.4997`, delta `-0.0302`
- tick `18777`, seconds `18.50`, LSTM `0.5982`, delta `-0.0259`
- tick `24537`, seconds `108.50`, LSTM `0.8288`, delta `+0.0229`
- tick `18073`, seconds `7.50`, LSTM `0.6129`, delta `+0.0228`
- tick `18521`, seconds `14.50`, LSTM `0.6386`, delta `+0.0224`

## Top 15 local ridge features

- `lag_10__T_place_QUAD`: coefficient `-0.002733`, |coef| `0.002733`
- `lag_05__CT2__flash_duration`: coefficient `-0.001247`, |coef| `0.001247`
- `lag_09__T_place_QUAD`: coefficient `0.001226`, |coef| `0.001226`
- `lag_02__T_shots_fired_sum`: coefficient `0.001206`, |coef| `0.001206`
- `lag_13__T_place_QUAD`: coefficient `0.001182`, |coef| `0.001182`
- `lag_08__CT4__flash_duration`: coefficient `0.001050`, |coef| `0.001050`
- `lag_03__CT3__duck_amount`: coefficient `-0.001028`, |coef| `0.001028`
- `lag_04__T_place_ARCH`: coefficient `-0.001020`, |coef| `0.001020`
- `lag_02__T1__shots_fired`: coefficient `0.000962`, |coef| `0.000962`
- `lag_05__T_place_BALCONY`: coefficient `-0.000919`, |coef| `0.000919`
- `lag_05__CT_flash_duration_sum`: coefficient `-0.000914`, |coef| `0.000914`
- `lag_07__T_place_ARCH`: coefficient `0.000910`, |coef| `0.000910`
- `lag_05__CT_flashed_players`: coefficient `-0.000902`, |coef| `0.000902`
- `lag_01__CT3__duck_amount`: coefficient `-0.000860`, |coef| `0.000860`
- `lag_04__CT_flashed_players`: coefficient `0.000858`, |coef| `0.000858`

## Top 10 utility ridge features

- `lag_05__CT2__flash_duration`: coefficient `-0.001247` (lowers CT win probability)
- `lag_08__CT4__flash_duration`: coefficient `0.001050` (raises CT win probability)
- `lag_05__CT_flash_duration_sum`: coefficient `-0.000914` (lowers CT win probability)
- `lag_01__CT2__flash_duration`: coefficient `-0.000857` (lowers CT win probability)
- `lag_08__CT_flash_duration_sum`: coefficient `0.000849` (raises CT win probability)
- `lag_02__CT2__flash_duration`: coefficient `0.000848` (raises CT win probability)
- `lag_04__CT2__flash_duration`: coefficient `0.000775` (raises CT win probability)
- `lag_08__CT2__flash_duration`: coefficient `0.000760` (raises CT win probability)
- `lag_04__CT_flash_duration_sum`: coefficient `0.000734` (raises CT win probability)
- `lag_04__CT4__flash_duration`: coefficient `0.000731` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_10__T_place_QUAD`: coefficient `-0.002733` (lowers CT win probability)
- `lag_09__T_place_QUAD`: coefficient `0.001226` (raises CT win probability)
- `lag_02__T_shots_fired_sum`: coefficient `0.001206` (raises CT win probability)
- `lag_13__T_place_QUAD`: coefficient `0.001182` (raises CT win probability)
- `lag_03__CT3__duck_amount`: coefficient `-0.001028` (lowers CT win probability)
- `lag_04__T_place_ARCH`: coefficient `-0.001020` (lowers CT win probability)
- `lag_02__T1__shots_fired`: coefficient `0.000962` (raises CT win probability)
- `lag_05__T_place_BALCONY`: coefficient `-0.000919` (lowers CT win probability)
- `lag_07__T_place_ARCH`: coefficient `0.000910` (raises CT win probability)
- `lag_05__CT_flashed_players`: coefficient `-0.000902` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `24217`, seconds `103.50`, LSTM delta `+0.2907`

Top all feature movements:
- `lag_10__T_place_QUAD`: contribution `+0.065829`
- `lag_09__T_place_QUAD`: contribution `+0.029529`
- `lag_07__T_place_QUAD`: contribution `+0.018631`
- `lag_01__T_shots_fired_sum`: contribution `+0.013186`
- `lag_02__T_shots_fired_sum`: contribution `+0.011752`

Top utility-only movements:
- `lag_01__CT2__flash_duration`: contribution `+0.005737`
- `lag_04__CT4__flash_duration`: contribution `+0.005584`
- `lag_04__CT2__flash_duration`: contribution `+0.005185`
- `lag_04__CT_flash_duration_sum`: contribution `+0.004975`
- `lag_08__CT_utility_damage_last_5s`: contribution `+0.002819`

### tick `24249`, seconds `104.00`, LSTM delta `-0.2898`

Top all feature movements:
- `lag_10__T_place_QUAD`: contribution `-0.065829`
- `lag_02__T_shots_fired_sum`: contribution `-0.020793`
- `lag_15__T_place_QUAD`: contribution `-0.016983`
- `lag_08__T_place_QUAD`: contribution `-0.015883`
- `lag_05__T_place_BALCONY`: contribution `-0.012632`

Top utility-only movements:
- `lag_05__CT2__flash_duration`: contribution `-0.008348`
- `lag_05__CT_flash_duration_sum`: contribution `-0.006199`
- `lag_02__CT2__flash_duration`: contribution `-0.005676`
- `lag_05__CT4__flash_duration`: contribution `-0.005133`
- `lag_09__CT_utility_damage_last_5s`: contribution `-0.003558`

### tick `24345`, seconds `105.50`, LSTM delta `+0.2344`

Top all feature movements:
- `lag_13__T_place_QUAD`: contribution `+0.028470`
- `lag_05__T_shots_fired_sum`: contribution `+0.014336`
- `lag_08__T_place_BALCONY`: contribution `+0.010857`
- `lag_06__T_place_BALCONY`: contribution `+0.009581`
- `lag_07__T_place_ARCH`: contribution `+0.008468`

Top utility-only movements:
- `lag_05__CT2__flash_duration`: contribution `+0.008348`
- `lag_08__CT4__flash_duration`: contribution `+0.008028`
- `lag_08__CT_flash_duration_sum`: contribution `+0.005758`
- `lag_08__CT2__flash_duration`: contribution `+0.005091`

### tick `24569`, seconds `109.00`, LSTM delta `+0.1024`

Top all feature movements:
- `lag_04__T_place_ARCH`: contribution `+0.009489`
- `lag_12__T_shots_fired_sum`: contribution `+0.005885`
- `lag_13__T_place_BALCONY`: contribution `+0.005837`
- `lag_14__T_place_ARCH`: contribution `+0.005362`
- `lag_13__T_shots_fired_sum`: contribution `+0.004221`

Top utility-only movements:
- `lag_09__CT_utility_damage_last_5s`: contribution `+0.003558`
- `lag_15__CT4__flash_duration`: contribution `+0.003544`
- `lag_09__utility_damage_diff_last_5s`: contribution `+0.002376`
- `lag_15__CT_flash_duration_sum`: contribution `+0.001824`

### tick `24281`, seconds `104.50`, LSTM delta `+0.0466`

Top all feature movements:
- `lag_09__T_place_QUAD`: contribution `-0.029529`
- `lag_06__T_place_BALCONY`: contribution `-0.009581`
- `lag_04__T_place_BALCONY`: contribution `-0.009147`
- `lag_12__T_place_QUAD`: contribution `+0.007259`
- `lag_05__T_shots_fired_sum`: contribution `-0.006233`

Top utility-only movements:
- `lag_03__CT2__flash_duration`: contribution `+0.003498`
- `lag_06__CT4__flash_duration`: contribution `+0.002956`
- `lag_06__CT2__flash_duration`: contribution `+0.002727`
- `lag_06__CT_flash_duration_sum`: contribution `+0.002533`
