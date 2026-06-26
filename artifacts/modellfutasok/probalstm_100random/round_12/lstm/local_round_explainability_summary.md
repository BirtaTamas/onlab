# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-legacy-vs-lynn-vision-bo3-80tf5tBYONxHYQuFp0AoSQ/legacy-vs-lynn-vision-m2-inferno.csv`
- round_num: `8`

## Largest probability jumps

- tick `58083`, seconds `41.50`, LSTM `0.7846`, delta `+0.2204`
- tick `61923`, seconds `101.50`, LSTM `0.9005`, delta `+0.1092`
- tick `57347`, seconds `30.00`, LSTM `0.5944`, delta `+0.0684`
- tick `62499`, seconds `110.50`, LSTM `0.9042`, delta `-0.0550`
- tick `61731`, seconds `98.50`, LSTM `0.7981`, delta `-0.0470`
- tick `57763`, seconds `36.50`, LSTM `0.5887`, delta `-0.0416`
- tick `58339`, seconds `45.50`, LSTM `0.8389`, delta `+0.0368`
- tick `59107`, seconds `57.50`, LSTM `0.8025`, delta `-0.0366`
- tick `61795`, seconds `99.50`, LSTM `0.7598`, delta `-0.0319`
- tick `59299`, seconds `60.50`, LSTM `0.8177`, delta `-0.0275`

## Top 15 local ridge features

- `lag_00__T3__flash_duration`: coefficient `0.001982`, |coef| `0.001982`
- `lag_00__CT_kills_last_3s`: coefficient `0.001337`, |coef| `0.001337`
- `lag_04__CT4__flash_duration`: coefficient `0.001304`, |coef| `0.001304`
- `lag_00__T2__duck_amount`: coefficient `-0.001063`, |coef| `0.001063`
- `lag_00__kill_diff_last_3s`: coefficient `0.001059`, |coef| `0.001059`
- `lag_02__CT_place_BALCONY`: coefficient `-0.000989`, |coef| `0.000989`
- `lag_00__CT_damage_last_5s`: coefficient `0.000985`, |coef| `0.000985`
- `lag_01__T3__flash_duration`: coefficient `0.000978`, |coef| `0.000978`
- `lag_04__CT5__duck_amount`: coefficient `-0.000952`, |coef| `0.000952`
- `lag_04__CT3__flash_duration`: coefficient `0.000948`, |coef| `0.000948`
- `lag_00__CT4__duck_amount`: coefficient `0.000930`, |coef| `0.000930`
- `lag_07__T5__flash_duration`: coefficient `0.000928`, |coef| `0.000928`
- `lag_05__CT5__duck_amount`: coefficient `-0.000928`, |coef| `0.000928`
- `lag_01__CT3__shots_fired`: coefficient `-0.000876`, |coef| `0.000876`
- `lag_00__CT1__flash_duration`: coefficient `-0.000868`, |coef| `0.000868`

## Top 10 utility ridge features

- `lag_00__T3__flash_duration`: coefficient `0.001982` (raises CT win probability)
- `lag_04__CT4__flash_duration`: coefficient `0.001304` (raises CT win probability)
- `lag_01__T3__flash_duration`: coefficient `0.000978` (raises CT win probability)
- `lag_04__CT3__flash_duration`: coefficient `0.000948` (raises CT win probability)
- `lag_07__T5__flash_duration`: coefficient `0.000928` (raises CT win probability)
- `lag_00__CT1__flash_duration`: coefficient `-0.000868` (lowers CT win probability)
- `lag_04__CT_flash_duration_sum`: coefficient `0.000846` (raises CT win probability)
- `lag_06__CT1__flash_duration`: coefficient `0.000829` (raises CT win probability)
- `lag_02__T3__flash_duration`: coefficient `0.000808` (raises CT win probability)
- `lag_15__T3__flash_duration`: coefficient `0.000789` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001337` (raises CT win probability)
- `lag_00__T2__duck_amount`: coefficient `-0.001063` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001059` (raises CT win probability)
- `lag_02__CT_place_BALCONY`: coefficient `-0.000989` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.000985` (raises CT win probability)
- `lag_04__CT5__duck_amount`: coefficient `-0.000952` (lowers CT win probability)
- `lag_00__CT4__duck_amount`: coefficient `0.000930` (raises CT win probability)
- `lag_05__CT5__duck_amount`: coefficient `-0.000928` (lowers CT win probability)
- `lag_01__CT3__shots_fired`: coefficient `-0.000876` (lowers CT win probability)
- `lag_03__CT_shots_fired_sum`: coefficient `0.000823` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `58083`, seconds `41.50`, LSTM delta `+0.2204`

Top all feature movements:
- `lag_00__T3__flash_duration`: contribution `+0.014184`
- `lag_04__CT4__flash_duration`: contribution `+0.009391`
- `lag_04__CT3__flash_duration`: contribution `+0.006655`
- `lag_02__CT_place_BALCONY`: contribution `+0.006347`
- `lag_01__CT3__shots_fired`: contribution `+0.005856`

Top utility-only movements:
- `lag_00__T3__flash_duration`: contribution `+0.014184`
- `lag_04__CT4__flash_duration`: contribution `+0.009391`
- `lag_04__CT3__flash_duration`: contribution `+0.006655`
- `lag_04__CT_flash_duration_sum`: contribution `+0.005378`
- `lag_07__T5__flash_duration`: contribution `+0.004822`

### tick `61923`, seconds `101.50`, LSTM delta `+0.1092`

Top all feature movements:
- `lag_06__CT1__flash_duration`: contribution `+0.005613`
- `lag_00__T4__flash_duration`: contribution `+0.004894`
- `lag_03__T5__flash_duration`: contribution `+0.004691`
- `lag_03__T_flash_duration_sum`: contribution `+0.003877`
- `lag_00__CT_kills_last_3s`: contribution `+0.003859`

Top utility-only movements:
- `lag_06__CT1__flash_duration`: contribution `+0.005613`
- `lag_00__T4__flash_duration`: contribution `+0.004894`
- `lag_03__T5__flash_duration`: contribution `+0.004691`
- `lag_03__T_flash_duration_sum`: contribution `+0.003877`
- `lag_03__T2__flash_duration`: contribution `+0.002550`

### tick `57347`, seconds `30.00`, LSTM delta `+0.0684`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.003859`
- `lag_00__kill_diff_last_3s`: contribution `+0.002549`
- `lag_15__CT5__duck_amount`: contribution `+0.002422`
- `lag_00__CT_damage_last_5s`: contribution `+0.002148`
- `lag_11__T4__is_walking`: contribution `+0.001860`

Top utility-only movements:
- `lag_07__CT_B_site_active_infernos`: contribution `+0.001568`
- `lag_00__T1__utility_total`: contribution `+0.001068`

### tick `62499`, seconds `110.50`, LSTM delta `-0.0550`

Top all feature movements:
- `lag_08__CT_shots_fired_sum`: contribution `-0.004482`
- `lag_08__T2__flash_duration`: contribution `-0.003894`
- `lag_15__T5__is_scoped`: contribution `-0.003288`
- `lag_08__CT3__shots_fired`: contribution `-0.002860`
- `lag_00__kill_diff_last_3s`: contribution `-0.002549`

Top utility-only movements:
- `lag_08__T2__flash_duration`: contribution `-0.003894`
- `lag_09__T5__flash_duration`: contribution `-0.002466`
- `lag_08__T_flash_duration_sum`: contribution `-0.001405`

### tick `61731`, seconds `98.50`, LSTM delta `-0.0470`

Top all feature movements:
- `lag_00__CT1__flash_duration`: contribution `-0.005877`
- `lag_02__CT3__flash_duration`: contribution `-0.002221`
- `lag_02__T_flashed_players`: contribution `-0.002066`
- `lag_06__CT_B_site_active_infernos`: contribution `-0.001723`
- `lag_02__CT5__duck_amount`: contribution `-0.001693`

Top utility-only movements:
- `lag_00__CT1__flash_duration`: contribution `-0.005877`
- `lag_02__CT3__flash_duration`: contribution `-0.002221`
- `lag_06__CT_B_site_active_infernos`: contribution `-0.001723`
- `lag_00__CT_flash_duration_sum`: contribution `-0.001680`
- `lag_02__T4__flash_duration`: contribution `-0.001616`
