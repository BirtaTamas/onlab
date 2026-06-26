# Local Round Explainability

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-pain-vs-rare-atom-bo3-Rmb0_mvtIpTmOfUIJVjwOw/pain-vs-rare-atom-m1-inferno.csv`
- round_num: `12`

## Largest probability jumps

- tick `84698`, seconds `27.50`, LSTM `0.1155`, delta `-0.2276`
- tick `83642`, seconds `11.00`, LSTM `0.4473`, delta `-0.2036`
- tick `84442`, seconds `23.50`, LSTM `0.3740`, delta `-0.1771`
- tick `84666`, seconds `27.00`, LSTM `0.3431`, delta `+0.1174`
- tick `84474`, seconds `24.00`, LSTM `0.3009`, delta `-0.0731`
- tick `84794`, seconds `29.00`, LSTM `0.1116`, delta `+0.0661`
- tick `83674`, seconds `11.50`, LSTM `0.3840`, delta `-0.0633`
- tick `84730`, seconds `28.00`, LSTM `0.0527`, delta `-0.0628`
- tick `87002`, seconds `63.50`, LSTM `0.0243`, delta `-0.0549`
- tick `84506`, seconds `24.50`, LSTM `0.2466`, delta `-0.0542`

## Top 15 local ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.002289`, |coef| `0.002289`
- `lag_00__kill_diff_last_3s`: coefficient `0.001894`, |coef| `0.001894`
- `lag_09__CT3__flash_duration`: coefficient `-0.001763`, |coef| `0.001763`
- `lag_04__CT_place_ARCH`: coefficient `-0.001612`, |coef| `0.001612`
- `lag_00__T_place_BALCONY`: coefficient `-0.001602`, |coef| `0.001602`
- `lag_09__T_mollies_last_5s`: coefficient `0.001597`, |coef| `0.001597`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001465`, |coef| `0.001465`
- `lag_13__T_place_LOWERMID`: coefficient `-0.001412`, |coef| `0.001412`
- `lag_01__T_kills_last_3s`: coefficient `-0.001342`, |coef| `0.001342`
- `lag_00__damage_diff_last_5s`: coefficient `0.001275`, |coef| `0.001275`
- `lag_11__CT_place_LIBRARY`: coefficient `-0.001273`, |coef| `0.001273`
- `lag_07__T5__flash_duration`: coefficient `0.001186`, |coef| `0.001186`
- `lag_00__T_damage_last_5s`: coefficient `-0.001172`, |coef| `0.001172`
- `lag_01__T_damage_last_5s`: coefficient `-0.001158`, |coef| `0.001158`
- `lag_02__CT3__is_walking`: coefficient `0.001134`, |coef| `0.001134`

## Top 10 utility ridge features

- `lag_09__CT3__flash_duration`: coefficient `-0.001763` (lowers CT win probability)
- `lag_09__T_mollies_last_5s`: coefficient `0.001597` (raises CT win probability)
- `lag_07__T5__flash_duration`: coefficient `0.001186` (raises CT win probability)
- `lag_06__CT3__flash_duration`: coefficient `0.001103` (raises CT win probability)
- `lag_09__CT1__flash`: coefficient `-0.000991` (lowers CT win probability)
- `lag_10__CT3__flash_duration`: coefficient `-0.000962` (lowers CT win probability)
- `lag_04__CT1__flash_duration`: coefficient `-0.000891` (lowers CT win probability)
- `lag_08__T5__flash_duration`: coefficient `0.000890` (raises CT win probability)
- `lag_10__active_infernos_total`: coefficient `0.000883` (raises CT win probability)
- `lag_14__CT_A_site_active_infernos`: coefficient `-0.000874` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.002289` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001894` (raises CT win probability)
- `lag_04__CT_place_ARCH`: coefficient `-0.001612` (lowers CT win probability)
- `lag_00__T_place_BALCONY`: coefficient `-0.001602` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001465` (raises CT win probability)
- `lag_13__T_place_LOWERMID`: coefficient `-0.001412` (lowers CT win probability)
- `lag_01__T_kills_last_3s`: coefficient `-0.001342` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001275` (raises CT win probability)
- `lag_11__CT_place_LIBRARY`: coefficient `-0.001273` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001172` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `84698`, seconds `27.50`, LSTM delta `-0.2276`

Top all feature movements:
- `lag_11__CT_place_LIBRARY`: contribution `-0.008165`
- `lag_00__T_kills_last_3s`: contribution `-0.007251`
- `lag_00__CT_shots_fired_sum`: contribution `-0.007124`
- `lag_04__CT_place_ARCH`: contribution `-0.006578`
- `lag_06__CT3__flash_duration`: contribution `-0.006478`

Top utility-only movements:
- `lag_06__CT3__flash_duration`: contribution `-0.006478`
- `lag_15__T5__flash_duration`: contribution `-0.003717`

### tick `83642`, seconds `11.00`, LSTM delta `-0.2036`

Top all feature movements:
- `lag_09__T_mollies_last_5s`: contribution `-0.032828`
- `lag_13__T_place_LOWERMID`: contribution `-0.009394`
- `lag_00__T_kills_last_3s`: contribution `-0.007251`
- `lag_09__T_place_LOWERMID`: contribution `-0.007103`
- `lag_08__T_place_LOWERMID`: contribution `-0.006794`

Top utility-only movements:
- `lag_09__T_mollies_last_5s`: contribution `-0.032828`
- `lag_03__T_A_site_active_infernos`: contribution `-0.004817`
- `lag_03__T_active_infernos`: contribution `-0.002390`
- `lag_00__CT4__utility_total`: contribution `-0.001720`

### tick `84442`, seconds `23.50`, LSTM delta `-0.1771`

Top all feature movements:
- `lag_09__CT3__flash_duration`: contribution `-0.010352`
- `lag_00__T_kills_last_3s`: contribution `-0.007251`
- `lag_07__T5__flash_duration`: contribution `-0.006289`
- `lag_00__kill_diff_last_3s`: contribution `-0.004558`
- `lag_09__CT1__flash`: contribution `-0.003546`

Top utility-only movements:
- `lag_09__CT3__flash_duration`: contribution `-0.010352`
- `lag_07__T5__flash_duration`: contribution `-0.006289`
- `lag_09__CT1__flash`: contribution `-0.003546`
- `lag_14__CT_A_site_active_infernos`: contribution `-0.003085`
- `lag_10__CT_B_site_active_infernos`: contribution `-0.002999`

### tick `84666`, seconds `27.00`, LSTM delta `+0.1174`

Top all feature movements:
- `lag_02__CT_place_LIBRARY`: contribution `+0.006411`
- `lag_00__CT_shots_fired_sum`: contribution `+0.005089`
- `lag_00__kill_diff_last_3s`: contribution `+0.004558`
- `lag_01__T_kills_last_3s`: contribution `+0.004251`
- `lag_05__CT3__flash_duration`: contribution `+0.003949`

Top utility-only movements:
- `lag_05__CT3__flash_duration`: contribution `+0.003949`
- `lag_14__T5__flash_duration`: contribution `+0.003416`
- `lag_09__CT_A_site_active_infernos`: contribution `+0.001779`

### tick `84474`, seconds `24.00`, LSTM delta `-0.0731`

Top all feature movements:
- `lag_04__CT_place_ARCH`: contribution `-0.006578`
- `lag_10__CT3__flash_duration`: contribution `-0.005650`
- `lag_08__T5__flash_duration`: contribution `-0.004720`
- `lag_01__T_kills_last_3s`: contribution `-0.004251`
- `lag_04__CT_place_LIBRARY`: contribution `-0.004045`

Top utility-only movements:
- `lag_10__CT3__flash_duration`: contribution `-0.005650`
- `lag_08__T5__flash_duration`: contribution `-0.004720`
- `lag_10__CT1__flash`: contribution `-0.002688`
- `lag_15__CT_A_site_active_infernos`: contribution `-0.002098`
- `lag_11__CT_B_site_active_infernos`: contribution `-0.001992`
