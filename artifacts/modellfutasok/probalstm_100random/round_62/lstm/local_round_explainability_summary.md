# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-furia-vs-virtuspro-bo3-E_bOFuD3YUjLJCO2xRj0mq/furia-vs-virtus-pro-m1-mirage.csv`
- round_num: `16`

## Largest probability jumps

- tick `145707`, seconds `19.50`, LSTM `0.3171`, delta `-0.2276`
- tick `149067`, seconds `72.00`, LSTM `0.1413`, delta `-0.2184`
- tick `145771`, seconds `20.50`, LSTM `0.2308`, delta `-0.0890`
- tick `148075`, seconds `56.50`, LSTM `0.2842`, delta `+0.0609`
- tick `146379`, seconds `30.00`, LSTM `0.2605`, delta `-0.0571`
- tick `149099`, seconds `72.50`, LSTM `0.0888`, delta `-0.0525`
- tick `146059`, seconds `25.00`, LSTM `0.2674`, delta `+0.0519`
- tick `145003`, seconds `8.50`, LSTM `0.4869`, delta `+0.0462`
- tick `145419`, seconds `15.00`, LSTM `0.5986`, delta `+0.0416`
- tick `145803`, seconds `21.00`, LSTM `0.1929`, delta `-0.0379`

## Top 15 local ridge features

- `lag_13__T_place_UNDERPASS`: coefficient `0.003688`, |coef| `0.003688`
- `lag_12__T_place_UNDERPASS`: coefficient `0.002906`, |coef| `0.002906`
- `lag_00__T_kills_last_3s`: coefficient `-0.002777`, |coef| `0.002777`
- `lag_13__T5__duck_amount`: coefficient `0.002466`, |coef| `0.002466`
- `lag_00__T_damage_last_5s`: coefficient `-0.002240`, |coef| `0.002240`
- `lag_00__CT_place_STAIRS`: coefficient `0.002095`, |coef| `0.002095`
- `lag_00__CT_place_CONNECTOR`: coefficient `0.002052`, |coef| `0.002052`
- `lag_11__CT3__is_walking`: coefficient `0.002020`, |coef| `0.002020`
- `lag_00__kill_diff_last_3s`: coefficient `0.001945`, |coef| `0.001945`
- `lag_00__T_B_site_active_infernos`: coefficient `-0.001936`, |coef| `0.001936`
- `lag_05__T_place_SIDEALLEY`: coefficient `0.001927`, |coef| `0.001927`
- `lag_00__CT2__alive`: coefficient `0.001918`, |coef| `0.001918`
- `lag_00__CT2__hp`: coefficient `0.001896`, |coef| `0.001896`
- `lag_01__CT4__duck_amount`: coefficient `-0.001883`, |coef| `0.001883`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001871`, |coef| `0.001871`

## Top 10 utility ridge features

- `lag_00__T_B_site_active_infernos`: coefficient `-0.001936` (lowers CT win probability)
- `lag_00__T3__smoke`: coefficient `0.001722` (raises CT win probability)
- `lag_02__T5__molly`: coefficient `0.001559` (raises CT win probability)
- `lag_11__T1__smoke`: coefficient `0.001546` (raises CT win probability)
- `lag_03__T4__molly`: coefficient `0.001525` (raises CT win probability)
- `lag_00__T_active_infernos`: coefficient `-0.001272` (lowers CT win probability)
- `lag_12__CT_utility_damage_last_5s`: coefficient `0.001191` (raises CT win probability)
- `lag_12__CT3__flash_duration`: coefficient `0.001062` (raises CT win probability)
- `lag_06__T_B_site_active_smokes`: coefficient `-0.001061` (lowers CT win probability)
- `lag_15__CT_he_last_5s`: coefficient `0.000930` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_13__T_place_UNDERPASS`: coefficient `0.003688` (raises CT win probability)
- `lag_12__T_place_UNDERPASS`: coefficient `0.002906` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002777` (lowers CT win probability)
- `lag_13__T5__duck_amount`: coefficient `0.002466` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002240` (lowers CT win probability)
- `lag_00__CT_place_STAIRS`: coefficient `0.002095` (raises CT win probability)
- `lag_00__CT_place_CONNECTOR`: coefficient `0.002052` (raises CT win probability)
- `lag_11__CT3__is_walking`: coefficient `0.002020` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001945` (raises CT win probability)
- `lag_05__T_place_SIDEALLEY`: coefficient `0.001927` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `145707`, seconds `19.50`, LSTM delta `-0.2276`

Top all feature movements:
- `lag_05__T_place_SIDEALLEY`: contribution `-0.012290`
- `lag_00__T_shots_fired_sum`: contribution `-0.009620`
- `lag_00__T_kills_last_3s`: contribution `-0.008798`
- `lag_13__T5__duck_amount`: contribution `-0.008770`
- `lag_12__CT_utility_damage_last_5s`: contribution `-0.008653`

Top utility-only movements:
- `lag_12__CT_utility_damage_last_5s`: contribution `-0.008653`
- `lag_12__CT3__flash_duration`: contribution `-0.005396`
- `lag_12__utility_damage_diff_last_5s`: contribution `-0.005059`
- `lag_11__T3__flash_duration`: contribution `-0.004876`
- `lag_12__T1__flash_duration`: contribution `-0.004461`

### tick `149067`, seconds `72.00`, LSTM delta `-0.2184`

Top all feature movements:
- `lag_13__T_place_UNDERPASS`: contribution `-0.014448`
- `lag_12__T_place_UNDERPASS`: contribution `-0.011384`
- `lag_13__T5__duck_amount`: contribution `-0.009364`
- `lag_00__T_kills_last_3s`: contribution `-0.008798`
- `lag_00__CT_place_CONNECTOR`: contribution `-0.007338`

Top utility-only movements:
- `lag_00__T_B_site_active_infernos`: contribution `-0.005473`
- `lag_00__T3__smoke`: contribution `-0.003742`
- `lag_02__T5__molly`: contribution `-0.003449`

### tick `145771`, seconds `20.50`, LSTM delta `-0.0890`

Top all feature movements:
- `lag_07__CT_place_JUNGLE`: contribution `+0.009786`
- `lag_11__CT3__is_scoped`: contribution `-0.005529`
- `lag_00__T_shots_fired_sum`: contribution `-0.005345`
- `lag_14__CT_utility_damage_last_5s`: contribution `-0.004935`
- `lag_07__T_place_SIDEALLEY`: contribution `-0.004880`

Top utility-only movements:
- `lag_14__CT_utility_damage_last_5s`: contribution `-0.004935`
- `lag_14__utility_damage_diff_last_5s`: contribution `-0.003694`
- `lag_14__CT3__flash_duration`: contribution `-0.002604`
- `lag_14__T1__flash_duration`: contribution `-0.002495`

### tick `148075`, seconds `56.50`, LSTM delta `+0.0609`

Top all feature movements:
- `lag_00__CT_place_STAIRS`: contribution `+0.016308`
- `lag_14__T1__duck_amount`: contribution `+0.005674`
- `lag_00__CT1__is_walking`: contribution `+0.003300`
- `lag_04__T4__is_walking`: contribution `+0.003141`
- `lag_15__T1__is_walking`: contribution `+0.003056`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `146379`, seconds `30.00`, LSTM delta `-0.0571`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.012995`
- `lag_07__CT_place_JUNGLE`: contribution `-0.009786`
- `lag_00__CT4__shots_fired`: contribution `-0.007295`
- `lag_10__CT3__is_scoped`: contribution `-0.005872`
- `lag_11__CT_place_JUNGLE`: contribution `-0.003697`

Top utility-only movements:
- No utility movement among the top local contributors.
