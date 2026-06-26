# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m2-dust2.csv`
- round_num: `5`

## Largest probability jumps

- tick `39034`, seconds `112.00`, LSTM `0.3477`, delta `-0.3307`
- tick `38874`, seconds `109.50`, LSTM `0.6832`, delta `+0.2805`
- tick `38810`, seconds `108.50`, LSTM `0.3841`, delta `-0.1742`
- tick `40058`, seconds `128.00`, LSTM `0.0496`, delta `-0.1293`
- tick `39738`, seconds `123.00`, LSTM `0.2549`, delta `+0.0993`
- tick `38010`, seconds `96.00`, LSTM `0.5785`, delta `+0.0755`
- tick `39194`, seconds `114.50`, LSTM `0.2642`, delta `-0.0722`
- tick `39898`, seconds `125.50`, LSTM `0.1946`, delta `-0.0673`
- tick `38394`, seconds `102.00`, LSTM `0.5753`, delta `-0.0591`
- tick `38074`, seconds `97.00`, LSTM `0.6598`, delta `+0.0591`

## Top 15 local ridge features

- `lag_06__T_place_EXTENDEDA`: coefficient `0.002825`, |coef| `0.002825`
- `lag_00__damage_diff_last_5s`: coefficient `0.002148`, |coef| `0.002148`
- `lag_09__CT1__flash_duration`: coefficient `0.002079`, |coef| `0.002079`
- `lag_00__CT_place_UNDERA`: coefficient `0.001926`, |coef| `0.001926`
- `lag_00__kill_diff_last_3s`: coefficient `0.001911`, |coef| `0.001911`
- `lag_02__CT1__flash_duration`: coefficient `0.001904`, |coef| `0.001904`
- `lag_05__T_bomb_zone_count`: coefficient `0.001826`, |coef| `0.001826`
- `lag_10__CT_flashed_players`: coefficient `-0.001798`, |coef| `0.001798`
- `lag_15__T_utility_damage_last_5s`: coefficient `0.001790`, |coef| `0.001790`
- `lag_15__CT_place_BDOORS`: coefficient `0.001770`, |coef| `0.001770`
- `lag_02__CT_flashed_players`: coefficient `0.001713`, |coef| `0.001713`
- `lag_01__T_place_EXTENDEDA`: coefficient `-0.001591`, |coef| `0.001591`
- `lag_10__T_utility_damage_last_5s`: coefficient `0.001548`, |coef| `0.001548`
- `lag_10__T1__flash_duration`: coefficient `-0.001536`, |coef| `0.001536`
- `lag_10__T_flashed_players`: coefficient `-0.001517`, |coef| `0.001517`

## Top 10 utility ridge features

- `lag_09__CT1__flash_duration`: coefficient `0.002079` (raises CT win probability)
- `lag_02__CT1__flash_duration`: coefficient `0.001904` (raises CT win probability)
- `lag_15__T_utility_damage_last_5s`: coefficient `0.001790` (raises CT win probability)
- `lag_10__T_utility_damage_last_5s`: coefficient `0.001548` (raises CT win probability)
- `lag_10__T1__flash_duration`: coefficient `-0.001536` (lowers CT win probability)
- `lag_05__T1__flash_duration`: coefficient `-0.001488` (lowers CT win probability)
- `lag_00__T1__flash_duration`: coefficient `0.001416` (raises CT win probability)
- `lag_02__T4__flash_duration`: coefficient `0.001401` (raises CT win probability)
- `lag_05__T_utility_damage_last_5s`: coefficient `-0.001349` (lowers CT win probability)
- `lag_07__T4__flash_duration`: coefficient `-0.001161` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_06__T_place_EXTENDEDA`: coefficient `0.002825` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002148` (raises CT win probability)
- `lag_00__CT_place_UNDERA`: coefficient `0.001926` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001911` (raises CT win probability)
- `lag_05__T_bomb_zone_count`: coefficient `0.001826` (raises CT win probability)
- `lag_10__CT_flashed_players`: coefficient `-0.001798` (lowers CT win probability)
- `lag_15__CT_place_BDOORS`: coefficient `0.001770` (raises CT win probability)
- `lag_02__CT_flashed_players`: coefficient `0.001713` (raises CT win probability)
- `lag_01__T_place_EXTENDEDA`: coefficient `-0.001591` (lowers CT win probability)
- `lag_10__T_flashed_players`: coefficient `-0.001517` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `39034`, seconds `112.00`, LSTM delta `-0.3307`

Top all feature movements:
- `lag_06__T_place_EXTENDEDA`: contribution `-0.028010`
- `lag_09__CT1__flash_duration`: contribution `-0.010419`
- `lag_10__T_utility_damage_last_5s`: contribution `-0.008841`
- `lag_15__CT_place_BDOORS`: contribution `-0.008514`
- `lag_10__T_bomb_zone_count`: contribution `-0.007190`

Top utility-only movements:
- `lag_09__CT1__flash_duration`: contribution `-0.010419`
- `lag_10__T_utility_damage_last_5s`: contribution `-0.008841`
- `lag_07__T4__flash_duration`: contribution `-0.006461`
- `lag_08__T4__flash_duration`: contribution `-0.005392`
- `lag_02__CT1__flash_duration`: contribution `-0.005212`

### tick `38874`, seconds `109.50`, LSTM delta `+0.2805`

Top all feature movements:
- `lag_01__T_place_EXTENDEDA`: contribution `+0.015779`
- `lag_06__T_place_EXTENDEDA`: contribution `+0.014005`
- `lag_05__T_bomb_zone_count`: contribution `+0.010630`
- `lag_15__T_utility_damage_last_5s`: contribution `+0.010222`
- `lag_02__T4__flash_duration`: contribution `+0.007799`

Top utility-only movements:
- `lag_15__T_utility_damage_last_5s`: contribution `+0.010222`
- `lag_02__T4__flash_duration`: contribution `+0.007799`
- `lag_05__T_utility_damage_last_5s`: contribution `+0.007705`
- `lag_02__CT1__flash_duration`: contribution `+0.005212`
- `lag_12__T4__flash_duration`: contribution `+0.004559`

### tick `38810`, seconds `108.50`, LSTM delta `-0.1742`

Top all feature movements:
- `lag_06__T_place_EXTENDEDA`: contribution `-0.014005`
- `lag_02__CT1__flash_duration`: contribution `-0.009540`
- `lag_10__T_flashed_players`: contribution `-0.008782`
- `lag_15__CT_place_LONGDOORS`: contribution `-0.006243`
- `lag_00__CT_place_UNDERA`: contribution `-0.005883`

Top utility-only movements:
- `lag_02__CT1__flash_duration`: contribution `-0.009540`
- `lag_03__T_utility_damage_last_5s`: contribution `-0.005648`
- `lag_01__T4__flash_duration`: contribution `-0.004835`
- `lag_13__T_utility_damage_last_5s`: contribution `-0.003275`

### tick `40058`, seconds `128.00`, LSTM delta `-0.1293`

Top all feature movements:
- `lag_10__CT_flashed_players`: contribution `-0.011810`
- `lag_10__T1__flash_duration`: contribution `-0.009797`
- `lag_00__CT_place_UNDERA`: contribution `-0.005883`
- `lag_10__T_flashed_players`: contribution `-0.005855`
- `lag_00__T_kills_last_3s`: contribution `-0.004623`

Top utility-only movements:
- `lag_10__T1__flash_duration`: contribution `-0.009797`

### tick `39738`, seconds `123.00`, LSTM delta `+0.0993`

Top all feature movements:
- `lag_00__T1__flash_duration`: contribution `+0.009029`
- `lag_00__CT_flashed_players`: contribution `+0.007696`
- `lag_09__T1__duck_amount`: contribution `+0.005578`
- `lag_07__CT_place_ARAMP`: contribution `+0.005076`
- `lag_14__T_place_EXTENDEDA`: contribution `+0.004942`

Top utility-only movements:
- `lag_00__T1__flash_duration`: contribution `+0.009029`
- `lag_00__CT3__flash_duration`: contribution `+0.001398`
- `lag_00__CT_flash_duration_sum`: contribution `+0.001290`
