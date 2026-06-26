# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-spirit-vs-the-huns-bo3-TWIJIxJZifB3vPv3OUvjVr/spirit-vs-the-huns-m2-dust2.csv`
- round_num: `7`

## Largest probability jumps

- tick `63469`, seconds `102.50`, LSTM `0.2093`, delta `-0.2732`
- tick `61293`, seconds `68.50`, LSTM `0.1225`, delta `-0.2426`
- tick `62157`, seconds `82.00`, LSTM `0.4925`, delta `+0.1944`
- tick `61229`, seconds `67.50`, LSTM `0.3678`, delta `+0.1263`
- tick `63277`, seconds `99.50`, LSTM `0.4621`, delta `-0.1256`
- tick `63501`, seconds `103.00`, LSTM `0.1321`, delta `-0.0772`
- tick `62317`, seconds `84.50`, LSTM `0.5684`, delta `+0.0749`
- tick `61869`, seconds `77.50`, LSTM `0.1308`, delta `+0.0635`
- tick `59021`, seconds `33.00`, LSTM `0.1479`, delta `-0.0617`
- tick `62093`, seconds `81.00`, LSTM `0.2912`, delta `+0.0554`

## Top 15 local ridge features

- `lag_00__T_bomb_zone_count`: coefficient `0.003646`, |coef| `0.003646`
- `lag_01__T_bomb_zone_count`: coefficient `0.003444`, |coef| `0.003444`
- `lag_02__T_bomb_zone_count`: coefficient `0.002227`, |coef| `0.002227`
- `lag_00__kill_diff_last_3s`: coefficient `0.002220`, |coef| `0.002220`
- `lag_00__T_place_ARAMP`: coefficient `-0.002094`, |coef| `0.002094`
- `lag_00__T_kills_last_3s`: coefficient `-0.001992`, |coef| `0.001992`
- `lag_03__CT5__duck_amount`: coefficient `0.001833`, |coef| `0.001833`
- `lag_00__T3__has_bomb`: coefficient `0.001829`, |coef| `0.001829`
- `lag_00__T_damage_last_5s`: coefficient `-0.001823`, |coef| `0.001823`
- `lag_01__T_place_ARAMP`: coefficient `-0.001773`, |coef| `0.001773`
- `lag_03__T_bomb_zone_count`: coefficient `0.001717`, |coef| `0.001717`
- `lag_00__CT_place_EXTENDEDA`: coefficient `0.001687`, |coef| `0.001687`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001651`, |coef| `0.001651`
- `lag_01__CT4__shots_fired`: coefficient `0.001642`, |coef| `0.001642`
- `lag_00__bomb_planted`: coefficient `-0.001623`, |coef| `0.001623`

## Top 10 utility ridge features

- `lag_13__T_A_site_active_smokes`: coefficient `0.001453` (raises CT win probability)
- `lag_15__T_A_site_active_smokes`: coefficient `0.001194` (raises CT win probability)
- `lag_09__T_A_site_active_smokes`: coefficient `0.001069` (raises CT win probability)
- `lag_13__T_active_smokes`: coefficient `0.001048` (raises CT win probability)
- `lag_12__T_A_site_active_smokes`: coefficient `0.000890` (raises CT win probability)
- `lag_15__T_active_smokes`: coefficient `0.000842` (raises CT win probability)
- `lag_13__active_smokes_total`: coefficient `0.000829` (raises CT win probability)
- `lag_11__T_A_site_active_smokes`: coefficient `0.000795` (raises CT win probability)
- `lag_04__CT2__flash_duration`: coefficient `-0.000766` (lowers CT win probability)
- `lag_09__T_active_smokes`: coefficient `0.000757` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_bomb_zone_count`: coefficient `0.003646` (raises CT win probability)
- `lag_01__T_bomb_zone_count`: coefficient `0.003444` (raises CT win probability)
- `lag_02__T_bomb_zone_count`: coefficient `0.002227` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002220` (raises CT win probability)
- `lag_00__T_place_ARAMP`: coefficient `-0.002094` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001992` (lowers CT win probability)
- `lag_03__CT5__duck_amount`: coefficient `0.001833` (raises CT win probability)
- `lag_00__T3__has_bomb`: coefficient `0.001829` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001823` (lowers CT win probability)
- `lag_01__T_place_ARAMP`: coefficient `-0.001773` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `63469`, seconds `102.50`, LSTM delta `-0.2732`

Top all feature movements:
- `lag_00__T_bomb_zone_count`: contribution `-0.021224`
- `lag_00__CT_place_EXTENDEDA`: contribution `-0.018941`
- `lag_01__CT4__shots_fired`: contribution `-0.009731`
- `lag_01__CT_shots_fired_sum`: contribution `-0.009003`
- `lag_06__CT_place_SHORTSTAIRS`: contribution `-0.006925`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `61293`, seconds `68.50`, LSTM delta `-0.2426`

Top all feature movements:
- `lag_07__T_place_ARAMP`: contribution `-0.011509`
- `lag_11__T_place_ARAMP`: contribution `-0.011225`
- `lag_02__T_place_PIT`: contribution `-0.006943`
- `lag_00__T_kills_last_3s`: contribution `-0.006312`
- `lag_02__T_place_ARAMP`: contribution `-0.006130`

Top utility-only movements:
- `lag_04__CT2__flash_duration`: contribution `-0.004011`
- `lag_04__CT1__flash_duration`: contribution `-0.003451`

### tick `62157`, seconds `82.00`, LSTM delta `+0.1944`

Top all feature movements:
- `lag_01__T_bomb_zone_count`: contribution `+0.020047`
- `lag_01__T_place_ARAMP`: contribution `+0.016045`
- `lag_00__CT_shots_fired_sum`: contribution `+0.011473`
- `lag_15__T_place_ARAMP`: contribution `+0.006754`
- `lag_00__kill_diff_last_3s`: contribution `+0.005343`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `61229`, seconds `67.50`, LSTM delta `+0.1263`

Top all feature movements:
- `lag_00__T_place_ARAMP`: contribution `+0.018947`
- `lag_00__CT_place_EXTENDEDA`: contribution `-0.009471`
- `lag_09__T_place_ARAMP`: contribution `+0.006394`
- `lag_00__CT_place_SHORTSTAIRS`: contribution `+0.005778`
- `lag_03__T_place_PIT`: contribution `+0.005518`

Top utility-only movements:
- `lag_02__CT1__flash_duration`: contribution `+0.002621`

### tick `63277`, seconds `99.50`, LSTM delta `-0.1256`

Top all feature movements:
- `lag_00__CT_place_EXTENDEDA`: contribution `+0.009471`
- `lag_00__T_kills_last_3s`: contribution `-0.006312`
- `lag_00__CT_place_SHORTSTAIRS`: contribution `-0.005778`
- `lag_00__kill_diff_last_3s`: contribution `-0.005343`
- `lag_14__T3__duck_amount`: contribution `-0.004277`

Top utility-only movements:
- `lag_13__T_A_site_active_smokes`: contribution `-0.002067`
