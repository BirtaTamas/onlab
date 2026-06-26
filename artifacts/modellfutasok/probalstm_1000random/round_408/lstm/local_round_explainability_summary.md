# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-big-vs-furia-bo3-8LyYppfzx0M6KmNUlhRuUi/big-vs-furia-m1-inferno.csv`
- round_num: `15`

## Largest probability jumps

- tick `129914`, seconds `18.50`, LSTM `0.1468`, delta `-0.1317`
- tick `130458`, seconds `27.00`, LSTM `0.0404`, delta `-0.1242`
- tick `130330`, seconds `25.00`, LSTM `0.1364`, delta `+0.0563`
- tick `131930`, seconds `50.00`, LSTM `0.0136`, delta `-0.0474`
- tick `130810`, seconds `32.50`, LSTM `0.0805`, delta `+0.0411`
- tick `130234`, seconds `23.50`, LSTM `0.0834`, delta `-0.0364`
- tick `129850`, seconds `17.50`, LSTM `0.2460`, delta `+0.0342`
- tick `129882`, seconds `18.00`, LSTM `0.2785`, delta `+0.0326`
- tick `129114`, seconds `6.00`, LSTM `0.3762`, delta `+0.0309`
- tick `129818`, seconds `17.00`, LSTM `0.2118`, delta `-0.0307`

## Top 15 local ridge features

- `lag_02__T_place_BALCONY`: coefficient `0.001281`, |coef| `0.001281`
- `lag_07__T_place_BALCONY`: coefficient `-0.001063`, |coef| `0.001063`
- `lag_00__T_place_BALCONY`: coefficient `-0.001061`, |coef| `0.001061`
- `lag_00__T_kills_last_3s`: coefficient `-0.000994`, |coef| `0.000994`
- `lag_00__CT1__flash_duration`: coefficient `-0.000960`, |coef| `0.000960`
- `lag_00__kill_diff_last_3s`: coefficient `0.000842`, |coef| `0.000842`
- `lag_06__CT_place_BRIDGE`: coefficient `-0.000814`, |coef| `0.000814`
- `lag_02__CT1__flash_duration`: coefficient `-0.000776`, |coef| `0.000776`
- `lag_04__T_place_BALCONY`: coefficient `0.000749`, |coef| `0.000749`
- `lag_01__CT1__flash_duration`: coefficient `-0.000723`, |coef| `0.000723`
- `lag_15__T_place_BALCONY`: coefficient `-0.000704`, |coef| `0.000704`
- `lag_06__T1__duck_amount`: coefficient `-0.000699`, |coef| `0.000699`
- `lag_09__CT1__flash_duration`: coefficient `-0.000682`, |coef| `0.000682`
- `lag_04__T5__shots_fired`: coefficient `0.000674`, |coef| `0.000674`
- `lag_03__CT1__flash_duration`: coefficient `-0.000631`, |coef| `0.000631`

## Top 10 utility ridge features

- `lag_00__CT1__flash_duration`: coefficient `-0.000960` (lowers CT win probability)
- `lag_02__CT1__flash_duration`: coefficient `-0.000776` (lowers CT win probability)
- `lag_01__CT1__flash_duration`: coefficient `-0.000723` (lowers CT win probability)
- `lag_09__CT1__flash_duration`: coefficient `-0.000682` (lowers CT win probability)
- `lag_03__CT1__flash_duration`: coefficient `-0.000631` (lowers CT win probability)
- `lag_10__CT1__flash_duration`: coefficient `-0.000622` (lowers CT win probability)
- `lag_00__CT5__flash`: coefficient `0.000601` (raises CT win probability)
- `lag_04__CT1__flash_duration`: coefficient `-0.000574` (lowers CT win probability)
- `lag_13__CT1__flash_duration`: coefficient `-0.000567` (lowers CT win probability)
- `lag_08__CT1__flash_duration`: coefficient `-0.000542` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_02__T_place_BALCONY`: coefficient `0.001281` (raises CT win probability)
- `lag_07__T_place_BALCONY`: coefficient `-0.001063` (lowers CT win probability)
- `lag_00__T_place_BALCONY`: coefficient `-0.001061` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.000994` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000842` (raises CT win probability)
- `lag_06__CT_place_BRIDGE`: coefficient `-0.000814` (lowers CT win probability)
- `lag_04__T_place_BALCONY`: coefficient `0.000749` (raises CT win probability)
- `lag_15__T_place_BALCONY`: coefficient `-0.000704` (lowers CT win probability)
- `lag_06__T1__duck_amount`: coefficient `-0.000699` (lowers CT win probability)
- `lag_04__T5__shots_fired`: coefficient `0.000674` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `129914`, seconds `18.50`, LSTM delta `-0.1317`

Top all feature movements:
- `lag_02__T_place_BALCONY`: contribution `-0.017612`
- `lag_03__T_place_BALCONY`: contribution `-0.005449`
- `lag_02__CT_flashed_players`: contribution `-0.004115`
- `lag_01__CT_place_BANANA`: contribution `-0.003507`
- `lag_00__T_kills_last_3s`: contribution `-0.003149`

Top utility-only movements:
- `lag_12__T_B_site_active_infernos`: contribution `-0.001474`

### tick `130458`, seconds `27.00`, LSTM delta `-0.1242`

Top all feature movements:
- `lag_07__T_place_BALCONY`: contribution `-0.014617`
- `lag_04__T_place_BALCONY`: contribution `-0.010305`
- `lag_04__T5__shots_fired`: contribution `-0.004561`
- `lag_00__T_kills_last_3s`: contribution `-0.003149`
- `lag_00__T_shots_fired_sum`: contribution `-0.002838`

Top utility-only movements:
- `lag_00__CT5__flash`: contribution `-0.002135`

### tick `130330`, seconds `25.00`, LSTM delta `+0.0563`

Top all feature movements:
- `lag_00__T_place_BALCONY`: contribution `+0.014589`
- `lag_15__T_place_BALCONY`: contribution `+0.009688`
- `lag_03__T_place_BALCONY`: contribution `-0.005449`
- `lag_00__T_shots_fired_sum`: contribution `+0.004729`
- `lag_15__CT_flashed_players`: contribution `+0.003193`

Top utility-only movements:
- `lag_12__CT1__flash_duration`: contribution `-0.002285`
- `lag_12__T_B_site_active_infernos`: contribution `+0.001474`
- `lag_12__T_active_infernos`: contribution `+0.000813`

### tick `131930`, seconds `50.00`, LSTM delta `-0.0474`

Top all feature movements:
- `lag_06__CT_place_BRIDGE`: contribution `-0.009328`
- `lag_00__T_kills_last_3s`: contribution `-0.003149`
- `lag_00__T_shots_fired_sum`: contribution `-0.002365`
- `lag_00__kill_diff_last_3s`: contribution `-0.002027`
- `lag_00__CT_place_BANANA`: contribution `-0.001739`

Top utility-only movements:
- `lag_11__T_B_site_active_infernos`: contribution `-0.000878`
- `lag_11__T_active_infernos`: contribution `-0.000532`

### tick `130810`, seconds `32.50`, LSTM delta `+0.0411`

Top all feature movements:
- `lag_15__T_place_BALCONY`: contribution `+0.009688`
- `lag_09__T_shots_fired_sum`: contribution `+0.003209`
- `lag_00__kill_diff_last_3s`: contribution `+0.002027`
- `lag_00__T_shots_fired_sum`: contribution `-0.001892`
- `lag_09__T4__shots_fired`: contribution `+0.001601`

Top utility-only movements:
- No utility movement among the top local contributors.
