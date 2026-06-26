# Local Round Explainability

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-jijiehao-vs-lynn-vision-bo3-vHZRr1xxhgwfg-A38MzOQQ/jijiehao-vs-lynn-vision-m2-dust2.csv`
- round_num: `8`

## Largest probability jumps

- tick `48837`, seconds `35.50`, LSTM `0.1010`, delta `-0.1722`
- tick `46597`, seconds `0.50`, LSTM `0.1480`, delta `-0.0748`
- tick `48933`, seconds `37.00`, LSTM `0.0212`, delta `-0.0537`
- tick `47589`, seconds `16.00`, LSTM `0.2086`, delta `+0.0305`
- tick `48485`, seconds `30.00`, LSTM `0.2972`, delta `+0.0290`
- tick `47077`, seconds `8.00`, LSTM `0.1342`, delta `+0.0256`
- tick `47397`, seconds `13.00`, LSTM `0.1449`, delta `-0.0252`
- tick `47845`, seconds `20.00`, LSTM `0.2142`, delta `+0.0198`
- tick `46757`, seconds `3.00`, LSTM `0.0887`, delta `-0.0191`
- tick `48517`, seconds `30.50`, LSTM `0.2783`, delta `-0.0189`

## Top 15 local ridge features

- `lag_05__CT_place_EXTENDEDA`: coefficient `-0.001234`, |coef| `0.001234`
- `lag_15__T5__is_scoped`: coefficient `-0.001016`, |coef| `0.001016`
- `lag_11__CT_place_EXTENDEDA`: coefficient `0.000990`, |coef| `0.000990`
- `lag_04__CT_place_EXTENDEDA`: coefficient `-0.000935`, |coef| `0.000935`
- `lag_09__T_place_LONGDOORS`: coefficient `-0.000881`, |coef| `0.000881`
- `lag_09__T_place_OUTSIDELONG`: coefficient `0.000841`, |coef| `0.000841`
- `lag_00__T_kills_last_3s`: coefficient `-0.000840`, |coef| `0.000840`
- `lag_07__T_place_OUTSIDELONG`: coefficient `0.000792`, |coef| `0.000792`
- `lag_10__CT2__duck_amount`: coefficient `-0.000770`, |coef| `0.000770`
- `lag_11__T4__duck_amount`: coefficient `0.000752`, |coef| `0.000752`
- `lag_00__T_damage_last_5s`: coefficient `-0.000751`, |coef| `0.000751`
- `lag_07__T_place_LONGDOORS`: coefficient `-0.000743`, |coef| `0.000743`
- `lag_01__CT3__is_walking`: coefficient `0.000708`, |coef| `0.000708`
- `lag_05__CT3__is_walking`: coefficient `0.000698`, |coef| `0.000698`
- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000693`, |coef| `0.000693`

## Top 10 utility ridge features

- `lag_00__CT3__smoke`: coefficient `0.000604` (raises CT win probability)
- `lag_00__CT_he_last_5s`: coefficient `-0.000596` (lowers CT win probability)
- `lag_04__CT_he_last_5s`: coefficient `-0.000570` (lowers CT win probability)
- `lag_14__T4__smoke`: coefficient `0.000551` (raises CT win probability)
- `lag_03__T_active_infernos`: coefficient `0.000548` (raises CT win probability)
- `lag_08__CT5__flash`: coefficient `0.000500` (raises CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000434` (raises CT win probability)
- `lag_03__active_infernos_total`: coefficient `0.000385` (raises CT win probability)
- `lag_01__CT5__flash`: coefficient `-0.000364` (lowers CT win probability)
- `lag_14__CT_he_last_5s`: coefficient `0.000352` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_05__CT_place_EXTENDEDA`: coefficient `-0.001234` (lowers CT win probability)
- `lag_15__T5__is_scoped`: coefficient `-0.001016` (lowers CT win probability)
- `lag_11__CT_place_EXTENDEDA`: coefficient `0.000990` (raises CT win probability)
- `lag_04__CT_place_EXTENDEDA`: coefficient `-0.000935` (lowers CT win probability)
- `lag_09__T_place_LONGDOORS`: coefficient `-0.000881` (lowers CT win probability)
- `lag_09__T_place_OUTSIDELONG`: coefficient `0.000841` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.000840` (lowers CT win probability)
- `lag_07__T_place_OUTSIDELONG`: coefficient `0.000792` (raises CT win probability)
- `lag_10__CT2__duck_amount`: coefficient `-0.000770` (lowers CT win probability)
- `lag_11__T4__duck_amount`: coefficient `0.000752` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `48837`, seconds `35.50`, LSTM delta `-0.1722`

Top all feature movements:
- `lag_05__CT_place_EXTENDEDA`: contribution `-0.006925`
- `lag_11__CT_place_EXTENDEDA`: contribution `-0.005558`
- `lag_04__CT_place_EXTENDEDA`: contribution `-0.005249`
- `lag_15__T5__is_scoped`: contribution `-0.004844`
- `lag_03__CT_place_EXTENDEDA`: contribution `-0.003187`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `46597`, seconds `0.50`, LSTM delta `-0.0748`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.003313`
- `lag_01__T_place_TSPAWN`: contribution `-0.002701`
- `lag_00__T_velocity_mean`: contribution `-0.002501`
- `lag_00__CT_velocity_mean`: contribution `-0.002152`
- `lag_01__T_closest_enemy_dist`: contribution `-0.001297`

Top utility-only movements:
- `lag_01__molly_inv_diff`: contribution `-0.001212`
- `lag_01__CT5__flash`: contribution `-0.001042`
- `lag_01__T_smoke_inv`: contribution `-0.000796`
- `lag_01__T_molly_inv`: contribution `-0.000790`
- `lag_01__CT5__utility_total`: contribution `-0.000556`

### tick `48933`, seconds `37.00`, LSTM delta `-0.0537`

Top all feature movements:
- `lag_01__T_place_SHORTSTAIRS`: contribution `-0.004786`
- `lag_14__CT_place_EXTENDEDA`: contribution `-0.002812`
- `lag_00__T_kills_last_3s`: contribution `-0.002662`
- `lag_00__CT_place_EXTENDEDA`: contribution `-0.002393`
- `lag_06__CT_place_EXTENDEDA`: contribution `+0.002393`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `47589`, seconds `16.00`, LSTM delta `+0.0305`

Top all feature movements:
- `lag_06__T_place_TUNNELSTAIRS`: contribution `+0.002720`
- `lag_12__T_place_TUNNELSTAIRS`: contribution `+0.001859`
- `lag_02__CT_place_EXTENDEDA`: contribution `+0.001756`
- `lag_06__T_place_LOWERTUNNEL`: contribution `+0.001530`
- `lag_08__CT_place_LONGDOORS`: contribution `+0.001463`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `48485`, seconds `30.00`, LSTM delta `+0.0290`

Top all feature movements:
- `lag_05__CT_place_EXTENDEDA`: contribution `+0.006925`
- `lag_00__CT_place_EXTENDEDA`: contribution `-0.002393`
- `lag_01__CT_place_EXTENDEDA`: contribution `+0.002263`
- `lag_12__T2__duck_amount`: contribution `+0.001952`
- `lag_10__T_place_OUTSIDELONG`: contribution `+0.001519`

Top utility-only movements:
- No utility movement among the top local contributors.
