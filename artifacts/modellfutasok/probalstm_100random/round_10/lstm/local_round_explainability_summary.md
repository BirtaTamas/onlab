# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-spirit-vs-flyquest-bo3-fQI-qOiPd1cRkmhkz0Xs5h/spirit-vs-flyquest-m1-mirage.csv`
- round_num: `1`

## Largest probability jumps

- tick `5116`, seconds `56.00`, LSTM `0.7620`, delta `+0.1656`
- tick `4156`, seconds `41.00`, LSTM `0.6541`, delta `+0.1384`
- tick `5404`, seconds `60.50`, LSTM `0.9391`, delta `+0.1090`
- tick `4188`, seconds `41.50`, LSTM `0.7241`, delta `+0.0700`
- tick `4860`, seconds `52.00`, LSTM `0.6701`, delta `-0.0592`
- tick `5564`, seconds `63.00`, LSTM `0.8896`, delta `-0.0558`
- tick `5628`, seconds `64.00`, LSTM `0.9524`, delta `+0.0440`
- tick `4508`, seconds `46.50`, LSTM `0.7395`, delta `-0.0427`
- tick `4956`, seconds `53.50`, LSTM `0.5858`, delta `-0.0423`
- tick `4252`, seconds `42.50`, LSTM `0.7825`, delta `+0.0415`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.002305`, |coef| `0.002305`
- `lag_00__kill_diff_last_3s`: coefficient `0.002157`, |coef| `0.002157`
- `lag_04__CT_place_CONNECTOR`: coefficient `-0.001916`, |coef| `0.001916`
- `lag_00__damage_diff_last_5s`: coefficient `0.001602`, |coef| `0.001602`
- `lag_00__T_place_UNDERPASS`: coefficient `-0.001538`, |coef| `0.001538`
- `lag_00__CT_damage_last_5s`: coefficient `0.001464`, |coef| `0.001464`
- `lag_00__CT_burning_players`: coefficient `0.001402`, |coef| `0.001402`
- `lag_05__CT_place_STAIRS`: coefficient `-0.001381`, |coef| `0.001381`
- `lag_05__CT_place_CONNECTOR`: coefficient `-0.001350`, |coef| `0.001350`
- `lag_00__T4__alive`: coefficient `-0.001341`, |coef| `0.001341`
- `lag_02__T_bomb_zone_count`: coefficient `-0.001300`, |coef| `0.001300`
- `lag_09__T_place_CTSPAWN`: coefficient `0.001265`, |coef| `0.001265`
- `lag_04__CT_place_MIDDLE`: coefficient `0.001250`, |coef| `0.001250`
- `lag_00__T_B_site_active_infernos`: coefficient `0.001216`, |coef| `0.001216`
- `lag_00__T4__armor`: coefficient `-0.001176`, |coef| `0.001176`

## Top 10 utility ridge features

- `lag_00__T_B_site_active_infernos`: coefficient `0.001216` (raises CT win probability)
- `lag_05__T1__molly`: coefficient `-0.001095` (lowers CT win probability)
- `lag_12__T1__smoke`: coefficient `-0.001041` (lowers CT win probability)
- `lag_00__T_active_infernos`: coefficient `0.000896` (raises CT win probability)
- `lag_01__T_B_site_active_infernos`: coefficient `0.000788` (raises CT win probability)
- `lag_04__T_A_site_active_smokes`: coefficient `0.000764` (raises CT win probability)
- `lag_06__T1__molly`: coefficient `-0.000757` (lowers CT win probability)
- `lag_13__T1__smoke`: coefficient `-0.000661` (lowers CT win probability)
- `lag_00__active_infernos_total`: coefficient `0.000618` (raises CT win probability)
- `lag_01__T_active_infernos`: coefficient `0.000566` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.002305` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002157` (raises CT win probability)
- `lag_04__CT_place_CONNECTOR`: coefficient `-0.001916` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001602` (raises CT win probability)
- `lag_00__T_place_UNDERPASS`: coefficient `-0.001538` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001464` (raises CT win probability)
- `lag_00__CT_burning_players`: coefficient `0.001402` (raises CT win probability)
- `lag_05__CT_place_STAIRS`: coefficient `-0.001381` (lowers CT win probability)
- `lag_05__CT_place_CONNECTOR`: coefficient `-0.001350` (lowers CT win probability)
- `lag_00__T4__alive`: coefficient `-0.001341` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `5116`, seconds `56.00`, LSTM delta `+0.1656`

Top all feature movements:
- `lag_05__CT_place_STAIRS`: contribution `+0.010751`
- `lag_02__T_bomb_zone_count`: contribution `+0.007570`
- `lag_11__CT_place_STAIRS`: contribution `+0.007352`
- `lag_04__CT_place_CONNECTOR`: contribution `+0.006853`
- `lag_00__CT_kills_last_3s`: contribution `+0.006655`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `4156`, seconds `41.00`, LSTM delta `+0.1384`

Top all feature movements:
- `lag_04__CT_place_CONNECTOR`: contribution `+0.006853`
- `lag_00__CT_kills_last_3s`: contribution `+0.006655`
- `lag_00__T_place_UNDERPASS`: contribution `+0.006024`
- `lag_00__kill_diff_last_3s`: contribution `+0.005192`
- `lag_13__T2__duck_amount`: contribution `+0.004384`

Top utility-only movements:
- `lag_00__T_B_site_active_infernos`: contribution `+0.003438`
- `lag_05__T1__molly`: contribution `+0.002425`

### tick `5404`, seconds `60.50`, LSTM delta `+0.1090`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.006655`
- `lag_09__CT_place_SNIPERSNEST`: contribution `+0.005991`
- `lag_00__kill_diff_last_3s`: contribution `+0.005192`
- `lag_13__CT_place_JUNGLE`: contribution `+0.004429`
- `lag_03__CT_place_JUNGLE`: contribution `+0.004400`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `4188`, seconds `41.50`, LSTM delta `+0.0700`

Top all feature movements:
- `lag_05__CT_place_CONNECTOR`: contribution `+0.004827`
- `lag_00__CT_place_SNIPERSNEST`: contribution `+0.003095`
- `lag_01__T_place_UNDERPASS`: contribution `+0.003058`
- `lag_02__T3__duck_amount`: contribution `+0.002917`
- `lag_01__T2__has_bomb`: contribution `+0.002481`

Top utility-only movements:
- `lag_01__T_B_site_active_infernos`: contribution `+0.002228`
- `lag_06__T1__molly`: contribution `+0.001676`
- `lag_13__T1__smoke`: contribution `+0.001426`

### tick `4860`, seconds `52.00`, LSTM delta `-0.0592`

Top all feature movements:
- `lag_13__T_place_SCAFFOLDING`: contribution `-0.034316`
- `lag_11__T_place_SCAFFOLDING`: contribution `-0.007901`
- `lag_04__CT_place_MIDDLE`: contribution `+0.006555`
- `lag_03__CT_place_STAIRS`: contribution `-0.004558`
- `lag_00__T2__duck_amount`: contribution `-0.002981`

Top utility-only movements:
- No utility movement among the top local contributors.
