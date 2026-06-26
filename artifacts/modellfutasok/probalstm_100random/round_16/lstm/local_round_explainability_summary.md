# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-furia-vs-aurora-bo3-0icw3xvkvOZhHsCT2PEavZ/furia-vs-aurora-m1-inferno.csv`
- round_num: `16`

## Largest probability jumps

- tick `132588`, seconds `21.50`, LSTM `0.1228`, delta `-0.1751`
- tick `133164`, seconds `30.50`, LSTM `0.0379`, delta `-0.1138`
- tick `132204`, seconds `15.50`, LSTM `0.2970`, delta `+0.0585`
- tick `131244`, seconds `0.50`, LSTM `0.2461`, delta `-0.0461`
- tick `132908`, seconds `26.50`, LSTM `0.1347`, delta `+0.0333`
- tick `131532`, seconds `5.00`, LSTM `0.2939`, delta `+0.0304`
- tick `131788`, seconds `9.00`, LSTM `0.2532`, delta `-0.0279`
- tick `132044`, seconds `13.00`, LSTM `0.2188`, delta `-0.0260`
- tick `132236`, seconds `16.00`, LSTM `0.3203`, delta `+0.0233`
- tick `132268`, seconds `16.50`, LSTM `0.3427`, delta `+0.0224`

## Top 15 local ridge features

- `lag_02__CT_utility_damage_last_5s`: coefficient `0.001321`, |coef| `0.001321`
- `lag_00__T_kills_last_3s`: coefficient `-0.001305`, |coef| `0.001305`
- `lag_12__CT_utility_damage_last_5s`: coefficient `-0.001240`, |coef| `0.001240`
- `lag_00__T_damage_last_5s`: coefficient `-0.001177`, |coef| `0.001177`
- `lag_04__T1__is_walking`: coefficient `0.001101`, |coef| `0.001101`
- `lag_02__utility_damage_diff_last_5s`: coefficient `0.001075`, |coef| `0.001075`
- `lag_07__CT_place_RUINS`: coefficient `-0.001029`, |coef| `0.001029`
- `lag_12__utility_damage_diff_last_5s`: coefficient `-0.001024`, |coef| `0.001024`
- `lag_00__kill_diff_last_3s`: coefficient `0.000991`, |coef| `0.000991`
- `lag_00__CT_place_TOPOFMID`: coefficient `0.000989`, |coef| `0.000989`
- `lag_07__CT2__duck_amount`: coefficient `-0.000944`, |coef| `0.000944`
- `lag_03__T_place_SECONDMID`: coefficient `0.000917`, |coef| `0.000917`
- `lag_00__damage_diff_last_5s`: coefficient `0.000882`, |coef| `0.000882`
- `lag_08__CT_place_RUINS`: coefficient `-0.000869`, |coef| `0.000869`
- `lag_12__T4__has_bomb`: coefficient `-0.000864`, |coef| `0.000864`

## Top 10 utility ridge features

- `lag_02__CT_utility_damage_last_5s`: coefficient `0.001321` (raises CT win probability)
- `lag_12__CT_utility_damage_last_5s`: coefficient `-0.001240` (lowers CT win probability)
- `lag_02__utility_damage_diff_last_5s`: coefficient `0.001075` (raises CT win probability)
- `lag_12__utility_damage_diff_last_5s`: coefficient `-0.001024` (lowers CT win probability)
- `lag_05__T_A_site_active_infernos`: coefficient `-0.000854` (lowers CT win probability)
- `lag_06__T_B_site_active_infernos`: coefficient `0.000757` (raises CT win probability)
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.000708` (raises CT win probability)
- `lag_01__CT_utility_damage_last_5s`: coefficient `0.000700` (raises CT win probability)
- `lag_08__T3__molly`: coefficient `0.000680` (raises CT win probability)
- `lag_09__T_A_site_active_infernos`: coefficient `0.000647` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.001305` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001177` (lowers CT win probability)
- `lag_04__T1__is_walking`: coefficient `0.001101` (raises CT win probability)
- `lag_07__CT_place_RUINS`: coefficient `-0.001029` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000991` (raises CT win probability)
- `lag_00__CT_place_TOPOFMID`: coefficient `0.000989` (raises CT win probability)
- `lag_07__CT2__duck_amount`: coefficient `-0.000944` (lowers CT win probability)
- `lag_03__T_place_SECONDMID`: coefficient `0.000917` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000882` (raises CT win probability)
- `lag_08__CT_place_RUINS`: coefficient `-0.000869` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `132588`, seconds `21.50`, LSTM delta `-0.1751`

Top all feature movements:
- `lag_02__CT_utility_damage_last_5s`: contribution `-0.005817`
- `lag_12__CT_utility_damage_last_5s`: contribution `-0.005458`
- `lag_00__T_kills_last_3s`: contribution `-0.004134`
- `lag_02__utility_damage_diff_last_5s`: contribution `-0.003884`
- `lag_12__utility_damage_diff_last_5s`: contribution `-0.003699`

Top utility-only movements:
- `lag_02__CT_utility_damage_last_5s`: contribution `-0.005817`
- `lag_12__CT_utility_damage_last_5s`: contribution `-0.005458`
- `lag_02__utility_damage_diff_last_5s`: contribution `-0.003884`
- `lag_12__utility_damage_diff_last_5s`: contribution `-0.003699`
- `lag_05__T_A_site_active_infernos`: contribution `-0.002543`

### tick `133164`, seconds `30.50`, LSTM delta `-0.1138`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.004134`
- `lag_07__CT_place_RUINS`: contribution `-0.003595`
- `lag_08__CT_place_RUINS`: contribution `-0.003036`
- `lag_00__CT_place_APARTMENTS`: contribution `-0.002992`
- `lag_00__T_damage_last_5s`: contribution `-0.002822`

Top utility-only movements:
- `lag_09__T_A_site_active_infernos`: contribution `-0.001925`
- `lag_09__T_active_infernos`: contribution `-0.001295`

### tick `132204`, seconds `15.50`, LSTM delta `+0.0585`

Top all feature movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.003117`
- `lag_07__CT_place_BALCONY`: contribution `+0.003099`
- `lag_04__T1__is_walking`: contribution `+0.002512`
- `lag_11__T_place_MIDDLE`: contribution `+0.002364`
- `lag_15__CT_place_RUINS`: contribution `+0.002235`

Top utility-only movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.003117`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.002097`
- `lag_08__T_active_infernos`: contribution `+0.001102`
- `lag_08__T_B_site_active_infernos`: contribution `+0.001022`

### tick `131244`, seconds `0.50`, LSTM delta `-0.0461`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.001990`
- `lag_01__T_place_TSPAWN`: contribution `-0.001662`
- `lag_01__T_closest_enemy_dist`: contribution `-0.001527`
- `lag_00__CT_velocity_mean`: contribution `-0.001522`
- `lag_00__T_velocity_mean`: contribution `-0.001390`

Top utility-only movements:
- `lag_01__CT3__flash`: contribution `-0.000785`
- `lag_01__T4__flash`: contribution `-0.000496`
- `lag_01__T4__utility_total`: contribution `-0.000444`
- `lag_01__utility_inv_diff`: contribution `-0.000410`
- `lag_01__T_utility_inv`: contribution `-0.000399`

### tick `132908`, seconds `26.50`, LSTM delta `+0.0333`

Top all feature movements:
- `lag_12__CT_utility_damage_last_5s`: contribution `+0.005458`
- `lag_12__utility_damage_diff_last_5s`: contribution `+0.003699`
- `lag_00__T_damage_last_5s`: contribution `+0.002822`
- `lag_03__T_place_TRAMP`: contribution `+0.002062`
- `lag_00__damage_diff_last_5s`: contribution `+0.001990`

Top utility-only movements:
- `lag_12__CT_utility_damage_last_5s`: contribution `+0.005458`
- `lag_12__utility_damage_diff_last_5s`: contribution `+0.003699`
- `lag_01__T_A_site_active_infernos`: contribution `+0.000823`
