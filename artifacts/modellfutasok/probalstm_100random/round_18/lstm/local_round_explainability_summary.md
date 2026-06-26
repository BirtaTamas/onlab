# Local Round Explainability

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-mouz-vs-m80-bo3-v7WxfaSDQDAUAgkS_SwEt2/mouz-vs-m80-m3-inferno.csv`
- round_num: `1`

## Largest probability jumps

- tick `17847`, seconds `29.50`, LSTM `0.8288`, delta `+0.2423`
- tick `17335`, seconds `21.50`, LSTM `0.7414`, delta `+0.1591`
- tick `17719`, seconds `27.50`, LSTM `0.6849`, delta `-0.1291`
- tick `18039`, seconds `32.50`, LSTM `0.7497`, delta `-0.1202`
- tick `16855`, seconds `14.00`, LSTM `0.6058`, delta `+0.1172`
- tick `18967`, seconds `47.00`, LSTM `0.8763`, delta `+0.1141`
- tick `17783`, seconds `28.50`, LSTM `0.5737`, delta `-0.0912`
- tick `17175`, seconds `19.00`, LSTM `0.6099`, delta `-0.0605`
- tick `18935`, seconds `46.50`, LSTM `0.7622`, delta `+0.0548`
- tick `18903`, seconds `46.00`, LSTM `0.7073`, delta `-0.0427`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.002698`, |coef| `0.002698`
- `lag_00__kill_diff_last_3s`: coefficient `0.002658`, |coef| `0.002658`
- `lag_07__CT_place_SECONDMID`: coefficient `0.002228`, |coef| `0.002228`
- `lag_02__T_bomb_zone_count`: coefficient `0.001932`, |coef| `0.001932`
- `lag_00__T_bomb_zone_count`: coefficient `-0.001899`, |coef| `0.001899`
- `lag_00__damage_diff_last_5s`: coefficient `0.001895`, |coef| `0.001895`
- `lag_10__T_place_BALCONY`: coefficient `-0.001750`, |coef| `0.001750`
- `lag_02__CT_place_SECONDMID`: coefficient `-0.001711`, |coef| `0.001711`
- `lag_00__CT_damage_last_5s`: coefficient `0.001704`, |coef| `0.001704`
- `lag_06__T_place_BALCONY`: coefficient `0.001666`, |coef| `0.001666`
- `lag_07__CT_place_TOPOFMID`: coefficient `0.001641`, |coef| `0.001641`
- `lag_00__bomb_events_last_5s`: coefficient `0.001611`, |coef| `0.001611`
- `lag_11__CT_place_APARTMENTS`: coefficient `0.001536`, |coef| `0.001536`
- `lag_12__T_place_BALCONY`: coefficient `0.001440`, |coef| `0.001440`
- `lag_15__T_place_PIT`: coefficient `-0.001426`, |coef| `0.001426`

## Top 10 utility ridge features

- `lag_15__T5__flash_duration`: coefficient `-0.001246` (lowers CT win probability)
- `lag_09__CT3__flash_duration`: coefficient `-0.001173` (lowers CT win probability)
- `lag_05__T4__flash_duration`: coefficient `-0.001115` (lowers CT win probability)
- `lag_02__T4__flash_duration`: coefficient `0.000878` (raises CT win probability)
- `lag_01__CT3__flash_duration`: coefficient `0.000804` (raises CT win probability)
- `lag_09__CT_flash_duration_sum`: coefficient `-0.000760` (lowers CT win probability)
- `lag_00__T5__flash_duration`: coefficient `-0.000686` (lowers CT win probability)
- `lag_00__T4__smoke`: coefficient `-0.000630` (lowers CT win probability)
- `lag_04__CT3__flash_duration`: coefficient `0.000623` (raises CT win probability)
- `lag_09__CT4__flash_duration`: coefficient `-0.000559` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.002698` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002658` (raises CT win probability)
- `lag_07__CT_place_SECONDMID`: coefficient `0.002228` (raises CT win probability)
- `lag_02__T_bomb_zone_count`: coefficient `0.001932` (raises CT win probability)
- `lag_00__T_bomb_zone_count`: coefficient `-0.001899` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001895` (raises CT win probability)
- `lag_10__T_place_BALCONY`: coefficient `-0.001750` (lowers CT win probability)
- `lag_02__CT_place_SECONDMID`: coefficient `-0.001711` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001704` (raises CT win probability)
- `lag_06__T_place_BALCONY`: coefficient `0.001666` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `17847`, seconds `29.50`, LSTM delta `+0.2423`

Top all feature movements:
- `lag_07__CT_place_SECONDMID`: contribution `+0.045677`
- `lag_02__CT_place_SECONDMID`: contribution `+0.035083`
- `lag_10__T_place_BALCONY`: contribution `+0.024060`
- `lag_12__T_place_BALCONY`: contribution `+0.019805`
- `lag_13__T_place_BALCONY`: contribution `+0.016957`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `17335`, seconds `21.50`, LSTM delta `+0.1591`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.007789`
- `lag_05__T4__flash_duration`: contribution `+0.007392`
- `lag_15__T5__flash_duration`: contribution `+0.006529`
- `lag_00__kill_diff_last_3s`: contribution `+0.006399`
- `lag_09__CT3__flash_duration`: contribution `+0.006144`

Top utility-only movements:
- `lag_05__T4__flash_duration`: contribution `+0.007392`
- `lag_15__T5__flash_duration`: contribution `+0.006529`
- `lag_09__CT3__flash_duration`: contribution `+0.006144`
- `lag_09__CT_flash_duration_sum`: contribution `+0.002542`

### tick `17719`, seconds `27.50`, LSTM delta `-0.1291`

Top all feature movements:
- `lag_03__CT_place_SECONDMID`: contribution `-0.028139`
- `lag_06__T_place_BALCONY`: contribution `-0.022909`
- `lag_09__T_place_BALCONY`: contribution `-0.007933`
- `lag_00__kill_diff_last_3s`: contribution `-0.006399`
- `lag_10__CT_place_ARCH`: contribution `-0.004963`

Top utility-only movements:
- `lag_13__CT4__flash_duration`: contribution `-0.001841`

### tick `18039`, seconds `32.50`, LSTM delta `-0.1202`

Top all feature movements:
- `lag_06__T_place_BALCONY`: contribution `-0.022909`
- `lag_08__CT_place_SECONDMID`: contribution `-0.018852`
- `lag_13__CT_place_SECONDMID`: contribution `-0.014776`
- `lag_00__kill_diff_last_3s`: contribution `-0.012797`
- `lag_00__CT_kills_last_3s`: contribution `-0.007789`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `16855`, seconds `14.00`, LSTM delta `+0.1172`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.007789`
- `lag_15__CT_place_TOPOFMID`: contribution `+0.006602`
- `lag_00__kill_diff_last_3s`: contribution `+0.006399`
- `lag_07__CT_place_TOPOFMID`: contribution `+0.005954`
- `lag_02__T4__flash_duration`: contribution `+0.005821`

Top utility-only movements:
- `lag_02__T4__flash_duration`: contribution `+0.005821`
- `lag_01__CT3__flash_duration`: contribution `+0.005684`
- `lag_00__T5__flash_duration`: contribution `+0.003595`
- `lag_01__CT_flash_alpha_mean`: contribution `+0.002022`
- `lag_01__CT4__flash_duration`: contribution `-0.001941`
