# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-imperial-vs-liquid-bo3-eiIGPV5tjvJFQ73hC8D8JI/imperial-vs-liquid-m3-anubis.csv`
- round_num: `10`

## Largest probability jumps

- tick `79062`, seconds `23.00`, LSTM `0.9085`, delta `+0.1182`
- tick `77846`, seconds `4.00`, LSTM `0.8747`, delta `+0.0555`
- tick `77814`, seconds `3.50`, LSTM `0.8192`, delta `+0.0342`
- tick `79222`, seconds `25.50`, LSTM `0.9413`, delta `+0.0268`
- tick `78294`, seconds `11.00`, LSTM `0.8399`, delta `-0.0240`
- tick `77878`, seconds `4.50`, LSTM `0.8965`, delta `+0.0217`
- tick `77782`, seconds `3.00`, LSTM `0.7850`, delta `+0.0214`
- tick `78070`, seconds `7.50`, LSTM `0.8730`, delta `-0.0191`
- tick `78838`, seconds `19.50`, LSTM `0.8211`, delta `-0.0189`
- tick `77718`, seconds `2.00`, LSTM `0.7624`, delta `+0.0187`

## Top 15 local ridge features

- `lag_07__T_place_TSTAIRS`: coefficient `-0.001321`, |coef| `0.001321`
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `-0.001217`, |coef| `0.001217`
- `lag_00__closest_enemy_dist_diff`: coefficient `0.000866`, |coef| `0.000866`
- `lag_07__CT_B_site_active_infernos`: coefficient `-0.000857`, |coef| `0.000857`
- `lag_00__T_place_CANAL`: coefficient `-0.000819`, |coef| `0.000819`
- `lag_00__CT_kills_last_3s`: coefficient `0.000746`, |coef| `0.000746`
- `lag_08__T_place_TSTAIRS`: coefficient `-0.000730`, |coef| `0.000730`
- `lag_08__CT_place_CTSIDEUPPER`: coefficient `0.000723`, |coef| `0.000723`
- `lag_00__T5__alive`: coefficient `-0.000707`, |coef| `0.000707`
- `lag_00__T5__hp`: coefficient `-0.000695`, |coef| `0.000695`
- `lag_01__CT5__duck_amount`: coefficient `-0.000695`, |coef| `0.000695`
- `lag_00__T3__is_walking`: coefficient `-0.000675`, |coef| `0.000675`
- `lag_09__T1__duck_amount`: coefficient `-0.000652`, |coef| `0.000652`
- `lag_00__damage_diff_last_5s`: coefficient `0.000643`, |coef| `0.000643`
- `lag_06__T_place_TSTAIRS`: coefficient `-0.000639`, |coef| `0.000639`

## Top 10 utility ridge features

- `lag_07__CT_B_site_active_infernos`: coefficient `-0.000857` (lowers CT win probability)
- `lag_07__CT_active_infernos`: coefficient `-0.000566` (lowers CT win probability)
- `lag_09__CT1__smoke`: coefficient `-0.000557` (lowers CT win probability)
- `lag_01__T5__flash_duration`: coefficient `0.000482` (raises CT win probability)
- `lag_02__T5__flash_duration`: coefficient `0.000443` (raises CT win probability)
- `lag_08__CT_B_site_active_infernos`: coefficient `-0.000418` (lowers CT win probability)
- `lag_06__CT_B_site_active_smokes`: coefficient `0.000405` (raises CT win probability)
- `lag_07__active_infernos_total`: coefficient `-0.000370` (lowers CT win probability)
- `lag_06__CT_B_site_active_infernos`: coefficient `-0.000360` (lowers CT win probability)
- `lag_03__T5__flash_duration`: coefficient `0.000357` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_07__T_place_TSTAIRS`: coefficient `-0.001321` (lowers CT win probability)
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `-0.001217` (lowers CT win probability)
- `lag_00__closest_enemy_dist_diff`: coefficient `0.000866` (raises CT win probability)
- `lag_00__T_place_CANAL`: coefficient `-0.000819` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000746` (raises CT win probability)
- `lag_08__T_place_TSTAIRS`: coefficient `-0.000730` (lowers CT win probability)
- `lag_08__CT_place_CTSIDEUPPER`: coefficient `0.000723` (raises CT win probability)
- `lag_00__T5__alive`: coefficient `-0.000707` (lowers CT win probability)
- `lag_00__T5__hp`: coefficient `-0.000695` (lowers CT win probability)
- `lag_01__CT5__duck_amount`: coefficient `-0.000695` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `79062`, seconds `23.00`, LSTM delta `+0.1182`

Top all feature movements:
- `lag_07__T_place_TSTAIRS`: contribution `+0.007491`
- `lag_07__CT_B_site_active_infernos`: contribution `+0.002945`
- `lag_00__closest_enemy_dist_diff`: contribution `+0.002790`
- `lag_01__CT5__duck_amount`: contribution `+0.002623`
- `lag_09__T1__duck_amount`: contribution `+0.002551`

Top utility-only movements:
- `lag_07__CT_B_site_active_infernos`: contribution `+0.002945`

### tick `77846`, seconds `4.00`, LSTM delta `+0.0555`

Top all feature movements:
- `lag_08__CT_place_CTSIDEUPPER`: contribution `+0.018637`
- `lag_05__CT_place_CTSIDEUPPER`: contribution `+0.002307`
- `lag_03__CT_place_LOWERTUNNEL`: contribution `+0.002012`
- `lag_01__CT_place_PALACEINTERIOR`: contribution `+0.001927`
- `lag_00__CT_place_LOWERTUNNEL`: contribution `+0.001701`

Top utility-only movements:
- `lag_08__CT3__utility_total`: contribution `+0.000381`

### tick `77814`, seconds `3.50`, LSTM delta `+0.0342`

Top all feature movements:
- `lag_07__CT_place_CTSIDEUPPER`: contribution `+0.008601`
- `lag_02__CT_place_CTSIDEUPPER`: contribution `+0.004902`
- `lag_03__CT_place_LOWERTUNNEL`: contribution `+0.002012`
- `lag_00__CT_place_PALACEINTERIOR`: contribution `+0.001455`
- `lag_00__CT_place_MIDDLE`: contribution `+0.001356`

Top utility-only movements:
- `lag_07__CT3__utility_total`: contribution `+0.000259`

### tick `79222`, seconds `25.50`, LSTM delta `+0.0268`

Top all feature movements:
- `lag_01__T5__flash_duration`: contribution `+0.003472`
- `lag_00__CT_kills_last_3s`: contribution `+0.002152`
- `lag_06__CT5__duck_amount`: contribution `-0.001715`
- `lag_00__kill_diff_last_3s`: contribution `+0.001512`
- `lag_00__damage_diff_last_5s`: contribution `+0.001451`

Top utility-only movements:
- `lag_01__T5__flash_duration`: contribution `+0.003472`
- `lag_12__CT_B_site_active_infernos`: contribution `+0.000628`

### tick `78294`, seconds `11.00`, LSTM delta `-0.0240`

Top all feature movements:
- `lag_06__T_place_TSTAIRS`: contribution `-0.003624`
- `lag_00__T_place_CANAL`: contribution `-0.002276`
- `lag_01__CT_place_MAIN`: contribution `-0.002093`
- `lag_00__T_place_TSTAIRS`: contribution `-0.001781`
- `lag_00__T3__is_walking`: contribution `-0.001567`

Top utility-only movements:
- `lag_07__CT_active_infernos`: contribution `-0.001305`
- `lag_07__active_infernos_total`: contribution `-0.000532`
