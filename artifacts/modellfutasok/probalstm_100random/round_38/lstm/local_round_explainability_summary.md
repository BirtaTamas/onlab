# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-vitality-vs-the-mongolz-bo3-vhm_UWcBfYfYcOeLh9JIDA/vitality-vs-the-mongolz-m2-dust2.csv`
- round_num: `8`

## Largest probability jumps

- tick `65135`, seconds `76.50`, LSTM `0.5337`, delta `-0.2618`
- tick `64431`, seconds `65.50`, LSTM `0.7735`, delta `+0.2567`
- tick `60879`, seconds `10.00`, LSTM `0.6163`, delta `+0.1426`
- tick `64463`, seconds `66.00`, LSTM `0.9148`, delta `+0.1413`
- tick `60911`, seconds `10.50`, LSTM `0.4870`, delta `-0.1292`
- tick `64719`, seconds `70.00`, LSTM `0.8275`, delta `-0.1140`
- tick `63439`, seconds `50.00`, LSTM `0.5658`, delta `+0.0781`
- tick `64399`, seconds `65.00`, LSTM `0.5168`, delta `-0.0599`
- tick `65231`, seconds `78.00`, LSTM `0.5731`, delta `+0.0532`
- tick `64367`, seconds `64.50`, LSTM `0.5767`, delta `-0.0485`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003309`, |coef| `0.003309`
- `lag_05__T2__is_scoped`: coefficient `-0.003163`, |coef| `0.003163`
- `lag_00__T_place_EXTENDEDA`: coefficient `-0.003132`, |coef| `0.003132`
- `lag_00__T2__is_scoped`: coefficient `0.002970`, |coef| `0.002970`
- `lag_00__damage_diff_last_5s`: coefficient `0.002652`, |coef| `0.002652`
- `lag_13__T2__flash_duration`: coefficient `-0.002610`, |coef| `0.002610`
- `lag_00__CT_kills_last_3s`: coefficient `0.002479`, |coef| `0.002479`
- `lag_02__T_place_SHORTSTAIRS`: coefficient `-0.002427`, |coef| `0.002427`
- `lag_02__T_place_EXTENDEDA`: coefficient `0.002319`, |coef| `0.002319`
- `lag_06__CT2__is_walking`: coefficient `0.001904`, |coef| `0.001904`
- `lag_13__T_place_SHORTSTAIRS`: coefficient `0.001894`, |coef| `0.001894`
- `lag_08__CT_place_EXTENDEDA`: coefficient `0.001827`, |coef| `0.001827`
- `lag_00__T2__flash_duration`: coefficient `0.001732`, |coef| `0.001732`
- `lag_01__T_place_SHORTSTAIRS`: coefficient `-0.001666`, |coef| `0.001666`
- `lag_00__T_kills_last_3s`: coefficient `-0.001636`, |coef| `0.001636`

## Top 10 utility ridge features

- `lag_13__T2__flash_duration`: coefficient `-0.002610` (lowers CT win probability)
- `lag_00__T2__flash_duration`: coefficient `0.001732` (raises CT win probability)
- `lag_07__T_A_site_active_infernos`: coefficient `0.001556` (raises CT win probability)
- `lag_04__T_A_site_active_infernos`: coefficient `0.001261` (raises CT win probability)
- `lag_07__T_active_infernos`: coefficient `0.001171` (raises CT win probability)
- `lag_12__T_A_site_active_infernos`: coefficient `0.001138` (raises CT win probability)
- `lag_07__CT_flashes_last_5s`: coefficient `0.001084` (raises CT win probability)
- `lag_06__CT_flashes_last_5s`: coefficient `-0.000990` (lowers CT win probability)
- `lag_13__T_flash_duration_sum`: coefficient `-0.000933` (lowers CT win probability)
- `lag_07__active_infernos_total`: coefficient `0.000896` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003309` (raises CT win probability)
- `lag_05__T2__is_scoped`: coefficient `-0.003163` (lowers CT win probability)
- `lag_00__T_place_EXTENDEDA`: coefficient `-0.003132` (lowers CT win probability)
- `lag_00__T2__is_scoped`: coefficient `0.002970` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002652` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002479` (raises CT win probability)
- `lag_02__T_place_SHORTSTAIRS`: coefficient `-0.002427` (lowers CT win probability)
- `lag_02__T_place_EXTENDEDA`: coefficient `0.002319` (raises CT win probability)
- `lag_06__CT2__is_walking`: coefficient `0.001904` (raises CT win probability)
- `lag_13__T_place_SHORTSTAIRS`: coefficient `0.001894` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `65135`, seconds `76.50`, LSTM delta `-0.2618`

Top all feature movements:
- `lag_05__T2__is_scoped`: contribution `-0.027878`
- `lag_00__T2__is_scoped`: contribution `-0.026180`
- `lag_13__T2__flash_duration`: contribution `-0.018946`
- `lag_00__T2__flash_duration`: contribution `-0.012572`
- `lag_13__CT_place_EXTENDEDA`: contribution `-0.008779`

Top utility-only movements:
- `lag_13__T2__flash_duration`: contribution `-0.018946`
- `lag_00__T2__flash_duration`: contribution `-0.012572`
- `lag_12__T_A_site_active_infernos`: contribution `-0.003388`
- `lag_13__T_flash_duration_sum`: contribution `-0.002774`

### tick `64431`, seconds `65.50`, LSTM delta `+0.2567`

Top all feature movements:
- `lag_00__T_place_EXTENDEDA`: contribution `+0.015527`
- `lag_02__T_place_EXTENDEDA`: contribution `+0.011496`
- `lag_08__CT_place_EXTENDEDA`: contribution `+0.010254`
- `lag_02__T_place_SHORTSTAIRS`: contribution `+0.010201`
- `lag_00__kill_diff_last_3s`: contribution `+0.007966`

Top utility-only movements:
- `lag_07__T_A_site_active_infernos`: contribution `+0.004631`
- `lag_04__T_A_site_active_infernos`: contribution `+0.003753`

### tick `60879`, seconds `10.00`, LSTM delta `+0.1426`

Top all feature movements:
- `lag_13__CT_place_MIDDOORS`: contribution `+0.011252`
- `lag_06__CT_flashes_last_5s`: contribution `+0.010883`
- `lag_00__kill_diff_last_3s`: contribution `+0.007966`
- `lag_00__CT_kills_last_3s`: contribution `+0.007157`
- `lag_05__CT2__duck_amount`: contribution `+0.006162`

Top utility-only movements:
- `lag_06__CT_flashes_last_5s`: contribution `+0.010883`

### tick `64463`, seconds `66.00`, LSTM delta `+0.1413`

Top all feature movements:
- `lag_00__T_place_EXTENDEDA`: contribution `+0.015527`
- `lag_02__T_place_EXTENDEDA`: contribution `+0.011496`
- `lag_02__T_place_SHORTSTAIRS`: contribution `+0.010201`
- `lag_09__CT_place_EXTENDEDA`: contribution `+0.008210`
- `lag_00__kill_diff_last_3s`: contribution `+0.007966`

Top utility-only movements:
- `lag_05__T_A_site_active_infernos`: contribution `+0.002655`

### tick `60911`, seconds `10.50`, LSTM delta `-0.1292`

Top all feature movements:
- `lag_07__CT_flashes_last_5s`: contribution `-0.011918`
- `lag_00__kill_diff_last_3s`: contribution `-0.007966`
- `lag_14__CT_place_MIDDOORS`: contribution `-0.005959`
- `lag_00__T_kills_last_3s`: contribution `-0.005182`
- `lag_13__T1__is_scoped`: contribution `-0.004624`

Top utility-only movements:
- `lag_07__CT_flashes_last_5s`: contribution `-0.011918`
