# Local Round Explainability

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-faze-vs-bcgame-bo3-daIGc_M_y7Qq42fF9AoQhi/faze-vs-bcgame-m2-anubis.csv`
- round_num: `15`

## Largest probability jumps

- tick `110603`, seconds `95.00`, LSTM `0.6692`, delta `+0.2547`
- tick `108459`, seconds `61.50`, LSTM `0.4000`, delta `-0.2446`
- tick `110923`, seconds `100.00`, LSTM `0.8562`, delta `+0.1537`
- tick `108427`, seconds `61.00`, LSTM `0.6445`, delta `+0.1286`
- tick `108395`, seconds `60.50`, LSTM `0.5160`, delta `-0.1181`
- tick `105643`, seconds `17.50`, LSTM `0.4887`, delta `-0.1126`
- tick `108715`, seconds `65.50`, LSTM `0.4623`, delta `+0.0875`
- tick `110891`, seconds `99.50`, LSTM `0.7025`, delta `+0.0686`
- tick `105931`, seconds `22.00`, LSTM `0.4799`, delta `+0.0663`
- tick `104779`, seconds `4.00`, LSTM `0.6461`, delta `+0.0542`

## Top 15 local ridge features

- `lag_00__CT_place_BRICKS`: coefficient `0.003455`, |coef| `0.003455`
- `lag_00__kill_diff_last_3s`: coefficient `0.002834`, |coef| `0.002834`
- `lag_01__T_place_CONNECTOR`: coefficient `-0.002493`, |coef| `0.002493`
- `lag_00__CT_kills_last_3s`: coefficient `0.002455`, |coef| `0.002455`
- `lag_15__T_place_MAIN`: coefficient `0.002219`, |coef| `0.002219`
- `lag_00__CT_place_PALACEINTERIOR`: coefficient `-0.002211`, |coef| `0.002211`
- `lag_01__CT_place_WALKWAY`: coefficient `0.001876`, |coef| `0.001876`
- `lag_04__T_place_CONNECTOR`: coefficient `0.001866`, |coef| `0.001866`
- `lag_00__T_spread_xy`: coefficient `-0.001753`, |coef| `0.001753`
- `lag_11__CT_place_CANAL`: coefficient `-0.001749`, |coef| `0.001749`
- `lag_03__CT5__is_walking`: coefficient `-0.001737`, |coef| `0.001737`
- `lag_14__T_place_CONNECTOR`: coefficient `0.001719`, |coef| `0.001719`
- `lag_04__T4__duck_amount`: coefficient `-0.001683`, |coef| `0.001683`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001664`, |coef| `0.001664`
- `lag_06__CT_place_TUNNEL`: coefficient `-0.001547`, |coef| `0.001547`

## Top 10 utility ridge features

- `lag_02__CT2__flash_duration`: coefficient `0.001073` (raises CT win probability)
- `lag_03__T3__flash_duration`: coefficient `-0.000919` (lowers CT win probability)
- `lag_00__T3__smoke`: coefficient `-0.000767` (lowers CT win probability)
- `lag_03__CT2__flash_duration`: coefficient `-0.000767` (lowers CT win probability)
- `lag_14__CT_A_site_active_infernos`: coefficient `-0.000651` (lowers CT win probability)
- `lag_01__CT2__flash_duration`: coefficient `-0.000638` (lowers CT win probability)
- `lag_15__CT_active_infernos`: coefficient `-0.000615` (lowers CT win probability)
- `lag_15__CT_A_site_active_infernos`: coefficient `-0.000593` (lowers CT win probability)
- `lag_00__T3__utility_total`: coefficient `-0.000566` (lowers CT win probability)
- `lag_07__CT_B_site_active_infernos`: coefficient `-0.000562` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_BRICKS`: coefficient `0.003455` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002834` (raises CT win probability)
- `lag_01__T_place_CONNECTOR`: coefficient `-0.002493` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002455` (raises CT win probability)
- `lag_15__T_place_MAIN`: coefficient `0.002219` (raises CT win probability)
- `lag_00__CT_place_PALACEINTERIOR`: coefficient `-0.002211` (lowers CT win probability)
- `lag_01__CT_place_WALKWAY`: coefficient `0.001876` (raises CT win probability)
- `lag_04__T_place_CONNECTOR`: coefficient `0.001866` (raises CT win probability)
- `lag_00__T_spread_xy`: coefficient `-0.001753` (lowers CT win probability)
- `lag_11__CT_place_CANAL`: coefficient `-0.001749` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `110603`, seconds `95.00`, LSTM delta `+0.2547`

Top all feature movements:
- `lag_00__CT_place_BRICKS`: contribution `+0.066343`
- `lag_15__T_place_MAIN`: contribution `+0.014348`
- `lag_01__T_place_CONNECTOR`: contribution `+0.012071`
- `lag_04__T_place_CONNECTOR`: contribution `+0.009037`
- `lag_00__CT_place_PALACEINTERIOR`: contribution `+0.009011`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `108459`, seconds `61.50`, LSTM delta `-0.2446`

Top all feature movements:
- `lag_06__CT_place_TUNNEL`: contribution `-0.024841`
- `lag_06__CT_place_TUNNELSTAIRS`: contribution `-0.021560`
- `lag_11__CT_place_CANAL`: contribution `-0.021259`
- `lag_01__CT_place_TUNNEL`: contribution `-0.015370`
- `lag_11__CT_place_MAIN`: contribution `-0.013479`

Top utility-only movements:
- `lag_02__CT2__flash_duration`: contribution `-0.006897`
- `lag_03__T3__flash_duration`: contribution `-0.005567`
- `lag_03__CT2__flash_duration`: contribution `-0.004927`

### tick `110923`, seconds `100.00`, LSTM delta `+0.1537`

Top all feature movements:
- `lag_10__CT_place_BRICKS`: contribution `+0.019878`
- `lag_08__CT_place_BRICKS`: contribution `+0.016590`
- `lag_00__CT_place_PALACEINTERIOR`: contribution `+0.009011`
- `lag_14__T_place_CONNECTOR`: contribution `+0.008324`
- `lag_00__CT_kills_last_3s`: contribution `+0.007088`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `108427`, seconds `61.00`, LSTM delta `+0.1286`

Top all feature movements:
- `lag_10__CT_place_TUNNELSTAIRS`: contribution `+0.008055`
- `lag_00__CT_kills_last_3s`: contribution `+0.007088`
- `lag_00__CT_shots_fired_sum`: contribution `+0.006936`
- `lag_02__CT2__flash_duration`: contribution `+0.006897`
- `lag_00__kill_diff_last_3s`: contribution `+0.006821`

Top utility-only movements:
- `lag_02__CT2__flash_duration`: contribution `+0.006897`
- `lag_01__CT2__flash_duration`: contribution `+0.004099`
- `lag_02__T3__flash_duration`: contribution `+0.002383`

### tick `108395`, seconds `60.50`, LSTM delta `-0.1181`

Top all feature movements:
- `lag_04__CT_place_TUNNELSTAIRS`: contribution `-0.019568`
- `lag_09__CT_place_MAIN`: contribution `-0.008167`
- `lag_09__CT_place_CANAL`: contribution `-0.007442`
- `lag_00__CT_place_CANAL`: contribution `-0.007141`
- `lag_00__kill_diff_last_3s`: contribution `-0.006821`

Top utility-only movements:
- `lag_01__CT2__flash_duration`: contribution `-0.004099`
- `lag_00__CT2__flash_duration`: contribution `-0.002636`
- `lag_01__T3__flash_duration`: contribution `-0.002500`
