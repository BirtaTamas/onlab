# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-legacy-vs-lynn-vision-bo3-80tf5tBYONxHYQuFp0AoSQ/legacy-vs-lynn-vision-m1-dust2.csv`
- round_num: `10`

## Largest probability jumps

- tick `85648`, seconds `51.50`, LSTM `0.2423`, delta `-0.3411`
- tick `86064`, seconds `58.00`, LSTM `0.4528`, delta `+0.3400`
- tick `86352`, seconds `62.50`, LSTM `0.1781`, delta `-0.2856`
- tick `85200`, seconds `44.50`, LSTM `0.7847`, delta `+0.2409`
- tick `85232`, seconds `45.00`, LSTM `0.5871`, delta `-0.1976`
- tick `85680`, seconds `52.00`, LSTM `0.1762`, delta `-0.0661`
- tick `85808`, seconds `54.00`, LSTM `0.0174`, delta `-0.0596`
- tick `86384`, seconds `63.00`, LSTM `0.1211`, delta `-0.0569`
- tick `85776`, seconds `53.50`, LSTM `0.0770`, delta `-0.0468`
- tick `85424`, seconds `48.00`, LSTM `0.5661`, delta `+0.0439`

## Top 15 local ridge features

- `lag_00__damage_diff_last_5s`: coefficient `0.002839`, |coef| `0.002839`
- `lag_00__kill_diff_last_3s`: coefficient `0.002686`, |coef| `0.002686`
- `lag_13__CT_place_HOLE`: coefficient `-0.002146`, |coef| `0.002146`
- `lag_00__T_kills_last_3s`: coefficient `-0.002105`, |coef| `0.002105`
- `lag_05__CT_place_HOLE`: coefficient `-0.002104`, |coef| `0.002104`
- `lag_14__CT5__is_scoped`: coefficient `0.002028`, |coef| `0.002028`
- `lag_11__CT_place_BDOORS`: coefficient `-0.001889`, |coef| `0.001889`
- `lag_13__CT_place_LOWERTUNNEL`: coefficient `0.001871`, |coef| `0.001871`
- `lag_00__T_damage_last_5s`: coefficient `-0.001802`, |coef| `0.001802`
- `lag_14__T_place_TUNNELSTAIRS`: coefficient `0.001797`, |coef| `0.001797`
- `lag_08__T_place_TUNNELSTAIRS`: coefficient `0.001714`, |coef| `0.001714`
- `lag_02__CT_place_MIDDOORS`: coefficient `-0.001691`, |coef| `0.001691`
- `lag_01__CT_flashed_players`: coefficient `0.001653`, |coef| `0.001653`
- `lag_05__CT5__is_scoped`: coefficient `-0.001635`, |coef| `0.001635`
- `lag_10__CT_place_BDOORS`: coefficient `-0.001601`, |coef| `0.001601`

## Top 10 utility ridge features

- `lag_02__CT3__flash_duration`: coefficient `0.001409` (raises CT win probability)
- `lag_10__CT_B_site_active_infernos`: coefficient `0.000975` (raises CT win probability)
- `lag_04__T_B_site_active_infernos`: coefficient `0.000892` (raises CT win probability)
- `lag_07__CT_B_site_active_infernos`: coefficient `0.000843` (raises CT win probability)
- `lag_12__CT3__flash_duration`: coefficient `0.000803` (raises CT win probability)
- `lag_13__T_B_site_active_infernos`: coefficient `-0.000784` (lowers CT win probability)
- `lag_05__T2__molly`: coefficient `-0.000771` (lowers CT win probability)
- `lag_13__CT2__utility_total`: coefficient `0.000739` (raises CT win probability)
- `lag_10__CT5__molly`: coefficient `0.000717` (raises CT win probability)
- `lag_02__CT_flash_duration_sum`: coefficient `0.000693` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__damage_diff_last_5s`: coefficient `0.002839` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002686` (raises CT win probability)
- `lag_13__CT_place_HOLE`: coefficient `-0.002146` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002105` (lowers CT win probability)
- `lag_05__CT_place_HOLE`: coefficient `-0.002104` (lowers CT win probability)
- `lag_14__CT5__is_scoped`: coefficient `0.002028` (raises CT win probability)
- `lag_11__CT_place_BDOORS`: coefficient `-0.001889` (lowers CT win probability)
- `lag_13__CT_place_LOWERTUNNEL`: coefficient `0.001871` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001802` (lowers CT win probability)
- `lag_14__T_place_TUNNELSTAIRS`: coefficient `0.001797` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `85648`, seconds `51.50`, LSTM delta `-0.3411`

Top all feature movements:
- `lag_13__CT_place_HOLE`: contribution `-0.023958`
- `lag_10__CT_place_HOLE`: contribution `-0.015936`
- `lag_13__CT_place_LOWERTUNNEL`: contribution `-0.013753`
- `lag_14__T_place_TUNNELSTAIRS`: contribution `-0.012546`
- `lag_10__CT_place_BDOORS`: contribution `-0.007701`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `86064`, seconds `58.00`, LSTM delta `+0.3400`

Top all feature movements:
- `lag_08__T_place_TUNNELSTAIRS`: contribution `+0.011963`
- `lag_02__CT_place_MIDDOORS`: contribution `+0.009764`
- `lag_08__CT_place_SHORTSTAIRS`: contribution `+0.008462`
- `lag_03__T_place_TUNNELSTAIRS`: contribution `+0.008030`
- `lag_02__CT_place_BDOORS`: contribution `+0.007657`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `86352`, seconds `62.50`, LSTM delta `-0.2856`

Top all feature movements:
- `lag_05__CT_place_HOLE`: contribution `-0.023487`
- `lag_00__CT_place_HOLE`: contribution `-0.013267`
- `lag_14__T_place_TUNNELSTAIRS`: contribution `-0.012546`
- `lag_11__CT_place_BDOORS`: contribution `-0.009087`
- `lag_14__CT5__is_scoped`: contribution `-0.007252`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `85200`, seconds `44.50`, LSTM delta `+0.2409`

Top all feature movements:
- `lag_02__CT3__flash_duration`: contribution `+0.009065`
- `lag_00__CT_place_SHORTSTAIRS`: contribution `+0.007470`
- `lag_00__kill_diff_last_3s`: contribution `+0.006466`
- `lag_00__T_place_TUNNELSTAIRS`: contribution `+0.006414`
- `lag_04__CT_place_EXTENDEDA`: contribution `+0.006303`

Top utility-only movements:
- `lag_02__CT3__flash_duration`: contribution `+0.009065`

### tick `85232`, seconds `45.00`, LSTM delta `-0.1976`

Top all feature movements:
- `lag_00__CT_place_HOLE`: contribution `+0.013267`
- `lag_00__CT_shots_fired_sum`: contribution `-0.007246`
- `lag_00__T_kills_last_3s`: contribution `-0.006670`
- `lag_01__T_place_TUNNELSTAIRS`: contribution `+0.006621`
- `lag_00__kill_diff_last_3s`: contribution `-0.006466`

Top utility-only movements:
- `lag_03__CT3__flash_duration`: contribution `-0.003261`
