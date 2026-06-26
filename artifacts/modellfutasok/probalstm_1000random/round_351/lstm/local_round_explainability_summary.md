# Local Round Explainability

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-falcons-vs-g2-bo3-Xf_UKx9fB2btv0Vy_VBVUC/falcons-vs-g2-m3-mirage.csv`
- round_num: `12`

## Largest probability jumps

- tick `77251`, seconds `23.00`, LSTM `0.5511`, delta `-0.3412`
- tick `77411`, seconds `25.50`, LSTM `0.3039`, delta `-0.2178`
- tick `77123`, seconds `21.00`, LSTM `0.8914`, delta `+0.1549`
- tick `77091`, seconds `20.50`, LSTM `0.7365`, delta `-0.1215`
- tick `77443`, seconds `26.00`, LSTM `0.1877`, delta `-0.1162`
- tick `76995`, seconds `19.00`, LSTM `0.8532`, delta `+0.1061`
- tick `79587`, seconds `59.50`, LSTM `0.1500`, delta `+0.1023`
- tick `79907`, seconds `64.50`, LSTM `0.2574`, delta `+0.0503`
- tick `77315`, seconds `24.00`, LSTM `0.5295`, delta `-0.0462`
- tick `77475`, seconds `26.50`, LSTM `0.1415`, delta `-0.0462`

## Top 15 local ridge features

- `lag_05__CT_shots_fired_sum`: coefficient `0.003183`, |coef| `0.003183`
- `lag_04__T_place_CONNECTOR`: coefficient `0.002202`, |coef| `0.002202`
- `lag_10__CT_shots_fired_sum`: coefficient `0.001997`, |coef| `0.001997`
- `lag_00__damage_diff_last_5s`: coefficient `0.001946`, |coef| `0.001946`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001866`, |coef| `0.001866`
- `lag_13__CT1__shots_fired`: coefficient `-0.001845`, |coef| `0.001845`
- `lag_12__CT1__shots_fired`: coefficient `-0.001818`, |coef| `0.001818`
- `lag_08__CT1__shots_fired`: coefficient `-0.001771`, |coef| `0.001771`
- `lag_00__kill_diff_last_3s`: coefficient `0.001728`, |coef| `0.001728`
- `lag_07__CT1__shots_fired`: coefficient `-0.001715`, |coef| `0.001715`
- `lag_08__CT_shots_fired_sum`: coefficient `-0.001644`, |coef| `0.001644`
- `lag_06__CT1__shots_fired`: coefficient `-0.001643`, |coef| `0.001643`
- `lag_11__CT1__shots_fired`: coefficient `-0.001633`, |coef| `0.001633`
- `lag_05__CT1__flash_duration`: coefficient `0.001489`, |coef| `0.001489`
- `lag_05__CT_place_SNIPERSNEST`: coefficient `0.001478`, |coef| `0.001478`

## Top 10 utility ridge features

- `lag_05__CT1__flash_duration`: coefficient `0.001489` (raises CT win probability)
- `lag_04__T1__flash_duration`: coefficient `0.001086` (raises CT win probability)
- `lag_10__CT1__flash_duration`: coefficient `0.001010` (raises CT win probability)
- `lag_13__T_flashes_last_5s`: coefficient `-0.000984` (lowers CT win probability)
- `lag_07__CT1__flash_duration`: coefficient `0.000950` (raises CT win probability)
- `lag_00__CT5__smoke`: coefficient `0.000877` (raises CT win probability)
- `lag_05__CT_B_site_active_infernos`: coefficient `0.000855` (raises CT win probability)
- `lag_11__CT1__flash_duration`: coefficient `-0.000797` (lowers CT win probability)
- `lag_00__T5__smoke`: coefficient `0.000778` (raises CT win probability)
- `lag_05__CT1__molly`: coefficient `0.000774` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_05__CT_shots_fired_sum`: coefficient `0.003183` (raises CT win probability)
- `lag_04__T_place_CONNECTOR`: coefficient `0.002202` (raises CT win probability)
- `lag_10__CT_shots_fired_sum`: coefficient `0.001997` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001946` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001866` (raises CT win probability)
- `lag_13__CT1__shots_fired`: coefficient `-0.001845` (lowers CT win probability)
- `lag_12__CT1__shots_fired`: coefficient `-0.001818` (lowers CT win probability)
- `lag_08__CT1__shots_fired`: coefficient `-0.001771` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001728` (raises CT win probability)
- `lag_07__CT1__shots_fired`: coefficient `-0.001715` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `77251`, seconds `23.00`, LSTM delta `-0.3412`

Top all feature movements:
- `lag_05__CT_shots_fired_sum`: contribution `-0.033175`
- `lag_08__CT_shots_fired_sum`: contribution `-0.011425`
- `lag_04__T_place_CONNECTOR`: contribution `-0.010665`
- `lag_05__CT_place_SNIPERSNEST`: contribution `-0.007917`
- `lag_05__CT1__flash_duration`: contribution `-0.007583`

Top utility-only movements:
- `lag_05__CT1__flash_duration`: contribution `-0.007583`
- `lag_11__CT1__flash_duration`: contribution `-0.004060`

### tick `77411`, seconds `25.50`, LSTM delta `-0.2178`

Top all feature movements:
- `lag_10__CT_shots_fired_sum`: contribution `-0.020813`
- `lag_10__CT_place_STAIRS`: contribution `-0.007641`
- `lag_10__CT_place_SNIPERSNEST`: contribution `-0.005392`
- `lag_10__CT1__flash_duration`: contribution `-0.005145`
- `lag_13__CT_shots_fired_sum`: contribution `-0.004922`

Top utility-only movements:
- `lag_10__CT1__flash_duration`: contribution `-0.005145`

### tick `77123`, seconds `21.00`, LSTM delta `+0.1549`

Top all feature movements:
- `lag_01__CT_shots_fired_sum`: contribution `+0.013687`
- `lag_04__T_place_CONNECTOR`: contribution `+0.010665`
- `lag_05__CT_shots_fired_sum`: contribution `+0.006635`
- `lag_01__CT_place_STAIRS`: contribution `+0.006116`
- `lag_04__CT_shots_fired_sum`: contribution `+0.005534`

Top utility-only movements:
- `lag_07__CT1__flash_duration`: contribution `+0.004837`
- `lag_01__CT1__flash_duration`: contribution `+0.002907`
- `lag_12__CT_B_site_active_infernos`: contribution `+0.002085`

### tick `77091`, seconds `20.50`, LSTM delta `-0.1215`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.019450`
- `lag_00__CT_place_STAIRS`: contribution `+0.007534`
- `lag_01__CT_shots_fired_sum`: contribution `-0.004562`
- `lag_00__kill_diff_last_3s`: contribution `-0.004158`
- `lag_00__T4__duck_amount`: contribution `-0.003579`

Top utility-only movements:
- `lag_06__CT1__flash_duration`: contribution `-0.002273`

### tick `77443`, seconds `26.00`, LSTM delta `-0.1162`

Top all feature movements:
- `lag_14__CT_shots_fired_sum`: contribution `-0.007464`
- `lag_06__CT_place_STAIRS`: contribution `-0.007464`
- `lag_13__CT1__shots_fired`: contribution `-0.004876`
- `lag_12__CT1__shots_fired`: contribution `-0.004804`
- `lag_00__kill_diff_last_3s`: contribution `+0.004158`

Top utility-only movements:
- `lag_11__CT1__flash_duration`: contribution `+0.004060`
