# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-faze-vs-heroic-dust2-PtQF8ASKD1754yZQHk6148/faze-vs-heroic-dust2.csv`
- round_num: `11`

## Largest probability jumps

- tick `82188`, seconds `21.00`, LSTM `0.7413`, delta `+0.1452`
- tick `82348`, seconds `23.50`, LSTM `0.8824`, delta `+0.1138`
- tick `83116`, seconds `35.50`, LSTM `0.9158`, delta `+0.0831`
- tick `82156`, seconds `20.50`, LSTM `0.5961`, delta `+0.0599`
- tick `82220`, seconds `21.50`, LSTM `0.7790`, delta `+0.0377`
- tick `82732`, seconds `29.50`, LSTM `0.8618`, delta `+0.0359`
- tick `80940`, seconds `1.50`, LSTM `0.5484`, delta `-0.0328`
- tick `84076`, seconds `50.50`, LSTM `0.9675`, delta `+0.0312`
- tick `82380`, seconds `24.00`, LSTM `0.8521`, delta `-0.0303`
- tick `82476`, seconds `25.50`, LSTM `0.8509`, delta `-0.0273`

## Top 15 local ridge features

- `lag_04__T1__flash_duration`: coefficient `0.001410`, |coef| `0.001410`
- `lag_03__T1__flash_duration`: coefficient `0.001408`, |coef| `0.001408`
- `lag_00__CT_kills_last_3s`: coefficient `0.001144`, |coef| `0.001144`
- `lag_02__T1__flash_duration`: coefficient `0.001040`, |coef| `0.001040`
- `lag_01__CT_place_CATWALK`: coefficient `-0.000972`, |coef| `0.000972`
- `lag_00__kill_diff_last_3s`: coefficient `0.000954`, |coef| `0.000954`
- `lag_00__CT_place_TOPOFMID`: coefficient `0.000941`, |coef| `0.000941`
- `lag_06__T5__duck_amount`: coefficient `-0.000887`, |coef| `0.000887`
- `lag_00__T_B_site_active_infernos`: coefficient `0.000876`, |coef| `0.000876`
- `lag_00__T1__flash`: coefficient `-0.000829`, |coef| `0.000829`
- `lag_04__T5__flash_duration`: coefficient `0.000824`, |coef| `0.000824`
- `lag_02__CT5__is_scoped`: coefficient `-0.000809`, |coef| `0.000809`
- `lag_00__T1__utility_total`: coefficient `-0.000787`, |coef| `0.000787`
- `lag_15__CT3__flash_duration`: coefficient `-0.000779`, |coef| `0.000779`
- `lag_04__T_flash_duration_sum`: coefficient `0.000764`, |coef| `0.000764`

## Top 10 utility ridge features

- `lag_04__T1__flash_duration`: coefficient `0.001410` (raises CT win probability)
- `lag_03__T1__flash_duration`: coefficient `0.001408` (raises CT win probability)
- `lag_02__T1__flash_duration`: coefficient `0.001040` (raises CT win probability)
- `lag_00__T_B_site_active_infernos`: coefficient `0.000876` (raises CT win probability)
- `lag_00__T1__flash`: coefficient `-0.000829` (lowers CT win probability)
- `lag_04__T5__flash_duration`: coefficient `0.000824` (raises CT win probability)
- `lag_00__T1__utility_total`: coefficient `-0.000787` (lowers CT win probability)
- `lag_15__CT3__flash_duration`: coefficient `-0.000779` (lowers CT win probability)
- `lag_04__T_flash_duration_sum`: coefficient `0.000764` (raises CT win probability)
- `lag_13__CT_B_site_active_infernos`: coefficient `-0.000706` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001144` (raises CT win probability)
- `lag_01__CT_place_CATWALK`: coefficient `-0.000972` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000954` (raises CT win probability)
- `lag_00__CT_place_TOPOFMID`: coefficient `0.000941` (raises CT win probability)
- `lag_06__T5__duck_amount`: coefficient `-0.000887` (lowers CT win probability)
- `lag_02__CT5__is_scoped`: coefficient `-0.000809` (lowers CT win probability)
- `lag_00__T1__alive`: coefficient `-0.000722` (lowers CT win probability)
- `lag_00__CT_place_CATWALK`: coefficient `-0.000716` (lowers CT win probability)
- `lag_06__CT_place_MIDDLE`: coefficient `0.000692` (raises CT win probability)
- `lag_01__CT1__duck_amount`: coefficient `0.000667` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `82188`, seconds `21.00`, LSTM delta `+0.1452`

Top all feature movements:
- `lag_03__T1__flash_duration`: contribution `+0.007484`
- `lag_01__CT_place_CATWALK`: contribution `+0.003870`
- `lag_00__CT_place_TOPOFMID`: contribution `+0.003414`
- `lag_06__T5__duck_amount`: contribution `+0.003369`
- `lag_00__CT_kills_last_3s`: contribution `+0.003303`

Top utility-only movements:
- `lag_03__T1__flash_duration`: contribution `+0.007484`
- `lag_15__CT3__flash_duration`: contribution `+0.003067`
- `lag_00__T_B_site_active_infernos`: contribution `+0.002476`
- `lag_13__CT_B_site_active_infernos`: contribution `+0.002425`
- `lag_00__T1__flash`: contribution `+0.002307`

### tick `82348`, seconds `23.50`, LSTM delta `+0.1138`

Top all feature movements:
- `lag_04__T1__flash_duration`: contribution `+0.009206`
- `lag_04__T5__flash_duration`: contribution `+0.006108`
- `lag_04__T_flash_duration_sum`: contribution `+0.005158`
- `lag_04__T_flashed_players`: contribution `+0.003851`
- `lag_00__CT_kills_last_3s`: contribution `+0.003303`

Top utility-only movements:
- `lag_04__T1__flash_duration`: contribution `+0.009206`
- `lag_04__T5__flash_duration`: contribution `+0.006108`
- `lag_04__T_flash_duration_sum`: contribution `+0.005158`
- `lag_04__T3__flash_duration`: contribution `+0.003011`
- `lag_08__T1__flash_duration`: contribution `+0.002678`

### tick `83116`, seconds `35.50`, LSTM delta `+0.0831`

Top all feature movements:
- `lag_01__T_place_HOLE`: contribution `+0.012425`
- `lag_02__T_place_HOLE`: contribution `+0.012104`
- `lag_02__CT_he_last_5s`: contribution `+0.008833`
- `lag_12__CT_he_last_5s`: contribution `+0.007818`
- `lag_00__CT_place_LOWERTUNNEL`: contribution `-0.004656`

Top utility-only movements:
- `lag_02__CT_he_last_5s`: contribution `+0.008833`
- `lag_12__CT_he_last_5s`: contribution `+0.007818`
- `lag_09__T_flashes_last_5s`: contribution `+0.002180`

### tick `82156`, seconds `20.50`, LSTM delta `+0.0599`

Top all feature movements:
- `lag_02__T1__flash_duration`: contribution `+0.005528`
- `lag_00__CT_place_CATWALK`: contribution `+0.002851`
- `lag_05__T5__duck_amount`: contribution `+0.002478`
- `lag_14__CT3__flash_duration`: contribution `+0.002396`
- `lag_01__CT5__is_scoped`: contribution `+0.002251`

Top utility-only movements:
- `lag_02__T1__flash_duration`: contribution `+0.005528`
- `lag_14__CT3__flash_duration`: contribution `+0.002396`
- `lag_12__CT_B_site_active_infernos`: contribution `+0.001522`
- `lag_12__CT_active_infernos`: contribution `+0.001143`
- `lag_06__T2__smoke`: contribution `+0.001014`

### tick `82220`, seconds `21.50`, LSTM delta `+0.0377`

Top all feature movements:
- `lag_04__T1__flash_duration`: contribution `+0.007498`
- `lag_00__CT_place_LOWERTUNNEL`: contribution `+0.004656`
- `lag_00__T3__flash_duration`: contribution `+0.003775`
- `lag_00__T5__flash_duration`: contribution `+0.002883`
- `lag_00__T_flashed_players`: contribution `-0.002747`

Top utility-only movements:
- `lag_04__T1__flash_duration`: contribution `+0.007498`
- `lag_00__T3__flash_duration`: contribution `+0.003775`
- `lag_00__T5__flash_duration`: contribution `+0.002883`
- `lag_04__T_flash_duration_sum`: contribution `+0.001690`
- `lag_00__T1__flash_duration`: contribution `-0.001492`
