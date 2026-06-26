# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-mouz-vs-falcons-bo3-xBECUqZMcQ8GCwi-GUyz8e/mouz-vs-falcons-m1-train.csv`
- round_num: `3`

## Largest probability jumps

- tick `31208`, seconds `76.50`, LSTM `0.0963`, delta `-0.2177`
- tick `26984`, seconds `10.50`, LSTM `0.3751`, delta `-0.1895`
- tick `27016`, seconds `11.00`, LSTM `0.2619`, delta `-0.1132`
- tick `27176`, seconds `13.50`, LSTM `0.3327`, delta `+0.0936`
- tick `27432`, seconds `17.50`, LSTM `0.2964`, delta `-0.0665`
- tick `27304`, seconds `15.50`, LSTM `0.3659`, delta `+0.0639`
- tick `29864`, seconds `55.50`, LSTM `0.3200`, delta `-0.0546`
- tick `29512`, seconds `50.00`, LSTM `0.3807`, delta `+0.0516`
- tick `30728`, seconds `69.00`, LSTM `0.3601`, delta `+0.0487`
- tick `26664`, seconds `5.50`, LSTM `0.5642`, delta `+0.0484`

## Top 15 local ridge features

- `lag_02__CT5__flash_duration`: coefficient `-0.002932`, |coef| `0.002932`
- `lag_00__CT_place_TUNNELS`: coefficient `0.002844`, |coef| `0.002844`
- `lag_10__CT_flashes_last_5s`: coefficient `0.002670`, |coef| `0.002670`
- `lag_00__T_kills_last_3s`: coefficient `-0.002193`, |coef| `0.002193`
- `lag_00__CT5__flash_duration`: coefficient `0.002179`, |coef| `0.002179`
- `lag_00__CT5__alive`: coefficient `0.001892`, |coef| `0.001892`
- `lag_00__CT5__hp`: coefficient `0.001868`, |coef| `0.001868`
- `lag_02__T_place_IVY`: coefficient `0.001843`, |coef| `0.001843`
- `lag_00__T_damage_last_5s`: coefficient `-0.001831`, |coef| `0.001831`
- `lag_00__CT5__armor`: coefficient `0.001772`, |coef| `0.001772`
- `lag_00__damage_diff_last_5s`: coefficient `0.001754`, |coef| `0.001754`
- `lag_00__CT_place_ELECTRICALBOX`: coefficient `0.001736`, |coef| `0.001736`
- `lag_00__kill_diff_last_3s`: coefficient `0.001666`, |coef| `0.001666`
- `lag_00__T3__duck_amount`: coefficient `-0.001577`, |coef| `0.001577`
- `lag_15__T3__duck_amount`: coefficient `0.001564`, |coef| `0.001564`

## Top 10 utility ridge features

- `lag_02__CT5__flash_duration`: coefficient `-0.002932` (lowers CT win probability)
- `lag_10__CT_flashes_last_5s`: coefficient `0.002670` (raises CT win probability)
- `lag_00__CT5__flash_duration`: coefficient `0.002179` (raises CT win probability)
- `lag_02__T2__molly`: coefficient `0.001526` (raises CT win probability)
- `lag_02__T3__flash_duration`: coefficient `-0.001455` (lowers CT win probability)
- `lag_03__CT5__flash_duration`: coefficient `-0.001364` (lowers CT win probability)
- `lag_11__CT_flashes_last_5s`: coefficient `0.001300` (raises CT win probability)
- `lag_02__CT_flash_duration_sum`: coefficient `-0.001262` (lowers CT win probability)
- `lag_00__CT_A_site_active_infernos`: coefficient `0.001072` (raises CT win probability)
- `lag_00__CT5__smoke`: coefficient `0.001039` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_TUNNELS`: coefficient `0.002844` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002193` (lowers CT win probability)
- `lag_00__CT5__alive`: coefficient `0.001892` (raises CT win probability)
- `lag_00__CT5__hp`: coefficient `0.001868` (raises CT win probability)
- `lag_02__T_place_IVY`: coefficient `0.001843` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001831` (lowers CT win probability)
- `lag_00__CT5__armor`: coefficient `0.001772` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001754` (raises CT win probability)
- `lag_00__CT_place_ELECTRICALBOX`: coefficient `0.001736` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001666` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `31208`, seconds `76.50`, LSTM delta `-0.2177`

Top all feature movements:
- `lag_02__CT5__flash_duration`: contribution `-0.022498`
- `lag_00__CT5__flash_duration`: contribution `-0.016715`
- `lag_02__T_place_IVY`: contribution `-0.009849`
- `lag_00__CT_place_TUNNELS`: contribution `-0.008705`
- `lag_00__T_kills_last_3s`: contribution `-0.006948`

Top utility-only movements:
- `lag_02__CT5__flash_duration`: contribution `-0.022498`
- `lag_00__CT5__flash_duration`: contribution `-0.016715`
- `lag_02__T3__flash_duration`: contribution `-0.004471`
- `lag_02__CT_flash_duration_sum`: contribution `-0.004464`
- `lag_02__T2__molly`: contribution `-0.003401`

### tick `26984`, seconds `10.50`, LSTM delta `-0.1895`

Top all feature movements:
- `lag_10__CT_flashes_last_5s`: contribution `-0.058706`
- `lag_00__CT_place_ELECTRICALBOX`: contribution `-0.020176`
- `lag_03__CT5__flash_duration`: contribution `-0.007481`
- `lag_00__T_kills_last_3s`: contribution `-0.006948`
- `lag_03__T_place_DUMPSTER`: contribution `-0.006759`

Top utility-only movements:
- `lag_10__CT_flashes_last_5s`: contribution `-0.058706`
- `lag_03__CT5__flash_duration`: contribution `-0.007481`
- `lag_03__CT_flash_duration_sum`: contribution `-0.005244`

### tick `27016`, seconds `11.00`, LSTM delta `-0.1132`

Top all feature movements:
- `lag_11__CT_flashes_last_5s`: contribution `-0.028597`
- `lag_07__T_place_TSTAIRS`: contribution `-0.010396`
- `lag_06__T_place_DUMPSTER`: contribution `-0.009333`
- `lag_02__CT_place_ELECTRICALBOX`: contribution `-0.006783`
- `lag_07__T_place_TSIDEUPPER`: contribution `-0.006579`

Top utility-only movements:
- `lag_11__CT_flashes_last_5s`: contribution `-0.028597`
- `lag_04__CT5__flash_duration`: contribution `-0.003347`
- `lag_00__T_A_site_active_infernos`: contribution `-0.002831`
- `lag_04__CT_flash_duration_sum`: contribution `-0.002126`

### tick `27176`, seconds `13.50`, LSTM delta `+0.0936`

Top all feature movements:
- `lag_09__T_place_DUMPSTER`: contribution `+0.007802`
- `lag_00__T_kills_last_3s`: contribution `+0.006948`
- `lag_10__CT_place_CONNECTOR`: contribution `+0.006836`
- `lag_07__CT_place_ELECTRICALBOX`: contribution `+0.006653`
- `lag_15__T_place_TSTAIRS`: contribution `-0.005420`

Top utility-only movements:
- `lag_09__CT4__flash_duration`: contribution `+0.003450`
- `lag_09__CT_flash_duration_sum`: contribution `+0.002979`
- `lag_01__CT4__flash_duration`: contribution `+0.002050`

### tick `27432`, seconds `17.50`, LSTM delta `-0.0665`

Top all feature movements:
- `lag_14__CT_place_ELECTRICALBOX`: contribution `-0.008822`
- `lag_00__T4__duck_amount`: contribution `-0.004374`
- `lag_00__CT_A_site_active_infernos`: contribution `-0.003782`
- `lag_00__CT_B_site_active_infernos`: contribution `-0.003566`
- `lag_09__CT4__flash_duration`: contribution `-0.003450`

Top utility-only movements:
- `lag_00__CT_A_site_active_infernos`: contribution `-0.003782`
- `lag_00__CT_B_site_active_infernos`: contribution `-0.003566`
- `lag_09__CT4__flash_duration`: contribution `-0.003450`
- `lag_00__CT_active_infernos`: contribution `-0.001612`
