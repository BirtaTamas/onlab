# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-tyloo-vs-falcons-bo3-MBKGKnSCeuy54EHzS5mmW8/tyloo-vs-falcons-m2-ancient.csv`
- round_num: `5`

## Largest probability jumps

- tick `36918`, seconds `104.50`, LSTM `0.1827`, delta `-0.3623`
- tick `36470`, seconds `97.50`, LSTM `0.7294`, delta `+0.3074`
- tick `35734`, seconds `86.00`, LSTM `0.4576`, delta `-0.1941`
- tick `36566`, seconds `99.00`, LSTM `0.5818`, delta `-0.1764`
- tick `37270`, seconds `110.00`, LSTM `0.2151`, delta `+0.1509`
- tick `34102`, seconds `60.50`, LSTM `0.5023`, delta `+0.1187`
- tick `35414`, seconds `81.00`, LSTM `0.5902`, delta `+0.1053`
- tick `35766`, seconds `86.50`, LSTM `0.3959`, delta `-0.0617`
- tick `37302`, seconds `110.50`, LSTM `0.1580`, delta `-0.0571`
- tick `35350`, seconds `80.00`, LSTM `0.5092`, delta `-0.0553`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005122`, |coef| `0.005122`
- `lag_00__CT_kills_last_3s`: coefficient `0.003422`, |coef| `0.003422`
- `lag_00__damage_diff_last_5s`: coefficient `0.003196`, |coef| `0.003196`
- `lag_00__T_kills_last_3s`: coefficient `-0.002987`, |coef| `0.002987`
- `lag_00__T_place_RAMP`: coefficient `-0.002857`, |coef| `0.002857`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002638`, |coef| `0.002638`
- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.002635`, |coef| `0.002635`
- `lag_08__CT_place_TSIDEUPPER`: coefficient `-0.002437`, |coef| `0.002437`
- `lag_08__T_place_SIDEENTRANCE`: coefficient `0.002313`, |coef| `0.002313`
- `lag_13__T5__is_scoped`: coefficient `0.002253`, |coef| `0.002253`
- `lag_12__T_place_SIDEENTRANCE`: coefficient `0.002222`, |coef| `0.002222`
- `lag_00__CT_damage_last_5s`: coefficient `0.002214`, |coef| `0.002214`
- `lag_12__T4__is_walking`: coefficient `0.002192`, |coef| `0.002192`
- `lag_10__T5__duck_amount`: coefficient `0.002155`, |coef| `0.002155`
- `lag_08__CT_place_ALLEY`: coefficient `0.002034`, |coef| `0.002034`

## Top 10 utility ridge features

- `lag_05__T4__smoke`: coefficient `0.001959` (raises CT win probability)
- `lag_00__T4__smoke`: coefficient `-0.001564` (lowers CT win probability)
- `lag_03__T4__smoke`: coefficient `0.001456` (raises CT win probability)
- `lag_07__T_B_site_active_infernos`: coefficient `-0.001359` (lowers CT win probability)
- `lag_08__T_B_site_active_infernos`: coefficient `-0.001130` (lowers CT win probability)
- `lag_02__T5__molly`: coefficient `0.001074` (raises CT win probability)
- `lag_04__T4__smoke`: coefficient `0.001066` (raises CT win probability)
- `lag_12__CT_B_site_active_smokes`: coefficient `-0.001060` (lowers CT win probability)
- `lag_14__T4__smoke`: coefficient `0.000992` (raises CT win probability)
- `lag_07__T_active_infernos`: coefficient `-0.000991` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005122` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003422` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003196` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002987` (lowers CT win probability)
- `lag_00__T_place_RAMP`: coefficient `-0.002857` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002638` (raises CT win probability)
- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.002635` (raises CT win probability)
- `lag_08__CT_place_TSIDEUPPER`: coefficient `-0.002437` (lowers CT win probability)
- `lag_08__T_place_SIDEENTRANCE`: coefficient `0.002313` (raises CT win probability)
- `lag_13__T5__is_scoped`: coefficient `0.002253` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `36918`, seconds `104.50`, LSTM delta `-0.3623`

Top all feature movements:
- `lag_00__CT_place_TSIDEUPPER`: contribution `-0.019807`
- `lag_08__CT_place_TSIDEUPPER`: contribution `-0.018316`
- `lag_00__kill_diff_last_3s`: contribution `-0.012328`
- `lag_08__T_place_SIDEENTRANCE`: contribution `-0.011287`
- `lag_12__T_place_SIDEENTRANCE`: contribution `-0.010846`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `36470`, seconds `97.50`, LSTM delta `+0.3074`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.012328`
- `lag_13__T5__is_scoped`: contribution `+0.010744`
- `lag_00__T_place_RAMP`: contribution `+0.010103`
- `lag_00__CT_kills_last_3s`: contribution `+0.009881`
- `lag_00__CT_shots_fired_sum`: contribution `+0.009165`

Top utility-only movements:
- `lag_05__T4__smoke`: contribution `+0.004259`

### tick `35734`, seconds `86.00`, LSTM delta `-0.1941`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.012328`
- `lag_00__T_kills_last_3s`: contribution `-0.009462`
- `lag_00__damage_diff_last_5s`: contribution `-0.008437`
- `lag_00__T_shots_fired_sum`: contribution `-0.005599`
- `lag_00__CT_place_SIDEENTRANCE`: contribution `+0.005134`

Top utility-only movements:
- `lag_07__T_B_site_active_infernos`: contribution `-0.003842`
- `lag_03__T4__smoke`: contribution `-0.003166`
- `lag_10__T2__flash`: contribution `-0.002241`

### tick `36566`, seconds `99.00`, LSTM delta `-0.1764`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.012328`
- `lag_00__T_kills_last_3s`: contribution `-0.009462`
- `lag_11__T5__duck_amount`: contribution `-0.006853`
- `lag_12__T4__is_walking`: contribution `-0.005059`
- `lag_07__T4__is_walking`: contribution `-0.004177`

Top utility-only movements:
- `lag_03__T4__smoke`: contribution `-0.003166`

### tick `37270`, seconds `110.00`, LSTM delta `+0.1509`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.012328`
- `lag_00__CT_kills_last_3s`: contribution `+0.009881`
- `lag_00__damage_diff_last_5s`: contribution `+0.007932`
- `lag_11__CT_place_TSIDEUPPER`: contribution `+0.007907`
- `lag_00__CT_shots_fired_sum`: contribution `+0.007332`

Top utility-only movements:
- No utility movement among the top local contributors.
