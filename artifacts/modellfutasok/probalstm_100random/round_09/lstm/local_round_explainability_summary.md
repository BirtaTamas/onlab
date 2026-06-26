# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-legacy-vs-nrg-bo3-_uO_eo-VIGwp_pYoaUs9Le/legacy-vs-nrg-m3-dust2.csv`
- round_num: `11`

## Largest probability jumps

- tick `82735`, seconds `23.00`, LSTM `0.5234`, delta `-0.2203`
- tick `82223`, seconds `15.00`, LSTM `0.6176`, delta `-0.1647`
- tick `85039`, seconds `59.00`, LSTM `0.0974`, delta `-0.1458`
- tick `82063`, seconds `12.50`, LSTM `0.7264`, delta `+0.1346`
- tick `82607`, seconds `21.00`, LSTM `0.8960`, delta `+0.1176`
- tick `82671`, seconds `22.00`, LSTM `0.7833`, delta `-0.1164`
- tick `82511`, seconds `19.50`, LSTM `0.7524`, delta `+0.1129`
- tick `83439`, seconds `34.00`, LSTM `0.4088`, delta `-0.0908`
- tick `83823`, seconds `40.00`, LSTM `0.2416`, delta `-0.0685`
- tick `82991`, seconds `27.00`, LSTM `0.5435`, delta `+0.0455`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003527`, |coef| `0.003527`
- `lag_00__T_kills_last_3s`: coefficient `-0.003191`, |coef| `0.003191`
- `lag_00__T_shots_fired_sum`: coefficient `-0.002861`, |coef| `0.002861`
- `lag_12__CT_place_SHORTSTAIRS`: coefficient `-0.002603`, |coef| `0.002603`
- `lag_15__T3__flash_duration`: coefficient `0.002512`, |coef| `0.002512`
- `lag_07__CT_place_EXTENDEDA`: coefficient `-0.002262`, |coef| `0.002262`
- `lag_00__T_damage_last_5s`: coefficient `-0.002044`, |coef| `0.002044`
- `lag_00__CT_place_EXTENDEDA`: coefficient `0.001971`, |coef| `0.001971`
- `lag_00__CT1__shots_fired`: coefficient `-0.001784`, |coef| `0.001784`
- `lag_00__damage_diff_last_5s`: coefficient `0.001633`, |coef| `0.001633`
- `lag_06__CT5__is_scoped`: coefficient `0.001619`, |coef| `0.001619`
- `lag_11__CT5__is_scoped`: coefficient `-0.001594`, |coef| `0.001594`
- `lag_02__T_place_SHORTSTAIRS`: coefficient `0.001588`, |coef| `0.001588`
- `lag_01__T_shots_fired_sum`: coefficient `-0.001575`, |coef| `0.001575`
- `lag_08__CT1__is_walking`: coefficient `-0.001562`, |coef| `0.001562`

## Top 10 utility ridge features

- `lag_15__T3__flash_duration`: coefficient `0.002512` (raises CT win probability)
- `lag_10__T3__flash_duration`: coefficient `0.001364` (raises CT win probability)
- `lag_00__CT1__smoke`: coefficient `0.001260` (raises CT win probability)
- `lag_11__T3__flash_duration`: coefficient `0.001217` (raises CT win probability)
- `lag_08__T5__flash_duration`: coefficient `0.001062` (raises CT win probability)
- `lag_15__T_flash_duration_sum`: coefficient `0.001054` (raises CT win probability)
- `lag_03__CT5__flash_duration`: coefficient `0.001030` (raises CT win probability)
- `lag_01__T2__flash_duration`: coefficient `-0.001014` (lowers CT win probability)
- `lag_00__T3__flash_duration`: coefficient `-0.001000` (lowers CT win probability)
- `lag_03__CT4__flash_duration`: coefficient `0.000924` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003527` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003191` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.002861` (lowers CT win probability)
- `lag_12__CT_place_SHORTSTAIRS`: coefficient `-0.002603` (lowers CT win probability)
- `lag_07__CT_place_EXTENDEDA`: coefficient `-0.002262` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002044` (lowers CT win probability)
- `lag_00__CT_place_EXTENDEDA`: coefficient `0.001971` (raises CT win probability)
- `lag_00__CT1__shots_fired`: coefficient `-0.001784` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001633` (raises CT win probability)
- `lag_06__CT5__is_scoped`: coefficient `0.001619` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `82735`, seconds `23.00`, LSTM delta `-0.2203`

Top all feature movements:
- `lag_00__CT_place_TUNNELSTAIRS`: contribution `-0.021617`
- `lag_00__CT_place_UPPERTUNNEL`: contribution `-0.011814`
- `lag_00__T_shots_fired_sum`: contribution `-0.010725`
- `lag_00__T_kills_last_3s`: contribution `-0.010110`
- `lag_00__kill_diff_last_3s`: contribution `-0.008489`

Top utility-only movements:
- `lag_01__T2__flash_duration`: contribution `-0.004961`
- `lag_01__T5__flash_duration`: contribution `-0.004332`

### tick `82223`, seconds `15.00`, LSTM delta `-0.1647`

Top all feature movements:
- `lag_12__CT_place_SHORTSTAIRS`: contribution `-0.014512`
- `lag_00__T_kills_last_3s`: contribution `-0.010110`
- `lag_00__kill_diff_last_3s`: contribution `-0.008489`
- `lag_15__CT_place_EXTENDEDA`: contribution `-0.007801`
- `lag_06__CT5__is_scoped`: contribution `-0.005790`

Top utility-only movements:
- `lag_03__CT4__flash_duration`: contribution `-0.005338`
- `lag_03__T5__flash_duration`: contribution `-0.004623`
- `lag_00__CT_B_site_active_infernos`: contribution `-0.002752`
- `lag_11__CT_B_site_active_infernos`: contribution `-0.001995`

### tick `85039`, seconds `59.00`, LSTM delta `-0.1458`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.012870`
- `lag_07__CT_place_EXTENDEDA`: contribution `-0.012699`
- `lag_00__CT_place_EXTENDEDA`: contribution `-0.011063`
- `lag_00__T_kills_last_3s`: contribution `-0.010110`
- `lag_00__kill_diff_last_3s`: contribution `-0.008489`

Top utility-only movements:
- `lag_00__CT1__smoke`: contribution `-0.002730`

### tick `82063`, seconds `12.50`, LSTM delta `+0.1346`

Top all feature movements:
- `lag_07__CT_place_EXTENDEDA`: contribution `+0.012699`
- `lag_00__kill_diff_last_3s`: contribution `+0.008489`
- `lag_08__T5__flash_duration`: contribution `+0.005803`
- `lag_06__CT5__is_scoped`: contribution `+0.005790`
- `lag_11__CT5__is_scoped`: contribution `+0.005701`

Top utility-only movements:
- `lag_08__T5__flash_duration`: contribution `+0.005803`
- `lag_11__CT4__flash_duration`: contribution `+0.004502`
- `lag_06__CT_B_site_active_infernos`: contribution `+0.001972`

### tick `82607`, seconds `21.00`, LSTM delta `+0.1176`

Top all feature movements:
- `lag_12__CT_place_SHORTSTAIRS`: contribution `+0.014512`
- `lag_05__T_place_SHORTSTAIRS`: contribution `+0.012967`
- `lag_10__CT_place_UPPERTUNNEL`: contribution `+0.010641`
- `lag_00__kill_diff_last_3s`: contribution `+0.008489`
- `lag_06__T_place_SHORTSTAIRS`: contribution `+0.004349`

Top utility-only movements:
- `lag_15__CT4__flash_duration`: contribution `+0.003281`
- `lag_15__T_flash_duration_sum`: contribution `-0.002339`
