# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-eternal-fire-vs-fluxo-bo3-Kqy3ohBVu1ANumI6Qdn26R/eternal-fire-vs-fluxo-m2-dust2.csv`
- round_num: `19`

## Largest probability jumps

- tick `146953`, seconds `12.00`, LSTM `0.6164`, delta `+0.2165`
- tick `147081`, seconds `14.00`, LSTM `0.8405`, delta `+0.1621`
- tick `149353`, seconds `49.50`, LSTM `0.9338`, delta `+0.0695`
- tick `147273`, seconds `17.00`, LSTM `0.8103`, delta `-0.0467`
- tick `146729`, seconds `8.50`, LSTM `0.3332`, delta `-0.0436`
- tick `148073`, seconds `29.50`, LSTM `0.7984`, delta `+0.0380`
- tick `146825`, seconds `10.00`, LSTM `0.3758`, delta `+0.0365`
- tick `147401`, seconds `19.00`, LSTM `0.8043`, delta `-0.0351`
- tick `146665`, seconds `7.50`, LSTM `0.3886`, delta `-0.0344`
- tick `148041`, seconds `29.00`, LSTM `0.7604`, delta `-0.0316`

## Top 15 local ridge features

- `lag_06__CT_place_HOLE`: coefficient `0.001586`, |coef| `0.001586`
- `lag_04__T2__is_scoped`: coefficient `0.001335`, |coef| `0.001335`
- `lag_03__T2__is_scoped`: coefficient `-0.001239`, |coef| `0.001239`
- `lag_01__CT_place_LONGDOORS`: coefficient `0.001224`, |coef| `0.001224`
- `lag_10__T_place_OUTSIDETUNNEL`: coefficient `-0.001224`, |coef| `0.001224`
- `lag_04__CT_place_HOLE`: coefficient `-0.001184`, |coef| `0.001184`
- `lag_10__CT_place_HOLE`: coefficient `0.001121`, |coef| `0.001121`
- `lag_00__CT_kills_last_3s`: coefficient `0.001116`, |coef| `0.001116`
- `lag_12__CT_place_LONGA`: coefficient `0.001114`, |coef| `0.001114`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001067`, |coef| `0.001067`
- `lag_09__CT4__flash_duration`: coefficient `0.000994`, |coef| `0.000994`
- `lag_14__CT_place_LONGA`: coefficient `0.000981`, |coef| `0.000981`
- `lag_00__damage_diff_last_5s`: coefficient `0.000979`, |coef| `0.000979`
- `lag_12__CT_place_UNDERA`: coefficient `-0.000951`, |coef| `0.000951`
- `lag_00__CT_damage_last_5s`: coefficient `0.000919`, |coef| `0.000919`

## Top 10 utility ridge features

- `lag_09__CT4__flash_duration`: coefficient `0.000994` (raises CT win probability)
- `lag_09__CT3__flash_duration`: coefficient `0.000887` (raises CT win probability)
- `lag_09__CT_flash_duration_sum`: coefficient `0.000874` (raises CT win probability)
- `lag_03__T1__flash_duration`: coefficient `0.000856` (raises CT win probability)
- `lag_00__T2__flash_duration`: coefficient `-0.000852` (lowers CT win probability)
- `lag_00__T2__flash`: coefficient `-0.000799` (lowers CT win probability)
- `lag_03__CT5__flash_duration`: coefficient `0.000755` (raises CT win probability)
- `lag_00__T_flash_duration_sum`: coefficient `-0.000710` (lowers CT win probability)
- `lag_03__T2__flash_duration`: coefficient `0.000695` (raises CT win probability)
- `lag_04__CT_active_infernos`: coefficient `0.000683` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_06__CT_place_HOLE`: coefficient `0.001586` (raises CT win probability)
- `lag_04__T2__is_scoped`: coefficient `0.001335` (raises CT win probability)
- `lag_03__T2__is_scoped`: coefficient `-0.001239` (lowers CT win probability)
- `lag_01__CT_place_LONGDOORS`: coefficient `0.001224` (raises CT win probability)
- `lag_10__T_place_OUTSIDETUNNEL`: coefficient `-0.001224` (lowers CT win probability)
- `lag_04__CT_place_HOLE`: coefficient `-0.001184` (lowers CT win probability)
- `lag_10__CT_place_HOLE`: coefficient `0.001121` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001116` (raises CT win probability)
- `lag_12__CT_place_LONGA`: coefficient `0.001114` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001067` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `146953`, seconds `12.00`, LSTM delta `+0.2165`

Top all feature movements:
- `lag_06__CT_place_HOLE`: contribution `+0.017703`
- `lag_04__CT_place_HOLE`: contribution `+0.013215`
- `lag_04__T2__is_scoped`: contribution `+0.011772`
- `lag_03__T2__is_scoped`: contribution `+0.010918`
- `lag_10__T_place_OUTSIDETUNNEL`: contribution `+0.006118`

Top utility-only movements:
- `lag_09__CT4__flash_duration`: contribution `+0.005679`
- `lag_09__CT3__flash_duration`: contribution `+0.004636`
- `lag_09__CT_flash_duration_sum`: contribution `+0.004279`
- `lag_00__T2__flash_duration`: contribution `+0.004169`
- `lag_03__T1__flash_duration`: contribution `+0.004136`

### tick `147081`, seconds `14.00`, LSTM delta `+0.1621`

Top all feature movements:
- `lag_10__CT_place_HOLE`: contribution `+0.012520`
- `lag_00__T_shots_fired_sum`: contribution `+0.006767`
- `lag_08__CT_place_HOLE`: contribution `+0.005808`
- `lag_08__T2__is_scoped`: contribution `+0.005581`
- `lag_02__T_place_TUNNELSTAIRS`: contribution `+0.005423`

Top utility-only movements:
- `lag_13__CT4__flash_duration`: contribution `+0.002569`
- `lag_13__CT3__flash_duration`: contribution `+0.002384`
- `lag_07__T2__flash_duration`: contribution `+0.002226`
- `lag_00__T1__flash_duration`: contribution `+0.002187`
- `lag_07__T1__flash_duration`: contribution `+0.002110`

### tick `149353`, seconds `49.50`, LSTM delta `+0.0695`

Top all feature movements:
- `lag_04__T_place_BDOORS`: contribution `+0.010161`
- `lag_00__T_place_BDOORS`: contribution `+0.009140`
- `lag_02__CT_place_HOLE`: contribution `+0.007799`
- `lag_00__CT_shots_fired_sum`: contribution `-0.005191`
- `lag_00__T_shots_fired_sum`: contribution `+0.003691`

Top utility-only movements:
- `lag_04__CT_B_site_active_infernos`: contribution `+0.001755`
- `lag_04__CT_active_infernos`: contribution `+0.001574`

### tick `147273`, seconds `17.00`, LSTM delta `-0.0467`

Top all feature movements:
- `lag_02__T_place_TUNNELSTAIRS`: contribution `-0.005423`
- `lag_14__CT_place_HOLE`: contribution `-0.005064`
- `lag_00__CT_kills_last_3s`: contribution `-0.003221`
- `lag_14__T2__is_scoped`: contribution `-0.003152`
- `lag_03__CT_flash_duration_sum`: contribution `-0.002517`

Top utility-only movements:
- `lag_03__CT_flash_duration_sum`: contribution `-0.002517`
- `lag_03__CT5__flash_duration`: contribution `-0.002329`
- `lag_06__T1__flash_duration`: contribution `-0.002073`
- `lag_04__CT_B_site_active_infernos`: contribution `-0.001755`
- `lag_04__CT_active_infernos`: contribution `-0.001574`

### tick `146729`, seconds `8.50`, LSTM delta `-0.0436`

Top all feature movements:
- `lag_10__T_place_OUTSIDETUNNEL`: contribution `-0.006118`
- `lag_13__CT_place_UNDERA`: contribution `-0.003272`
- `lag_03__CT_place_BDOORS`: contribution `-0.002010`
- `lag_02__CT3__flash_duration`: contribution `-0.001899`
- `lag_02__CT4__flash_duration`: contribution `-0.001899`

Top utility-only movements:
- `lag_02__CT3__flash_duration`: contribution `-0.001899`
- `lag_02__CT4__flash_duration`: contribution `-0.001899`
- `lag_02__CT_flash_duration_sum`: contribution `-0.001211`
- `lag_02__T4__flash_duration`: contribution `-0.001193`
