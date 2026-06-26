# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-heroic-vs-flyquest-bo3-ErQHzvBcWPHiA-H04IjPMf/heroic-vs-flyquest-m2-anubis.csv`
- round_num: `9`

## Largest probability jumps

- tick `68010`, seconds `10.50`, LSTM `0.1006`, delta `-0.0933`
- tick `68650`, seconds `20.50`, LSTM `0.0316`, delta `-0.0572`
- tick `67370`, seconds `0.50`, LSTM `0.0906`, delta `-0.0435`
- tick `67978`, seconds `10.00`, LSTM `0.1940`, delta `+0.0335`
- tick `67882`, seconds `8.50`, LSTM `0.1548`, delta `+0.0298`
- tick `68202`, seconds `13.50`, LSTM `0.0853`, delta `+0.0191`
- tick `68106`, seconds `12.00`, LSTM `0.0710`, delta `-0.0177`
- tick `68586`, seconds `19.50`, LSTM `0.0933`, delta `-0.0165`
- tick `68490`, seconds `18.00`, LSTM `0.0973`, delta `-0.0136`
- tick `68298`, seconds `15.00`, LSTM `0.0799`, delta `-0.0131`

## Top 15 local ridge features

- `lag_01__CT_place_CTSIDEUPPER`: coefficient `-0.000865`, |coef| `0.000865`
- `lag_02__CT_place_BRIDGE`: coefficient `-0.000724`, |coef| `0.000724`
- `lag_11__T_place_STREET`: coefficient `-0.000555`, |coef| `0.000555`
- `lag_12__CT_place_PALACEINTERIOR`: coefficient `0.000539`, |coef| `0.000539`
- `lag_02__CT_place_OUTSIDELONG`: coefficient `-0.000537`, |coef| `0.000537`
- `lag_12__T_utility_damage_last_5s`: coefficient `0.000464`, |coef| `0.000464`
- `lag_00__CT_place_OUTSIDELONG`: coefficient `0.000452`, |coef| `0.000452`
- `lag_11__CT_place_LOWERTUNNEL`: coefficient `0.000447`, |coef| `0.000447`
- `lag_00__T_kills_last_3s`: coefficient `-0.000445`, |coef| `0.000445`
- `lag_13__CT_place_LOWERTUNNEL`: coefficient `0.000424`, |coef| `0.000424`
- `lag_15__T_place_TSTAIRS`: coefficient `-0.000399`, |coef| `0.000399`
- `lag_05__T_place_STREET`: coefficient `0.000386`, |coef| `0.000386`
- `lag_12__T_place_STREET`: coefficient `-0.000377`, |coef| `0.000377`
- `lag_05__T_place_TSTAIRS`: coefficient `-0.000372`, |coef| `0.000372`
- `lag_01__T2__is_walking`: coefficient `0.000361`, |coef| `0.000361`

## Top 10 utility ridge features

- `lag_12__T_utility_damage_last_5s`: coefficient `0.000464` (raises CT win probability)
- `lag_02__T_utility_damage_last_5s`: coefficient `-0.000330` (lowers CT win probability)
- `lag_12__utility_damage_diff_last_5s`: coefficient `-0.000292` (lowers CT win probability)
- `lag_10__T_utility_damage_last_5s`: coefficient `0.000237` (raises CT win probability)
- `lag_02__utility_damage_diff_last_5s`: coefficient `0.000212` (raises CT win probability)
- `lag_02__CT5__flash_duration`: coefficient `-0.000209` (lowers CT win probability)
- `lag_01__T_utility_damage_last_5s`: coefficient `0.000204` (raises CT win probability)
- `lag_11__CT3__smoke`: coefficient `0.000192` (raises CT win probability)
- `lag_02__T2__flash_duration`: coefficient `-0.000183` (lowers CT win probability)
- `lag_09__CT5__smoke`: coefficient `0.000183` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_CTSIDEUPPER`: coefficient `-0.000865` (lowers CT win probability)
- `lag_02__CT_place_BRIDGE`: coefficient `-0.000724` (lowers CT win probability)
- `lag_11__T_place_STREET`: coefficient `-0.000555` (lowers CT win probability)
- `lag_12__CT_place_PALACEINTERIOR`: coefficient `0.000539` (raises CT win probability)
- `lag_02__CT_place_OUTSIDELONG`: coefficient `-0.000537` (lowers CT win probability)
- `lag_00__CT_place_OUTSIDELONG`: coefficient `0.000452` (raises CT win probability)
- `lag_11__CT_place_LOWERTUNNEL`: coefficient `0.000447` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.000445` (lowers CT win probability)
- `lag_13__CT_place_LOWERTUNNEL`: coefficient `0.000424` (raises CT win probability)
- `lag_15__T_place_TSTAIRS`: coefficient `-0.000399` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `68010`, seconds `10.50`, LSTM delta `-0.0933`

Top all feature movements:
- `lag_11__T_place_STREET`: contribution `-0.006107`
- `lag_02__CT_place_OUTSIDELONG`: contribution `-0.005448`
- `lag_00__CT_place_OUTSIDELONG`: contribution `-0.004580`
- `lag_12__CT_place_PALACEINTERIOR`: contribution `-0.004395`
- `lag_05__T_place_STREET`: contribution `-0.004240`

Top utility-only movements:
- `lag_02__T_utility_damage_last_5s`: contribution `-0.002262`

### tick `68650`, seconds `20.50`, LSTM delta `-0.0572`

Top all feature movements:
- `lag_02__CT_place_BRIDGE`: contribution `-0.008301`
- `lag_12__T_utility_damage_last_5s`: contribution `-0.003179`
- `lag_00__T_kills_last_3s`: contribution `-0.001409`
- `lag_08__T_place_BRIDGE`: contribution `-0.001355`
- `lag_12__utility_damage_diff_last_5s`: contribution `-0.001265`

Top utility-only movements:
- `lag_12__T_utility_damage_last_5s`: contribution `-0.003179`
- `lag_12__utility_damage_diff_last_5s`: contribution `-0.001265`
- `lag_02__CT5__flash_duration`: contribution `-0.000775`
- `lag_00__CT5__flash_duration`: contribution `-0.000593`

### tick `67370`, seconds `0.50`, LSTM delta `-0.0435`

Top all feature movements:
- `lag_01__CT_place_CTSIDEUPPER`: contribution `-0.022292`
- `lag_01__T_place_TSPAWN`: contribution `-0.000771`
- `lag_00__CT_velocity_mean`: contribution `-0.000650`
- `lag_00__T_velocity_mean`: contribution `-0.000515`
- `lag_01__T_closest_enemy_dist`: contribution `-0.000452`

Top utility-only movements:
- `lag_01__utility_inv_diff`: contribution `-0.000394`
- `lag_01__flash_inv_diff`: contribution `-0.000342`
- `lag_01__molly_inv_diff`: contribution `-0.000312`
- `lag_01__T5__utility_total`: contribution `-0.000310`
- `lag_01__T5__flash`: contribution `-0.000257`

### tick `67978`, seconds `10.00`, LSTM delta `+0.0335`

Top all feature movements:
- `lag_15__CT_place_CTSIDEUPPER`: contribution `+0.003949`
- `lag_04__T_place_TSTAIRS`: contribution `+0.003611`
- `lag_11__T_place_STREET`: contribution `-0.003053`
- `lag_01__CT_place_OUTSIDELONG`: contribution `+0.002467`
- `lag_10__T_place_STREET`: contribution `+0.002458`

Top utility-only movements:
- `lag_01__T_utility_damage_last_5s`: contribution `+0.001401`

### tick `67882`, seconds `8.50`, LSTM delta `+0.0298`

Top all feature movements:
- `lag_12__CT_place_PALACEINTERIOR`: contribution `+0.004395`
- `lag_07__T_place_STREET`: contribution `+0.002535`
- `lag_01__T_place_STREET`: contribution `+0.001978`
- `lag_01__T_place_TSTAIRS`: contribution `+0.001683`
- `lag_14__CT_place_LOWERTUNNEL`: contribution `+0.001480`

Top utility-only movements:
- No utility movement among the top local contributors.
