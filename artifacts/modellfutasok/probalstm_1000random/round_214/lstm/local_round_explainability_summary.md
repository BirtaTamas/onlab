# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-3dmax-vs-vitality-nuke-h8drweGjLe5Dwjfuh5VfUb/3dmax-vs-vitality-nuke.csv`
- round_num: `3`

## Largest probability jumps

- tick `23799`, seconds `58.50`, LSTM `0.3549`, delta `-0.4535`
- tick `23703`, seconds `57.00`, LSTM `0.7088`, delta `+0.3277`
- tick `23607`, seconds `55.50`, LSTM `0.3594`, delta `+0.2596`
- tick `22455`, seconds `37.50`, LSTM `0.3366`, delta `-0.1749`
- tick `23991`, seconds `61.50`, LSTM `0.3057`, delta `+0.1349`
- tick `22487`, seconds `38.00`, LSTM `0.2150`, delta `-0.1217`
- tick `23895`, seconds `60.00`, LSTM `0.2255`, delta `-0.1072`
- tick `21847`, seconds `28.00`, LSTM `0.5042`, delta `-0.0955`
- tick `23767`, seconds `58.00`, LSTM `0.8084`, delta `+0.0748`
- tick `21815`, seconds `27.50`, LSTM `0.5997`, delta `+0.0746`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003494`, |coef| `0.003494`
- `lag_12__CT_place_MINI`: coefficient `-0.002860`, |coef| `0.002860`
- `lag_00__CT_place_VENTS`: coefficient `0.002624`, |coef| `0.002624`
- `lag_00__T_kills_last_3s`: coefficient `-0.002288`, |coef| `0.002288`
- `lag_09__T_place_CONTROL`: coefficient `-0.002283`, |coef| `0.002283`
- `lag_03__CT_place_VENTS`: coefficient `-0.002226`, |coef| `0.002226`
- `lag_06__T_place_CONTROL`: coefficient `0.002220`, |coef| `0.002220`
- `lag_12__T_place_CONTROL`: coefficient `-0.002139`, |coef| `0.002139`
- `lag_15__T_place_CONTROL`: coefficient `0.002122`, |coef| `0.002122`
- `lag_00__CT_kills_last_3s`: coefficient `0.002106`, |coef| `0.002106`
- `lag_11__T_place_VENDING`: coefficient `0.002021`, |coef| `0.002021`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001984`, |coef| `0.001984`
- `lag_09__T_kills_last_3s`: coefficient `0.001908`, |coef| `0.001908`
- `lag_12__CT_place_VENTS`: coefficient `-0.001801`, |coef| `0.001801`
- `lag_00__CT_place_ADMIN`: coefficient `0.001763`, |coef| `0.001763`

## Top 10 utility ridge features

- `lag_00__CT_flash_alpha_mean`: coefficient `0.001554` (raises CT win probability)
- `lag_00__CT_A_site_active_infernos`: coefficient `0.001109` (raises CT win probability)
- `lag_00__CT_B_site_active_infernos`: coefficient `0.001080` (raises CT win probability)
- `lag_06__CT_A_site_active_infernos`: coefficient `-0.001028` (lowers CT win probability)
- `lag_03__CT_A_site_active_infernos`: coefficient `0.001009` (raises CT win probability)
- `lag_06__CT_B_site_active_infernos`: coefficient `-0.000999` (lowers CT win probability)
- `lag_03__CT_B_site_active_infernos`: coefficient `0.000983` (raises CT win probability)
- `lag_00__CT1__flash`: coefficient `0.000963` (raises CT win probability)
- `lag_00__smoke_inv_diff`: coefficient `0.000946` (raises CT win probability)
- `lag_05__CT4__molly`: coefficient `-0.000889` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003494` (raises CT win probability)
- `lag_12__CT_place_MINI`: coefficient `-0.002860` (lowers CT win probability)
- `lag_00__CT_place_VENTS`: coefficient `0.002624` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002288` (lowers CT win probability)
- `lag_09__T_place_CONTROL`: coefficient `-0.002283` (lowers CT win probability)
- `lag_03__CT_place_VENTS`: coefficient `-0.002226` (lowers CT win probability)
- `lag_06__T_place_CONTROL`: coefficient `0.002220` (raises CT win probability)
- `lag_12__T_place_CONTROL`: coefficient `-0.002139` (lowers CT win probability)
- `lag_15__T_place_CONTROL`: coefficient `0.002122` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002106` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `23799`, seconds `58.50`, LSTM delta `-0.4535`

Top all feature movements:
- `lag_03__CT_place_VENTS`: contribution `-0.018676`
- `lag_12__CT_place_MINI`: contribution `-0.017535`
- `lag_00__kill_diff_last_3s`: contribution `-0.016819`
- `lag_09__T_place_CONTROL`: contribution `-0.016225`
- `lag_06__T_place_CONTROL`: contribution `-0.015773`

Top utility-only movements:
- `lag_00__CT_flash_alpha_mean`: contribution `-0.005964`

### tick `23703`, seconds `57.00`, LSTM delta `+0.3277`

Top all feature movements:
- `lag_00__CT_place_VENTS`: contribution `+0.022014`
- `lag_06__T_place_CONTROL`: contribution `+0.015773`
- `lag_12__T_place_CONTROL`: contribution `+0.015197`
- `lag_03__T_place_CONTROL`: contribution `+0.009703`
- `lag_00__kill_diff_last_3s`: contribution `+0.008409`

Top utility-only movements:
- `lag_03__CT_A_site_active_infernos`: contribution `+0.003559`

### tick `23607`, seconds `55.50`, LSTM delta `+0.2596`

Top all feature movements:
- `lag_12__CT_place_MINI`: contribution `+0.017535`
- `lag_09__T_place_CONTROL`: contribution `+0.016225`
- `lag_00__T_place_CONTROL`: contribution `+0.010249`
- `lag_03__T_place_CONTROL`: contribution `-0.009703`
- `lag_00__kill_diff_last_3s`: contribution `+0.008409`

Top utility-only movements:
- `lag_00__CT_A_site_active_infernos`: contribution `+0.003915`
- `lag_00__CT_B_site_active_infernos`: contribution `+0.003710`

### tick `22455`, seconds `37.50`, LSTM delta `-0.1749`

Top all feature movements:
- `lag_12__CT_place_VENTS`: contribution `-0.015113`
- `lag_00__CT_place_GARAGE`: contribution `-0.010932`
- `lag_12__CT_shots_fired_sum`: contribution `-0.010202`
- `lag_02__T_place_SILO`: contribution `-0.008575`
- `lag_00__kill_diff_last_3s`: contribution `-0.008409`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `23991`, seconds `61.50`, LSTM delta `+0.1349`

Top all feature movements:
- `lag_00__CT_place_SQUEAKY`: contribution `+0.016314`
- `lag_12__T_place_CONTROL`: contribution `+0.015197`
- `lag_15__T_place_CONTROL`: contribution `+0.015077`
- `lag_00__kill_diff_last_3s`: contribution `+0.008409`
- `lag_05__CT_place_VENTS`: contribution `+0.007387`

Top utility-only movements:
- No utility movement among the top local contributors.
