# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-flyquest-vs-spirit-bo3-NmwBJVzYbgyZgcQrbNESHr/flyquest-vs-spirit-m1-anubis.csv`
- round_num: `10`

## Largest probability jumps

- tick `76980`, seconds `103.00`, LSTM `0.5772`, delta `-0.2838`
- tick `76148`, seconds `90.00`, LSTM `0.7326`, delta `+0.2663`
- tick `73428`, seconds `47.50`, LSTM `0.4611`, delta `+0.2080`
- tick `73172`, seconds `43.50`, LSTM `0.3185`, delta `-0.1817`
- tick `73140`, seconds `43.00`, LSTM `0.5003`, delta `+0.1467`
- tick `71156`, seconds `12.00`, LSTM `0.3412`, delta `-0.1427`
- tick `79540`, seconds `143.00`, LSTM `0.2015`, delta `+0.1010`
- tick `79636`, seconds `144.50`, LSTM `0.1217`, delta `-0.0992`
- tick `77204`, seconds `106.50`, LSTM `0.3853`, delta `+0.0949`
- tick `71124`, seconds `11.50`, LSTM `0.4839`, delta `-0.0911`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004904`, |coef| `0.004904`
- `lag_00__T_place_MAIN`: coefficient `-0.004390`, |coef| `0.004390`
- `lag_00__damage_diff_last_5s`: coefficient `0.004209`, |coef| `0.004209`
- `lag_00__CT_place_FOUNTAIN`: coefficient `-0.003757`, |coef| `0.003757`
- `lag_11__T_place_BRIDGE`: coefficient `0.003695`, |coef| `0.003695`
- `lag_00__CT_kills_last_3s`: coefficient `0.003437`, |coef| `0.003437`
- `lag_01__T4__is_scoped`: coefficient `-0.003342`, |coef| `0.003342`
- `lag_14__CT_place_WALKWAY`: coefficient `0.003289`, |coef| `0.003289`
- `lag_00__bomb_events_last_5s`: coefficient `0.003241`, |coef| `0.003241`
- `lag_13__bomb_events_last_5s`: coefficient `0.003178`, |coef| `0.003178`
- `lag_00__CT_velocity_mean`: coefficient `-0.003176`, |coef| `0.003176`
- `lag_02__T_kills_last_3s`: coefficient `-0.003123`, |coef| `0.003123`
- `lag_06__CT_place_WALKWAY`: coefficient `-0.003066`, |coef| `0.003066`
- `lag_09__CT_place_CONNECTOR`: coefficient `0.002862`, |coef| `0.002862`
- `lag_03__CT_place_CANAL`: coefficient `-0.002857`, |coef| `0.002857`

## Top 10 utility ridge features

- `lag_01__CT_A_site_active_infernos`: coefficient `0.002848` (raises CT win probability)
- `lag_00__T4__molly`: coefficient `-0.002233` (lowers CT win probability)
- `lag_13__CT_A_site_active_infernos`: coefficient `-0.002059` (lowers CT win probability)
- `lag_13__T4__molly`: coefficient `0.001982` (raises CT win probability)
- `lag_15__CT2__molly`: coefficient `0.001938` (raises CT win probability)
- `lag_01__CT_active_infernos`: coefficient `0.001872` (raises CT win probability)
- `lag_06__T_smokes_last_5s`: coefficient `0.001573` (raises CT win probability)
- `lag_00__CT2__flash`: coefficient `0.001550` (raises CT win probability)
- `lag_11__T_A_site_active_infernos`: coefficient `0.001471` (raises CT win probability)
- `lag_12__T_he_last_5s`: coefficient `0.001403` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004904` (raises CT win probability)
- `lag_00__T_place_MAIN`: coefficient `-0.004390` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.004209` (raises CT win probability)
- `lag_00__CT_place_FOUNTAIN`: coefficient `-0.003757` (lowers CT win probability)
- `lag_11__T_place_BRIDGE`: coefficient `0.003695` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003437` (raises CT win probability)
- `lag_01__T4__is_scoped`: coefficient `-0.003342` (lowers CT win probability)
- `lag_14__CT_place_WALKWAY`: coefficient `0.003289` (raises CT win probability)
- `lag_00__bomb_events_last_5s`: coefficient `0.003241` (raises CT win probability)
- `lag_13__bomb_events_last_5s`: coefficient `0.003178` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `76980`, seconds `103.00`, LSTM delta `-0.2838`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.011803`
- `lag_01__CT3__is_scoped`: contribution `-0.010827`
- `lag_01__CT_A_site_active_infernos`: contribution `-0.010051`
- `lag_02__T_kills_last_3s`: contribution `-0.009895`
- `lag_00__damage_diff_last_5s`: contribution `-0.009496`

Top utility-only movements:
- `lag_01__CT_A_site_active_infernos`: contribution `-0.010051`
- `lag_13__CT_A_site_active_infernos`: contribution `-0.007267`

### tick `76148`, seconds `90.00`, LSTM delta `+0.2663`

Top all feature movements:
- `lag_00__T_place_MAIN`: contribution `+0.028380`
- `lag_14__CT_place_WALKWAY`: contribution `+0.016145`
- `lag_01__T4__is_scoped`: contribution `+0.015523`
- `lag_06__CT_place_WALKWAY`: contribution `+0.015050`
- `lag_00__kill_diff_last_3s`: contribution `+0.011803`

Top utility-only movements:
- `lag_00__T4__molly`: contribution `+0.004868`

### tick `73428`, seconds `47.50`, LSTM delta `+0.2080`

Top all feature movements:
- `lag_00__T_place_MAIN`: contribution `+0.028380`
- `lag_00__kill_diff_last_3s`: contribution `+0.011803`
- `lag_00__CT_kills_last_3s`: contribution `+0.009924`
- `lag_02__T_kills_last_3s`: contribution `+0.009895`
- `lag_00__CT1__duck_amount`: contribution `+0.009346`

Top utility-only movements:
- `lag_11__T_A_site_active_infernos`: contribution `+0.004378`
- `lag_08__CT_A_site_active_infernos`: contribution `-0.003301`

### tick `73172`, seconds `43.50`, LSTM delta `-0.1817`

Top all feature movements:
- `lag_02__CT_place_MAIN`: contribution `-0.014220`
- `lag_00__kill_diff_last_3s`: contribution `-0.011803`
- `lag_00__CT_shots_fired_sum`: contribution `-0.011418`
- `lag_00__CT_place_MAIN`: contribution `-0.011224`
- `lag_12__CT_place_FOUNTAIN`: contribution `-0.009325`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `73140`, seconds `43.00`, LSTM delta `+0.1467`

Top all feature movements:
- `lag_00__T_place_MAIN`: contribution `+0.028380`
- `lag_01__CT_place_FOUNTAIN`: contribution `+0.014975`
- `lag_00__kill_diff_last_3s`: contribution `+0.011803`
- `lag_00__CT_kills_last_3s`: contribution `+0.009924`
- `lag_00__CT_shots_fired_sum`: contribution `+0.009515`

Top utility-only movements:
- No utility movement among the top local contributors.
