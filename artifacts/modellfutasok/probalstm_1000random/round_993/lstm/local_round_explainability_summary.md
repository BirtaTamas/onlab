# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-mouz-vs-vitality-bo5-g3-5jFl1QSVPqll-eeCKIE/mouz-vs-vitality-m1-inferno.csv`
- round_num: `12`

## Largest probability jumps

- tick `116149`, seconds `32.50`, LSTM `0.2456`, delta `-0.1986`
- tick `116469`, seconds `37.50`, LSTM `0.0395`, delta `-0.0910`
- tick `116181`, seconds `33.00`, LSTM `0.1916`, delta `-0.0540`
- tick `114389`, seconds `5.00`, LSTM `0.3776`, delta `+0.0430`
- tick `116213`, seconds `33.50`, LSTM `0.1616`, delta `-0.0301`
- tick `114645`, seconds `9.00`, LSTM `0.4005`, delta `+0.0300`
- tick `115637`, seconds `24.50`, LSTM `0.3853`, delta `-0.0270`
- tick `118869`, seconds `75.00`, LSTM `0.0129`, delta `-0.0235`
- tick `114421`, seconds `5.50`, LSTM `0.4001`, delta `+0.0225`
- tick `114869`, seconds `12.50`, LSTM `0.3798`, delta `-0.0224`

## Top 15 local ridge features

- `lag_00__CT_place_RUINS`: coefficient `0.001973`, |coef| `0.001973`
- `lag_10__CT_place_BANANA`: coefficient `0.001906`, |coef| `0.001906`
- `lag_00__T_kills_last_3s`: coefficient `-0.001668`, |coef| `0.001668`
- `lag_00__CT_place_BANANA`: coefficient `0.001659`, |coef| `0.001659`
- `lag_14__CT1__duck_amount`: coefficient `0.001628`, |coef| `0.001628`
- `lag_08__T_place_BACKALLEY`: coefficient `-0.001606`, |coef| `0.001606`
- `lag_00__CT1__alive`: coefficient `0.001554`, |coef| `0.001554`
- `lag_14__T_B_site_active_infernos`: coefficient `0.001536`, |coef| `0.001536`
- `lag_00__CT1__hp`: coefficient `0.001532`, |coef| `0.001532`
- `lag_00__CT1__armor`: coefficient `0.001438`, |coef| `0.001438`
- `lag_08__CT_place_TOPOFMID`: coefficient `0.001438`, |coef| `0.001438`
- `lag_00__CT1__smoke`: coefficient `0.001379`, |coef| `0.001379`
- `lag_00__T_damage_last_5s`: coefficient `-0.001294`, |coef| `0.001294`
- `lag_13__T5__smoke`: coefficient `0.001286`, |coef| `0.001286`
- `lag_00__CT1__utility_total`: coefficient `0.001282`, |coef| `0.001282`

## Top 10 utility ridge features

- `lag_14__T_B_site_active_infernos`: coefficient `0.001536` (raises CT win probability)
- `lag_00__CT1__smoke`: coefficient `0.001379` (raises CT win probability)
- `lag_13__T5__smoke`: coefficient `0.001286` (raises CT win probability)
- `lag_00__CT1__utility_total`: coefficient `0.001282` (raises CT win probability)
- `lag_14__T_active_infernos`: coefficient `0.001147` (raises CT win probability)
- `lag_00__CT1__flash`: coefficient `0.001138` (raises CT win probability)
- `lag_04__CT5__smoke`: coefficient `0.001019` (raises CT win probability)
- `lag_15__CT4__flash`: coefficient `-0.000935` (lowers CT win probability)
- `lag_14__active_infernos_total`: coefficient `0.000880` (raises CT win probability)
- `lag_04__CT4__flash`: coefficient `0.000873` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_RUINS`: coefficient `0.001973` (raises CT win probability)
- `lag_10__CT_place_BANANA`: coefficient `0.001906` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001668` (lowers CT win probability)
- `lag_00__CT_place_BANANA`: coefficient `0.001659` (raises CT win probability)
- `lag_14__CT1__duck_amount`: coefficient `0.001628` (raises CT win probability)
- `lag_08__T_place_BACKALLEY`: coefficient `-0.001606` (lowers CT win probability)
- `lag_00__CT1__alive`: coefficient `0.001554` (raises CT win probability)
- `lag_00__CT1__hp`: coefficient `0.001532` (raises CT win probability)
- `lag_00__CT1__armor`: coefficient `0.001438` (raises CT win probability)
- `lag_08__CT_place_TOPOFMID`: coefficient `0.001438` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `116149`, seconds `32.50`, LSTM delta `-0.1986`

Top all feature movements:
- `lag_00__CT_place_RUINS`: contribution `-0.006894`
- `lag_14__CT1__duck_amount`: contribution `-0.006022`
- `lag_10__CT_place_BANANA`: contribution `-0.005642`
- `lag_00__T_kills_last_3s`: contribution `-0.005286`
- `lag_08__CT_place_TOPOFMID`: contribution `-0.005218`

Top utility-only movements:
- `lag_14__T_B_site_active_infernos`: contribution `-0.004342`
- `lag_00__CT1__smoke`: contribution `-0.002988`

### tick `116469`, seconds `37.50`, LSTM delta `-0.0910`

Top all feature movements:
- `lag_00__CT_place_RUINS`: contribution `-0.006894`
- `lag_10__CT_place_BANANA`: contribution `-0.005642`
- `lag_00__T_kills_last_3s`: contribution `-0.005286`
- `lag_00__kill_diff_last_3s`: contribution `-0.003051`
- `lag_00__CT5__duck_amount`: contribution `-0.002335`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `116181`, seconds `33.00`, LSTM delta `-0.0540`

Top all feature movements:
- `lag_01__CT_place_BANANA`: contribution `-0.003078`
- `lag_09__CT_place_TOPOFMID`: contribution `-0.002730`
- `lag_09__T_place_BACKALLEY`: contribution `-0.002301`
- `lag_01__CT1__alive`: contribution `-0.002166`
- `lag_12__CT_place_TOPOFMID`: contribution `-0.002143`

Top utility-only movements:
- `lag_15__T_B_site_active_infernos`: contribution `-0.001939`
- `lag_14__T5__smoke`: contribution `-0.001581`
- `lag_01__CT1__smoke`: contribution `-0.001504`

### tick `114389`, seconds `5.00`, LSTM delta `+0.0430`

Top all feature movements:
- `lag_00__CT_place_RUINS`: contribution `+0.006894`
- `lag_01__T_place_LOWERMID`: contribution `+0.006837`
- `lag_09__T_velocity_mean`: contribution `+0.001627`
- `lag_10__T_place_TSPAWN`: contribution `+0.001375`
- `lag_10__CT_macro_OTHER`: contribution `+0.001221`

Top utility-only movements:
- `lag_10__CT1__utility_total`: contribution `+0.001208`
- `lag_10__CT1__smoke`: contribution `+0.000833`
- `lag_10__CT_molly_inv`: contribution `+0.000706`
- `lag_10__CT1__molly`: contribution `+0.000703`

### tick `116213`, seconds `33.50`, LSTM delta `-0.0301`

Top all feature movements:
- `lag_02__CT_place_BANANA`: contribution `-0.002695`
- `lag_09__T1__is_walking`: contribution `+0.002225`
- `lag_04__CT3__is_walking`: contribution `-0.001775`
- `lag_03__CT2__is_walking`: contribution `-0.001764`
- `lag_02__CT1__alive`: contribution `-0.001536`

Top utility-only movements:
- No utility movement among the top local contributors.
