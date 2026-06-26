# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-spirit-vs-heroic-bo3-8PNegF4uXnTykkGvqzloIi/spirit-vs-heroic-m2-nuke.csv`
- round_num: `3`

## Largest probability jumps

- tick `18538`, seconds `24.00`, LSTM `0.1104`, delta `-0.1624`
- tick `22442`, seconds `85.00`, LSTM `0.2471`, delta `-0.0968`
- tick `22410`, seconds `84.50`, LSTM `0.3439`, delta `+0.0954`
- tick `22570`, seconds `87.00`, LSTM `0.3045`, delta `+0.0881`
- tick `21034`, seconds `63.00`, LSTM `0.2814`, delta `+0.0752`
- tick `18026`, seconds `16.00`, LSTM `0.2100`, delta `+0.0683`
- tick `22058`, seconds `79.00`, LSTM `0.2755`, delta `-0.0661`
- tick `19818`, seconds `44.00`, LSTM `0.0808`, delta `+0.0645`
- tick `19114`, seconds `33.00`, LSTM `0.0250`, delta `-0.0543`
- tick `17994`, seconds `15.50`, LSTM `0.1417`, delta `+0.0489`

## Top 15 local ridge features

- `lag_00__CT_duck_amount_mean`: coefficient `0.005345`, |coef| `0.005345`
- `lag_00__CT3__duck_amount`: coefficient `0.003676`, |coef| `0.003676`
- `lag_00__CT_velocity_mean`: coefficient `-0.002945`, |coef| `0.002945`
- `lag_10__CT_place_HUT`: coefficient `0.002394`, |coef| `0.002394`
- `lag_00__T_place_HUT`: coefficient `-0.002169`, |coef| `0.002169`
- `lag_00__CT3__is_walking`: coefficient `-0.002124`, |coef| `0.002124`
- `lag_05__CT_place_MINI`: coefficient `0.001994`, |coef| `0.001994`
- `lag_15__CT_place_HUTROOF`: coefficient `0.001884`, |coef| `0.001884`
- `lag_05__CT3__is_walking`: coefficient `0.001836`, |coef| `0.001836`
- `lag_09__T_velocity_mean`: coefficient `0.001675`, |coef| `0.001675`
- `lag_08__T_place_HUT`: coefficient `-0.001618`, |coef| `0.001618`
- `lag_01__T_place_TROPHY`: coefficient `0.001574`, |coef| `0.001574`
- `lag_00__T_place_SQUEAKY`: coefficient `-0.001537`, |coef| `0.001537`
- `lag_10__T_place_CONTROL`: coefficient `-0.001518`, |coef| `0.001518`
- `lag_07__T_place_TROPHY`: coefficient `-0.001514`, |coef| `0.001514`

## Top 10 utility ridge features

- `lag_03__T_A_site_active_smokes`: coefficient `-0.000653` (lowers CT win probability)
- `lag_13__T_A_site_active_infernos`: coefficient `0.000646` (raises CT win probability)
- `lag_03__T_B_site_active_smokes`: coefficient `-0.000641` (lowers CT win probability)
- `lag_14__T_A_site_active_infernos`: coefficient `0.000621` (raises CT win probability)
- `lag_13__T_B_site_active_infernos`: coefficient `0.000612` (raises CT win probability)
- `lag_14__T_B_site_active_infernos`: coefficient `0.000589` (raises CT win probability)
- `lag_00__T_A_site_active_infernos`: coefficient `-0.000569` (lowers CT win probability)
- `lag_10__T1__smoke`: coefficient `0.000551` (raises CT win probability)
- `lag_00__T_A_site_active_smokes`: coefficient `-0.000546` (lowers CT win probability)
- `lag_04__T_A_site_active_smokes`: coefficient `-0.000541` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_duck_amount_mean`: coefficient `0.005345` (raises CT win probability)
- `lag_00__CT3__duck_amount`: coefficient `0.003676` (raises CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.002945` (lowers CT win probability)
- `lag_10__CT_place_HUT`: coefficient `0.002394` (raises CT win probability)
- `lag_00__T_place_HUT`: coefficient `-0.002169` (lowers CT win probability)
- `lag_00__CT3__is_walking`: coefficient `-0.002124` (lowers CT win probability)
- `lag_05__CT_place_MINI`: coefficient `0.001994` (raises CT win probability)
- `lag_15__CT_place_HUTROOF`: coefficient `0.001884` (raises CT win probability)
- `lag_05__CT3__is_walking`: coefficient `0.001836` (raises CT win probability)
- `lag_09__T_velocity_mean`: coefficient `0.001675` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `18538`, seconds `24.00`, LSTM delta `-0.1624`

Top all feature movements:
- `lag_00__CT3__duck_amount`: contribution `-0.013396`
- `lag_05__CT_place_MINI`: contribution `-0.012224`
- `lag_10__T_place_CONTROL`: contribution `-0.010788`
- `lag_07__T_place_CONTROL`: contribution `-0.010683`
- `lag_01__T_place_TROPHY`: contribution `-0.009983`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `22442`, seconds `85.00`, LSTM delta `-0.0968`

Top all feature movements:
- `lag_00__CT_duck_amount_mean`: contribution `-0.032010`
- `lag_00__CT3__duck_amount`: contribution `-0.013677`
- `lag_12__T_place_HUT`: contribution `-0.009166`
- `lag_10__T_duck_amount_mean`: contribution `-0.006830`
- `lag_00__CT3__is_walking`: contribution `-0.005071`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `22410`, seconds `84.50`, LSTM delta `+0.0954`

Top all feature movements:
- `lag_00__CT_duck_amount_mean`: contribution `+0.029895`
- `lag_00__CT3__duck_amount`: contribution `+0.012773`
- `lag_10__T_duck_amount_mean`: contribution `+0.006830`
- `lag_11__T_place_HUT`: contribution `+0.006313`
- `lag_12__CT3__is_walking`: contribution `+0.003459`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `22570`, seconds `87.00`, LSTM delta `+0.0881`

Top all feature movements:
- `lag_00__CT_duck_amount_mean`: contribution `+0.030994`
- `lag_00__CT3__duck_amount`: contribution `+0.013243`
- `lag_00__CT_velocity_mean`: contribution `+0.008320`
- `lag_14__T_duck_amount_mean`: contribution `+0.003817`
- `lag_05__CT_duck_amount_mean`: contribution `+0.003709`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `21034`, seconds `63.00`, LSTM delta `+0.0752`

Top all feature movements:
- `lag_00__CT_duck_amount_mean`: contribution `+0.032010`
- `lag_00__CT3__duck_amount`: contribution `+0.013677`
- `lag_05__CT_place_HUTROOF`: contribution `+0.006921`
- `lag_00__CT3__is_walking`: contribution `+0.005071`
- `lag_10__T_duck_amount_mean`: contribution `+0.004779`

Top utility-only movements:
- No utility movement among the top local contributors.
