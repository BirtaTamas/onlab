# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-falcons-bo3-yayytstbo8IxTFlUpfbUPR/mouz-vs-falcons-m1-train.csv`
- round_num: `16`

## Largest probability jumps

- tick `136136`, seconds `35.50`, LSTM `0.3221`, delta `-0.1487`
- tick `136200`, seconds `36.50`, LSTM `0.2026`, delta `-0.0631`
- tick `136424`, seconds `40.00`, LSTM `0.0231`, delta `-0.0580`
- tick `134632`, seconds `12.00`, LSTM `0.3587`, delta `-0.0572`
- tick `136168`, seconds `36.00`, LSTM `0.2656`, delta `-0.0564`
- tick `136264`, seconds `37.50`, LSTM `0.1058`, delta `-0.0562`
- tick `134856`, seconds `15.50`, LSTM `0.4469`, delta `+0.0505`
- tick `136232`, seconds `37.00`, LSTM `0.1620`, delta `-0.0406`
- tick `134600`, seconds `11.50`, LSTM `0.4159`, delta `-0.0391`
- tick `134248`, seconds `6.00`, LSTM `0.4938`, delta `+0.0309`

## Top 15 local ridge features

- `lag_07__CT_place_ELECTRICALBOX`: coefficient `0.001710`, |coef| `0.001710`
- `lag_03__T_place_LONGDOG`: coefficient `-0.001473`, |coef| `0.001473`
- `lag_03__T_place_BACKOFB`: coefficient `0.001212`, |coef| `0.001212`
- `lag_07__T_place_IVY`: coefficient `-0.001136`, |coef| `0.001136`
- `lag_01__T_place_BACKOFB`: coefficient `0.001123`, |coef| `0.001123`
- `lag_05__T_place_LONGDOG`: coefficient `-0.001110`, |coef| `0.001110`
- `lag_04__T_place_LONGDOG`: coefficient `-0.001110`, |coef| `0.001110`
- `lag_02__CT_A_site_active_infernos`: coefficient `0.001007`, |coef| `0.001007`
- `lag_09__CT_place_ELECTRICALBOX`: coefficient `0.000981`, |coef| `0.000981`
- `lag_00__CT1__alive`: coefficient `0.000976`, |coef| `0.000976`
- `lag_05__T_place_BACKOFB`: coefficient `0.000975`, |coef| `0.000975`
- `lag_00__CT1__hp`: coefficient `0.000962`, |coef| `0.000962`
- `lag_10__CT_B_site_active_infernos`: coefficient `0.000939`, |coef| `0.000939`
- `lag_06__T_place_LONGDOG`: coefficient `-0.000915`, |coef| `0.000915`
- `lag_04__T_place_BACKOFB`: coefficient `0.000914`, |coef| `0.000914`

## Top 10 utility ridge features

- `lag_02__CT_A_site_active_infernos`: coefficient `0.001007` (raises CT win probability)
- `lag_10__CT_B_site_active_infernos`: coefficient `0.000939` (raises CT win probability)
- `lag_11__T_A_site_active_smokes`: coefficient `0.000792` (raises CT win probability)
- `lag_15__CT2__molly`: coefficient `0.000787` (raises CT win probability)
- `lag_06__CT_A_site_active_infernos`: coefficient `0.000775` (raises CT win probability)
- `lag_01__CT5__molly`: coefficient `0.000712` (raises CT win probability)
- `lag_10__CT_A_site_active_infernos`: coefficient `0.000712` (raises CT win probability)
- `lag_13__CT_he_last_5s`: coefficient `0.000711` (raises CT win probability)
- `lag_11__T_active_smokes`: coefficient `0.000701` (raises CT win probability)
- `lag_02__CT_active_infernos`: coefficient `0.000681` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_07__CT_place_ELECTRICALBOX`: coefficient `0.001710` (raises CT win probability)
- `lag_03__T_place_LONGDOG`: coefficient `-0.001473` (lowers CT win probability)
- `lag_03__T_place_BACKOFB`: coefficient `0.001212` (raises CT win probability)
- `lag_07__T_place_IVY`: coefficient `-0.001136` (lowers CT win probability)
- `lag_01__T_place_BACKOFB`: coefficient `0.001123` (raises CT win probability)
- `lag_05__T_place_LONGDOG`: coefficient `-0.001110` (lowers CT win probability)
- `lag_04__T_place_LONGDOG`: coefficient `-0.001110` (lowers CT win probability)
- `lag_09__CT_place_ELECTRICALBOX`: coefficient `0.000981` (raises CT win probability)
- `lag_00__CT1__alive`: coefficient `0.000976` (raises CT win probability)
- `lag_05__T_place_BACKOFB`: coefficient `0.000975` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `136136`, seconds `35.50`, LSTM delta `-0.1487`

Top all feature movements:
- `lag_07__CT_place_ELECTRICALBOX`: contribution `-0.019884`
- `lag_14__CT_place_ELECTRICALBOX`: contribution `-0.010045`
- `lag_03__T_place_LONGDOG`: contribution `-0.006854`
- `lag_07__T_place_IVY`: contribution `-0.006069`
- `lag_01__CT_place_ELECTRICALBOX`: contribution `-0.004421`

Top utility-only movements:
- `lag_02__CT_A_site_active_infernos`: contribution `-0.003553`
- `lag_10__CT_B_site_active_infernos`: contribution `-0.003225`
- `lag_10__CT_A_site_active_infernos`: contribution `-0.002513`
- `lag_11__T_A_site_active_smokes`: contribution `-0.002254`
- `lag_15__CT2__molly`: contribution `-0.001940`

### tick `136200`, seconds `36.50`, LSTM delta `-0.0631`

Top all feature movements:
- `lag_09__CT_place_ELECTRICALBOX`: contribution `-0.011403`
- `lag_14__CT_place_ELECTRICALBOX`: contribution `+0.010045`
- `lag_05__T_place_LONGDOG`: contribution `-0.005167`
- `lag_09__T_place_IVY`: contribution `-0.003916`
- `lag_03__T_place_BACKOFB`: contribution `-0.003254`

Top utility-only movements:
- `lag_04__CT_A_site_active_infernos`: contribution `-0.002218`
- `lag_12__CT_B_site_active_infernos`: contribution `-0.002127`
- `lag_13__T_A_site_active_smokes`: contribution `-0.001433`
- `lag_03__CT5__molly`: contribution `-0.001409`
- `lag_01__T5__molly`: contribution `-0.001396`

### tick `136424`, seconds `40.00`, LSTM delta `-0.0580`

Top all feature movements:
- `lag_07__CT_place_ELECTRICALBOX`: contribution `-0.019884`
- `lag_10__CT_place_ELECTRICALBOX`: contribution `+0.004605`
- `lag_00__CT_place_ELECTRICALBOX`: contribution `-0.002823`
- `lag_05__T_place_BACKOFB`: contribution `-0.002618`
- `lag_12__T_place_LONGDOG`: contribution `-0.002235`

Top utility-only movements:
- `lag_11__CT_A_site_active_infernos`: contribution `-0.001031`

### tick `134632`, seconds `12.00`, LSTM delta `-0.0572`

Top all feature movements:
- `lag_13__CT_he_last_5s`: contribution `-0.013038`
- `lag_11__T_he_last_5s`: contribution `-0.006665`
- `lag_02__T4__is_scoped`: contribution `-0.003109`
- `lag_00__CT_place_ELECTRICALBOX`: contribution `-0.002823`
- `lag_03__T_utility_damage_last_5s`: contribution `-0.002718`

Top utility-only movements:
- `lag_13__CT_he_last_5s`: contribution `-0.013038`
- `lag_11__T_he_last_5s`: contribution `-0.006665`
- `lag_03__T_utility_damage_last_5s`: contribution `-0.002718`
- `lag_03__utility_damage_diff_last_5s`: contribution `-0.001166`

### tick `136168`, seconds `36.00`, LSTM delta `-0.0564`

Top all feature movements:
- `lag_08__CT_place_ELECTRICALBOX`: contribution `-0.006248`
- `lag_04__T_place_LONGDOG`: contribution `-0.005163`
- `lag_02__CT_place_ELECTRICALBOX`: contribution `+0.003982`
- `lag_08__T_place_IVY`: contribution `-0.003737`
- `lag_15__CT_place_ELECTRICALBOX`: contribution `-0.002764`

Top utility-only movements:
- `lag_11__CT_B_site_active_infernos`: contribution `-0.001814`
- `lag_03__CT_A_site_active_infernos`: contribution `-0.001594`
- `lag_12__T_A_site_active_smokes`: contribution `-0.001474`
- `lag_02__CT5__molly`: contribution `-0.001314`
