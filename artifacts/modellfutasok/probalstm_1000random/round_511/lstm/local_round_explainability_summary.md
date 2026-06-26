# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-inner-circle-vs-gentle-mates-bo3-u31MSfrH-KJtKM4rM-4jj7/inner-circle-vs-gentle-mates-m1-nuke.csv`
- round_num: `6`

## Largest probability jumps

- tick `45333`, seconds `69.00`, LSTM `0.2765`, delta `-0.2277`
- tick `43765`, seconds `44.50`, LSTM `0.7701`, delta `+0.1844`
- tick `44149`, seconds `50.50`, LSTM `0.5176`, delta `-0.1576`
- tick `45397`, seconds `70.00`, LSTM `0.1327`, delta `-0.0978`
- tick `45269`, seconds `68.00`, LSTM `0.5595`, delta `+0.0954`
- tick `44469`, seconds `55.50`, LSTM `0.4420`, delta `+0.0602`
- tick `46389`, seconds `85.50`, LSTM `0.0382`, delta `-0.0594`
- tick `45141`, seconds `66.00`, LSTM `0.4575`, delta `-0.0559`
- tick `45301`, seconds `68.50`, LSTM `0.5043`, delta `-0.0553`
- tick `44309`, seconds `53.00`, LSTM `0.3672`, delta `-0.0480`

## Top 15 local ridge features

- `lag_06__CT_place_OBSERVATION`: coefficient `-0.003100`, |coef| `0.003100`
- `lag_01__CT_place_OBSERVATION`: coefficient `0.001796`, |coef| `0.001796`
- `lag_00__kill_diff_last_3s`: coefficient `0.001681`, |coef| `0.001681`
- `lag_03__T_place_HELL`: coefficient `0.001542`, |coef| `0.001542`
- `lag_00__damage_diff_last_5s`: coefficient `0.001481`, |coef| `0.001481`
- `lag_09__CT_place_HUTROOF`: coefficient `0.001461`, |coef| `0.001461`
- `lag_04__CT_place_OBSERVATION`: coefficient `0.001408`, |coef| `0.001408`
- `lag_08__T_place_TROPHY`: coefficient `0.001404`, |coef| `0.001404`
- `lag_03__CT2__is_scoped`: coefficient `-0.001340`, |coef| `0.001340`
- `lag_08__CT_place_OBSERVATION`: coefficient `-0.001309`, |coef| `0.001309`
- `lag_07__CT2__is_scoped`: coefficient `-0.001298`, |coef| `0.001298`
- `lag_00__T_kills_last_3s`: coefficient `-0.001296`, |coef| `0.001296`
- `lag_11__CT5__is_walking`: coefficient `0.001295`, |coef| `0.001295`
- `lag_03__CT_place_OBSERVATION`: coefficient `0.001271`, |coef| `0.001271`
- `lag_09__T_place_ADMIN`: coefficient `-0.001231`, |coef| `0.001231`

## Top 10 utility ridge features

- `lag_15__CT5__flash_duration`: coefficient `-0.000820` (lowers CT win probability)
- `lag_03__CT5__smoke`: coefficient `0.000723` (raises CT win probability)
- `lag_15__CT_A_site_active_infernos`: coefficient `-0.000565` (lowers CT win probability)
- `lag_09__T_A_site_active_smokes`: coefficient `0.000468` (raises CT win probability)
- `lag_00__CT_molly_inv`: coefficient `0.000461` (raises CT win probability)
- `lag_06__CT5__flash_duration`: coefficient `0.000454` (raises CT win probability)
- `lag_00__CT_smoke_inv`: coefficient `0.000453` (raises CT win probability)
- `lag_15__CT_B_site_active_smokes`: coefficient `0.000451` (raises CT win probability)
- `lag_08__CT_B_site_active_smokes`: coefficient `0.000446` (raises CT win probability)
- `lag_15__CT_B_site_active_infernos`: coefficient `-0.000445` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_06__CT_place_OBSERVATION`: coefficient `-0.003100` (lowers CT win probability)
- `lag_01__CT_place_OBSERVATION`: coefficient `0.001796` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001681` (raises CT win probability)
- `lag_03__T_place_HELL`: coefficient `0.001542` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001481` (raises CT win probability)
- `lag_09__CT_place_HUTROOF`: coefficient `0.001461` (raises CT win probability)
- `lag_04__CT_place_OBSERVATION`: coefficient `0.001408` (raises CT win probability)
- `lag_08__T_place_TROPHY`: coefficient `0.001404` (raises CT win probability)
- `lag_03__CT2__is_scoped`: coefficient `-0.001340` (lowers CT win probability)
- `lag_08__CT_place_OBSERVATION`: coefficient `-0.001309` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `45333`, seconds `69.00`, LSTM delta `-0.2277`

Top all feature movements:
- `lag_06__CT_place_OBSERVATION`: contribution `-0.053991`
- `lag_01__CT_place_OBSERVATION`: contribution `-0.031276`
- `lag_09__CT_place_HUTROOF`: contribution `-0.010224`
- `lag_07__CT_place_VENTS`: contribution `-0.009287`
- `lag_08__T_bomb_zone_count`: contribution `-0.006331`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `43765`, seconds `44.50`, LSTM delta `+0.1844`

Top all feature movements:
- `lag_03__T_place_HELL`: contribution `+0.032891`
- `lag_15__CT_place_CRANE`: contribution `+0.017682`
- `lag_08__T_place_TROPHY`: contribution `+0.008905`
- `lag_07__CT2__is_scoped`: contribution `+0.007946`
- `lag_12__CT_place_HUTROOF`: contribution `+0.007658`

Top utility-only movements:
- `lag_15__CT5__flash_duration`: contribution `+0.005680`
- `lag_15__CT_A_site_active_infernos`: contribution `+0.001995`

### tick `44149`, seconds `50.50`, LSTM delta `-0.1576`

Top all feature movements:
- `lag_09__T_place_ADMIN`: contribution `-0.023934`
- `lag_09__T_place_HELL`: contribution `-0.022870`
- `lag_15__T_place_HELL`: contribution `-0.014538`
- `lag_00__T_place_HELL`: contribution `-0.009159`
- `lag_03__CT2__is_scoped`: contribution `-0.008199`

Top utility-only movements:
- `lag_03__CT5__smoke`: contribution `-0.001587`

### tick `45397`, seconds `70.00`, LSTM delta `-0.0978`

Top all feature movements:
- `lag_08__CT_place_OBSERVATION`: contribution `-0.022793`
- `lag_03__CT_place_OBSERVATION`: contribution `-0.022132`
- `lag_11__CT_place_HUTROOF`: contribution `-0.008379`
- `lag_10__T_bomb_zone_count`: contribution `-0.004791`
- `lag_03__CT_place_VENTS`: contribution `+0.003722`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `45269`, seconds `68.00`, LSTM delta `+0.0954`

Top all feature movements:
- `lag_04__CT_place_OBSERVATION`: contribution `+0.024525`
- `lag_07__CT_place_HUTROOF`: contribution `+0.004319`
- `lag_00__kill_diff_last_3s`: contribution `+0.004046`
- `lag_06__T_bomb_zone_count`: contribution `+0.003971`
- `lag_04__T_place_SECRET`: contribution `+0.003707`

Top utility-only movements:
- No utility movement among the top local contributors.
