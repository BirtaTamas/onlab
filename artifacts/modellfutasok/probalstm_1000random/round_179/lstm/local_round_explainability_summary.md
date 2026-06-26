# Local Round Explainability

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-the-mongolz-vs-3dmax-bo3-NhOpC3bR-AJd86c-60IeuJ/the-mongolz-vs-3dmax-m1-nuke.csv`
- round_num: `4`

## Largest probability jumps

- tick `25822`, seconds `97.50`, LSTM `0.6125`, delta `+0.2748`
- tick `25854`, seconds `98.00`, LSTM `0.8365`, delta `+0.2240`
- tick `25982`, seconds `100.00`, LSTM `0.5826`, delta `-0.1784`
- tick `26078`, seconds `101.50`, LSTM `0.8230`, delta `+0.1752`
- tick `25950`, seconds `99.50`, LSTM `0.7610`, delta `-0.1377`
- tick `26174`, seconds `103.00`, LSTM `0.9385`, delta `+0.0988`
- tick `24126`, seconds `71.00`, LSTM `0.4069`, delta `-0.0850`
- tick `25534`, seconds `93.00`, LSTM `0.3995`, delta `-0.0719`
- tick `26046`, seconds `101.00`, LSTM `0.6478`, delta `+0.0707`
- tick `25086`, seconds `86.00`, LSTM `0.4665`, delta `+0.0656`

## Top 15 local ridge features

- `lag_12__CT_place_VENTS`: coefficient `0.005684`, |coef| `0.005684`
- `lag_00__kill_diff_last_3s`: coefficient `0.004350`, |coef| `0.004350`
- `lag_00__CT_kills_last_3s`: coefficient `0.003809`, |coef| `0.003809`
- `lag_13__CT_place_VENTS`: coefficient `0.003645`, |coef| `0.003645`
- `lag_00__T_place_HUT`: coefficient `0.003565`, |coef| `0.003565`
- `lag_00__CT_place_DECON`: coefficient `-0.003338`, |coef| `0.003338`
- `lag_00__damage_diff_last_5s`: coefficient `0.002907`, |coef| `0.002907`
- `lag_00__T_place_BOMBSITEA`: coefficient `-0.002659`, |coef| `0.002659`
- `lag_00__T_macro_A`: coefficient `-0.002659`, |coef| `0.002659`
- `lag_00__T_place_SQUEAKY`: coefficient `0.002491`, |coef| `0.002491`
- `lag_03__T_place_SQUEAKY`: coefficient `-0.002442`, |coef| `0.002442`
- `lag_00__CT_damage_last_5s`: coefficient `0.001997`, |coef| `0.001997`
- `lag_05__T5__is_walking`: coefficient `0.001977`, |coef| `0.001977`
- `lag_11__CT_place_VENTS`: coefficient `0.001934`, |coef| `0.001934`
- `lag_12__CT_place_TUNNELS`: coefficient `-0.001932`, |coef| `0.001932`

## Top 10 utility ridge features

- `lag_04__T_A_site_active_infernos`: coefficient `-0.001787` (lowers CT win probability)
- `lag_04__T_B_site_active_infernos`: coefficient `-0.001700` (lowers CT win probability)
- `lag_04__T_active_infernos`: coefficient `-0.001304` (lowers CT win probability)
- `lag_05__T_A_site_active_infernos`: coefficient `-0.001184` (lowers CT win probability)
- `lag_05__T_B_site_active_infernos`: coefficient `-0.001127` (lowers CT win probability)
- `lag_12__T3__flash`: coefficient `-0.001061` (lowers CT win probability)
- `lag_04__active_infernos_total`: coefficient `-0.000940` (lowers CT win probability)
- `lag_05__T_active_infernos`: coefficient `-0.000891` (lowers CT win probability)
- `lag_12__T_utility_damage_last_5s`: coefficient `-0.000852` (lowers CT win probability)
- `lag_12__T_A_site_active_infernos`: coefficient `-0.000807` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_12__CT_place_VENTS`: coefficient `0.005684` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.004350` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003809` (raises CT win probability)
- `lag_13__CT_place_VENTS`: coefficient `0.003645` (raises CT win probability)
- `lag_00__T_place_HUT`: coefficient `0.003565` (raises CT win probability)
- `lag_00__CT_place_DECON`: coefficient `-0.003338` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002907` (raises CT win probability)
- `lag_00__T_place_BOMBSITEA`: coefficient `-0.002659` (lowers CT win probability)
- `lag_00__T_macro_A`: coefficient `-0.002659` (lowers CT win probability)
- `lag_00__T_place_SQUEAKY`: coefficient `0.002491` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `25822`, seconds `97.50`, LSTM delta `+0.2748`

Top all feature movements:
- `lag_12__CT_place_VENTS`: contribution `+0.047692`
- `lag_03__T_place_SQUEAKY`: contribution `+0.015205`
- `lag_00__CT_kills_last_3s`: contribution `+0.010996`
- `lag_00__kill_diff_last_3s`: contribution `+0.010470`
- `lag_05__T_place_SQUEAKY`: contribution `+0.010072`

Top utility-only movements:
- `lag_04__T_A_site_active_infernos`: contribution `+0.005320`
- `lag_04__T_B_site_active_infernos`: contribution `+0.004805`

### tick `25854`, seconds `98.00`, LSTM delta `+0.2240`

Top all feature movements:
- `lag_00__T_place_HUT`: contribution `+0.033235`
- `lag_13__CT_place_VENTS`: contribution `+0.030581`
- `lag_00__CT_kills_last_3s`: contribution `+0.010996`
- `lag_00__kill_diff_last_3s`: contribution `+0.010470`
- `lag_10__T_place_SQUEAKY`: contribution `+0.008837`

Top utility-only movements:
- `lag_05__T_A_site_active_infernos`: contribution `+0.003523`
- `lag_05__T_B_site_active_infernos`: contribution `+0.003187`

### tick `25982`, seconds `100.00`, LSTM delta `-0.1784`

Top all feature movements:
- `lag_02__CT_place_CRANE`: contribution `-0.019186`
- `lag_04__T_place_HUT`: contribution `-0.017254`
- `lag_02__CT_place_HUTROOF`: contribution `-0.010729`
- `lag_00__kill_diff_last_3s`: contribution `-0.010470`
- `lag_08__T_place_SQUEAKY`: contribution `-0.009890`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `26078`, seconds `101.50`, LSTM delta `+0.1752`

Top all feature movements:
- `lag_05__CT_place_CRANE`: contribution `+0.016289`
- `lag_07__T_place_HUT`: contribution `+0.012205`
- `lag_00__CT_kills_last_3s`: contribution `+0.010996`
- `lag_00__kill_diff_last_3s`: contribution `+0.010470`
- `lag_00__CT_place_VENTS`: contribution `+0.008903`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `25950`, seconds `99.50`, LSTM delta `-0.1377`

Top all feature movements:
- `lag_03__T_place_HUT`: contribution `-0.016826`
- `lag_00__kill_diff_last_3s`: contribution `-0.010470`
- `lag_07__T_place_SQUEAKY`: contribution `-0.010317`
- `lag_01__CT_place_HUTROOF`: contribution `-0.009129`
- `lag_09__T_place_SQUEAKY`: contribution `+0.006018`

Top utility-only movements:
- No utility movement among the top local contributors.
