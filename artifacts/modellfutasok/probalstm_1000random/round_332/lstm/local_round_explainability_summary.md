# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-gamerlegion-vs-tyloo-bo3-0g9mXt3FIxC8XzjXNUjRL7/gamerlegion-vs-tyloo-m1-ancient-p3.csv`
- round_num: `12`

## Largest probability jumps

- tick `103290`, seconds `30.00`, LSTM `0.7172`, delta `+0.1132`
- tick `103578`, seconds `34.50`, LSTM `0.8579`, delta `+0.1060`
- tick `103642`, seconds `35.50`, LSTM `0.9303`, delta `+0.0557`
- tick `103322`, seconds `30.50`, LSTM `0.7683`, delta `+0.0511`
- tick `104378`, seconds `47.00`, LSTM `0.9048`, delta `-0.0480`
- tick `103802`, seconds `38.00`, LSTM `0.9598`, delta `+0.0386`
- tick `103418`, seconds `32.00`, LSTM `0.7430`, delta `-0.0269`
- tick `103482`, seconds `33.00`, LSTM `0.7191`, delta `-0.0240`
- tick `103066`, seconds `26.50`, LSTM `0.6040`, delta `-0.0210`
- tick `104506`, seconds `49.00`, LSTM `0.9200`, delta `+0.0205`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001253`, |coef| `0.001253`
- `lag_00__kill_diff_last_3s`: coefficient `0.001225`, |coef| `0.001225`
- `lag_05__CT_shots_fired_sum`: coefficient `-0.001113`, |coef| `0.001113`
- `lag_00__CT_shots_fired_sum`: coefficient `0.000990`, |coef| `0.000990`
- `lag_00__CT2__shots_fired`: coefficient `0.000967`, |coef| `0.000967`
- `lag_00__damage_diff_last_5s`: coefficient `0.000925`, |coef| `0.000925`
- `lag_08__T2__is_walking`: coefficient `-0.000911`, |coef| `0.000911`
- `lag_00__CT_damage_last_5s`: coefficient `0.000904`, |coef| `0.000904`
- `lag_05__T5__flash_duration`: coefficient `0.000844`, |coef| `0.000844`
- `lag_05__T2__duck_amount`: coefficient `-0.000830`, |coef| `0.000830`
- `lag_07__T2__duck_amount`: coefficient `0.000797`, |coef| `0.000797`
- `lag_00__T5__alive`: coefficient `-0.000789`, |coef| `0.000789`
- `lag_05__CT2__shots_fired`: coefficient `-0.000786`, |coef| `0.000786`
- `lag_00__T5__hp`: coefficient `-0.000775`, |coef| `0.000775`
- `lag_11__CT1__duck_amount`: coefficient `-0.000752`, |coef| `0.000752`

## Top 10 utility ridge features

- `lag_05__T5__flash_duration`: coefficient `0.000844` (raises CT win probability)
- `lag_00__T5__molly`: coefficient `-0.000715` (lowers CT win probability)
- `lag_08__T_A_site_active_infernos`: coefficient `0.000714` (raises CT win probability)
- `lag_00__CT5__smoke`: coefficient `-0.000709` (lowers CT win probability)
- `lag_11__T5__flash_duration`: coefficient `0.000680` (raises CT win probability)
- `lag_14__T_A_site_active_infernos`: coefficient `-0.000674` (lowers CT win probability)
- `lag_11__T3__molly`: coefficient `-0.000644` (lowers CT win probability)
- `lag_08__T_B_site_active_infernos`: coefficient `0.000643` (raises CT win probability)
- `lag_14__T_B_site_active_infernos`: coefficient `-0.000639` (lowers CT win probability)
- `lag_07__T3__smoke`: coefficient `-0.000619` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001253` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001225` (raises CT win probability)
- `lag_05__CT_shots_fired_sum`: coefficient `-0.001113` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.000990` (raises CT win probability)
- `lag_00__CT2__shots_fired`: coefficient `0.000967` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000925` (raises CT win probability)
- `lag_08__T2__is_walking`: coefficient `-0.000911` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.000904` (raises CT win probability)
- `lag_05__T2__duck_amount`: coefficient `-0.000830` (lowers CT win probability)
- `lag_07__T2__duck_amount`: coefficient `0.000797` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `103290`, seconds `30.00`, LSTM delta `+0.1132`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.003617`
- `lag_00__CT_shots_fired_sum`: contribution `+0.003438`
- `lag_05__T2__duck_amount`: contribution `+0.003173`
- `lag_07__T2__duck_amount`: contribution `+0.003047`
- `lag_00__kill_diff_last_3s`: contribution `+0.002949`

Top utility-only movements:
- `lag_08__T_A_site_active_infernos`: contribution `+0.002126`
- `lag_14__T_A_site_active_infernos`: contribution `+0.002005`
- `lag_08__T_B_site_active_infernos`: contribution `+0.001819`
- `lag_14__T_B_site_active_infernos`: contribution `+0.001807`

### tick `103578`, seconds `34.50`, LSTM delta `+0.1060`

Top all feature movements:
- `lag_05__CT_shots_fired_sum`: contribution `+0.016240`
- `lag_05__CT2__shots_fired`: contribution `+0.008201`
- `lag_05__T5__flash_duration`: contribution `+0.005592`
- `lag_00__CT_kills_last_3s`: contribution `+0.003617`
- `lag_00__kill_diff_last_3s`: contribution `+0.002949`

Top utility-only movements:
- `lag_05__T5__flash_duration`: contribution `+0.005592`
- `lag_04__CT_A_site_active_infernos`: contribution `+0.001665`

### tick `103642`, seconds `35.50`, LSTM delta `+0.0557`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.003617`
- `lag_00__CT_shots_fired_sum`: contribution `-0.003438`
- `lag_07__T5__flash_duration`: contribution `+0.003284`
- `lag_00__kill_diff_last_3s`: contribution `+0.002949`
- `lag_04__T_place_HOUSE`: contribution `+0.002598`

Top utility-only movements:
- `lag_07__T5__flash_duration`: contribution `+0.003284`

### tick `103322`, seconds `30.50`, LSTM delta `+0.0511`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.003438`
- `lag_14__CT1__duck_amount`: contribution `+0.002423`
- `lag_08__T2__duck_amount`: contribution `+0.002407`
- `lag_00__CT2__shots_fired`: contribution `+0.002405`
- `lag_12__CT1__duck_amount`: contribution `+0.001470`

Top utility-only movements:
- `lag_12__T3__molly`: contribution `+0.000783`

### tick `104378`, seconds `47.00`, LSTM delta `-0.0480`

Top all feature movements:
- `lag_03__CT_place_TSIDELOWER`: contribution `-0.007731`
- `lag_00__kill_diff_last_3s`: contribution `-0.002949`
- `lag_08__T_place_SIDEHALL`: contribution `-0.002703`
- `lag_01__T_bomb_zone_count`: contribution `-0.001893`
- `lag_14__T_duck_amount_mean`: contribution `-0.001686`

Top utility-only movements:
- `lag_14__CT2__flash_duration`: contribution `-0.001307`
- `lag_11__T4__flash_duration`: contribution `-0.001175`
- `lag_03__CT_A_site_active_infernos`: contribution `-0.000931`
