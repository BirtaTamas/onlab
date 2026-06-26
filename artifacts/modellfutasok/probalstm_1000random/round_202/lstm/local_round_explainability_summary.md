# Local Round Explainability

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-faze-vs-aurora-bo3-ZssSxRC3p7Nn5A_BOLQ-lD/faze-vs-aurora-m2-mirage.csv`
- round_num: `8`

## Largest probability jumps

- tick `56189`, seconds `65.50`, LSTM `0.4803`, delta `+0.3480`
- tick `56285`, seconds `67.00`, LSTM `0.7375`, delta `+0.2983`
- tick `55773`, seconds `59.00`, LSTM `0.1178`, delta `-0.2849`
- tick `55421`, seconds `53.50`, LSTM `0.1941`, delta `-0.2802`
- tick `55613`, seconds `56.50`, LSTM `0.3773`, delta `+0.2732`
- tick `54589`, seconds `40.50`, LSTM `0.5713`, delta `-0.1929`
- tick `54077`, seconds `32.50`, LSTM `0.7987`, delta `+0.1369`
- tick `54173`, seconds `34.00`, LSTM `0.8682`, delta `+0.1105`
- tick `53725`, seconds `27.00`, LSTM `0.7156`, delta `+0.0897`
- tick `55997`, seconds `62.50`, LSTM `0.1115`, delta `+0.0781`

## Top 15 local ridge features

- `lag_00__CT_defusing_count`: coefficient `0.005576`, |coef| `0.005576`
- `lag_00__kill_diff_last_3s`: coefficient `0.004501`, |coef| `0.004501`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.003956`, |coef| `0.003956`
- `lag_00__damage_diff_last_5s`: coefficient `0.003741`, |coef| `0.003741`
- `lag_00__T_kills_last_3s`: coefficient `-0.003505`, |coef| `0.003505`
- `lag_03__T_flash_alpha_mean`: coefficient `-0.003498`, |coef| `0.003498`
- `lag_00__CT_shots_fired_sum`: coefficient `0.003446`, |coef| `0.003446`
- `lag_00__T_shots_fired_sum`: coefficient `-0.003024`, |coef| `0.003024`
- `lag_13__CT_place_APARTMENTS`: coefficient `-0.002881`, |coef| `0.002881`
- `lag_01__CT_defusing_count`: coefficient `0.002849`, |coef| `0.002849`
- `lag_09__T2__shots_fired`: coefficient `0.002751`, |coef| `0.002751`
- `lag_00__CT_velocity_mean`: coefficient `-0.002683`, |coef| `0.002683`
- `lag_09__CT_B_site_active_infernos`: coefficient `-0.002614`, |coef| `0.002614`
- `lag_12__CT_B_site_active_infernos`: coefficient `-0.002485`, |coef| `0.002485`
- `lag_01__CT3__shots_fired`: coefficient `-0.002374`, |coef| `0.002374`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.003956` (lowers CT win probability)
- `lag_03__T_flash_alpha_mean`: coefficient `-0.003498` (lowers CT win probability)
- `lag_09__CT_B_site_active_infernos`: coefficient `-0.002614` (lowers CT win probability)
- `lag_12__CT_B_site_active_infernos`: coefficient `-0.002485` (lowers CT win probability)
- `lag_04__T_flash_alpha_mean`: coefficient `-0.001788` (lowers CT win probability)
- `lag_10__CT_B_site_active_infernos`: coefficient `-0.001759` (lowers CT win probability)
- `lag_09__CT_active_infernos`: coefficient `-0.001615` (lowers CT win probability)
- `lag_02__T_flash_alpha_mean`: coefficient `-0.001553` (lowers CT win probability)
- `lag_12__CT_active_infernos`: coefficient `-0.001506` (lowers CT win probability)
- `lag_11__CT_B_site_active_infernos`: coefficient `-0.001439` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_defusing_count`: coefficient `0.005576` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.004501` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003741` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003505` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.003446` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.003024` (lowers CT win probability)
- `lag_13__CT_place_APARTMENTS`: coefficient `-0.002881` (lowers CT win probability)
- `lag_01__CT_defusing_count`: coefficient `0.002849` (raises CT win probability)
- `lag_09__T2__shots_fired`: coefficient `0.002751` (raises CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.002683` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `56189`, seconds `65.50`, LSTM delta `+0.3480`

Top all feature movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.024001`
- `lag_00__CT_shots_fired_sum`: contribution `+0.011969`
- `lag_13__CT_place_APARTMENTS`: contribution `+0.011067`
- `lag_09__CT_B_site_active_infernos`: contribution `+0.008981`
- `lag_00__CT_duck_amount_mean`: contribution `+0.008620`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.024001`
- `lag_09__CT_B_site_active_infernos`: contribution `+0.008981`
- `lag_09__CT_active_infernos`: contribution `+0.003722`

### tick `56285`, seconds `67.00`, LSTM delta `+0.2983`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `+0.054049`
- `lag_03__T_flash_alpha_mean`: contribution `+0.021222`
- `lag_03__CT_duck_amount_mean`: contribution `+0.011351`
- `lag_02__CT_duck_amount_mean`: contribution `+0.009927`
- `lag_00__CT_velocity_mean`: contribution `+0.008724`

Top utility-only movements:
- `lag_03__T_flash_alpha_mean`: contribution `+0.021222`
- `lag_12__CT_B_site_active_infernos`: contribution `+0.008536`
- `lag_12__CT_active_infernos`: contribution `+0.003471`

### tick `55773`, seconds `59.00`, LSTM delta `-0.2849`

Top all feature movements:
- `lag_09__T2__shots_fired`: contribution `-0.017806`
- `lag_09__T_shots_fired_sum`: contribution `-0.013998`
- `lag_00__T_kills_last_3s`: contribution `-0.011105`
- `lag_00__kill_diff_last_3s`: contribution `-0.010833`
- `lag_00__damage_diff_last_5s`: contribution `-0.008440`

Top utility-only movements:
- `lag_10__CT_B_site_active_infernos`: contribution `-0.006042`

### tick `55421`, seconds `53.50`, LSTM delta `-0.2802`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.016756`
- `lag_00__T_shots_fired_sum`: contribution `-0.011336`
- `lag_00__T_kills_last_3s`: contribution `-0.011105`
- `lag_00__kill_diff_last_3s`: contribution `-0.010833`
- `lag_00__CT_place_SHOP`: contribution `-0.007966`

Top utility-only movements:
- `lag_08__CT1__molly`: contribution `-0.003400`
- `lag_00__CT3__molly`: contribution `-0.003376`
- `lag_01__CT1__molly`: contribution `-0.003319`

### tick `55613`, seconds `56.50`, LSTM delta `+0.2732`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.021665`
- `lag_04__T_shots_fired_sum`: contribution `+0.019088`
- `lag_00__CT_shots_fired_sum`: contribution `+0.014362`
- `lag_06__CT_shots_fired_sum`: contribution `+0.011523`
- `lag_04__T2__shots_fired`: contribution `+0.011269`

Top utility-only movements:
- `lag_05__CT_B_site_active_infernos`: contribution `+0.003784`
