# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-vitality-vs-the-mongolz-bo3-vhm_UWcBfYfYcOeLh9JIDA/vitality-vs-the-mongolz-m1-mirage.csv`
- round_num: `10`

## Largest probability jumps

- tick `84581`, seconds `28.50`, LSTM `0.4616`, delta `+0.2622`
- tick `86693`, seconds `61.50`, LSTM `0.8335`, delta `+0.2383`
- tick `86341`, seconds `56.00`, LSTM `0.7081`, delta `+0.2120`
- tick `86789`, seconds `63.00`, LSTM `0.9354`, delta `+0.0520`
- tick `83493`, seconds `11.50`, LSTM `0.4076`, delta `+0.0499`
- tick `84101`, seconds `21.00`, LSTM `0.3155`, delta `-0.0489`
- tick `86533`, seconds `59.00`, LSTM `0.6362`, delta `-0.0489`
- tick `84453`, seconds `26.50`, LSTM `0.1991`, delta `-0.0476`
- tick `86213`, seconds `54.00`, LSTM `0.4922`, delta `+0.0430`
- tick `85477`, seconds `42.50`, LSTM `0.5138`, delta `+0.0408`

## Top 15 local ridge features

- `lag_05__CT_place_SCAFFOLDING`: coefficient `0.003715`, |coef| `0.003715`
- `lag_00__CT_kills_last_3s`: coefficient `0.003057`, |coef| `0.003057`
- `lag_00__kill_diff_last_3s`: coefficient `0.002548`, |coef| `0.002548`
- `lag_12__T_place_CONNECTOR`: coefficient `0.002147`, |coef| `0.002147`
- `lag_12__CT_place_SHOP`: coefficient `0.001956`, |coef| `0.001956`
- `lag_00__CT_damage_last_5s`: coefficient `0.001823`, |coef| `0.001823`
- `lag_00__damage_diff_last_5s`: coefficient `0.001790`, |coef| `0.001790`
- `lag_03__T3__duck_amount`: coefficient `0.001720`, |coef| `0.001720`
- `lag_12__CT3__is_walking`: coefficient `-0.001689`, |coef| `0.001689`
- `lag_06__CT_place_SCAFFOLDING`: coefficient `0.001484`, |coef| `0.001484`
- `lag_01__CT_kills_last_3s`: coefficient `0.001453`, |coef| `0.001453`
- `lag_05__T5__is_walking`: coefficient `-0.001444`, |coef| `0.001444`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001242`, |coef| `0.001242`
- `lag_05__CT_shots_fired_sum`: coefficient `0.001241`, |coef| `0.001241`
- `lag_05__CT_kills_last_3s`: coefficient `-0.001238`, |coef| `0.001238`

## Top 10 utility ridge features

- `lag_02__T2__molly`: coefficient `-0.000947` (lowers CT win probability)
- `lag_07__T4__molly`: coefficient `-0.000928` (lowers CT win probability)
- `lag_15__T_utility_damage_last_5s`: coefficient `0.000926` (raises CT win probability)
- `lag_13__T2__molly`: coefficient `-0.000894` (lowers CT win probability)
- `lag_09__T_B_site_active_infernos`: coefficient `0.000885` (raises CT win probability)
- `lag_03__T_B_site_active_infernos`: coefficient `0.000775` (raises CT win probability)
- `lag_05__T_utility_damage_last_5s`: coefficient `-0.000722` (lowers CT win probability)
- `lag_00__T_B_site_active_infernos`: coefficient `-0.000719` (lowers CT win probability)
- `lag_08__T_B_site_active_smokes`: coefficient `-0.000670` (lowers CT win probability)
- `lag_02__T_utility_damage_last_5s`: coefficient `-0.000667` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_05__CT_place_SCAFFOLDING`: coefficient `0.003715` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003057` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002548` (raises CT win probability)
- `lag_12__T_place_CONNECTOR`: coefficient `0.002147` (raises CT win probability)
- `lag_12__CT_place_SHOP`: coefficient `0.001956` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001823` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001790` (raises CT win probability)
- `lag_03__T3__duck_amount`: coefficient `0.001720` (raises CT win probability)
- `lag_12__CT3__is_walking`: coefficient `-0.001689` (lowers CT win probability)
- `lag_06__CT_place_SCAFFOLDING`: coefficient `0.001484` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `84581`, seconds `28.50`, LSTM delta `+0.2622`

Top all feature movements:
- `lag_05__CT_place_SCAFFOLDING`: contribution `+0.077532`
- `lag_08__T3__shots_fired`: contribution `+0.008936`
- `lag_08__T_shots_fired_sum`: contribution `+0.008882`
- `lag_00__CT_kills_last_3s`: contribution `+0.008826`
- `lag_00__kill_diff_last_3s`: contribution `+0.006134`

Top utility-only movements:
- `lag_15__T_utility_damage_last_5s`: contribution `+0.004627`
- `lag_05__T_utility_damage_last_5s`: contribution `+0.003610`

### tick `86693`, seconds `61.50`, LSTM delta `+0.2383`

Top all feature movements:
- `lag_12__CT_place_SHOP`: contribution `+0.009812`
- `lag_00__CT_kills_last_3s`: contribution `+0.008826`
- `lag_00__T_shots_fired_sum`: contribution `+0.007447`
- `lag_00__kill_diff_last_3s`: contribution `+0.006134`
- `lag_00__T5__shots_fired`: contribution `+0.004448`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `86341`, seconds `56.00`, LSTM delta `+0.2120`

Top all feature movements:
- `lag_12__T_place_CONNECTOR`: contribution `+0.010397`
- `lag_00__CT_kills_last_3s`: contribution `+0.008826`
- `lag_03__T3__duck_amount`: contribution `+0.006485`
- `lag_00__kill_diff_last_3s`: contribution `+0.006134`
- `lag_01__CT_place_SHOP`: contribution `+0.005877`

Top utility-only movements:
- `lag_03__T_B_site_active_infernos`: contribution `+0.002191`

### tick `86789`, seconds `63.00`, LSTM delta `+0.0520`

Top all feature movements:
- `lag_01__CT_place_SHOP`: contribution `-0.005877`
- `lag_03__T5__duck_amount`: contribution `+0.004188`
- `lag_03__T_shots_fired_sum`: contribution `+0.004143`
- `lag_05__CT_shots_fired_sum`: contribution `-0.003449`
- `lag_14__T3__duck_amount`: contribution `-0.003053`

Top utility-only movements:
- `lag_03__T_B_site_active_infernos`: contribution `-0.002191`

### tick `83493`, seconds `11.50`, LSTM delta `+0.0499`

Top all feature movements:
- `lag_12__CT_place_SHOP`: contribution `+0.009812`
- `lag_00__CT_place_TRUCK`: contribution `+0.005739`
- `lag_02__CT1__is_walking`: contribution `-0.002149`
- `lag_04__T_place_HOUSE`: contribution `+0.002051`
- `lag_00__T5__flash_duration`: contribution `+0.001854`

Top utility-only movements:
- `lag_00__T5__flash_duration`: contribution `+0.001854`
- `lag_00__T1__flash_duration`: contribution `+0.001672`
- `lag_00__T_flash_duration_sum`: contribution `+0.001154`
