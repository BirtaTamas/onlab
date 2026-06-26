# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-mouz-vs-virtuspro-bo3-RgsQGjmI__aLZMP1KntvtG/mouz-vs-virtus-pro-m2-mirage.csv`
- round_num: `7`

## Largest probability jumps

- tick `62914`, seconds `34.50`, LSTM `0.1072`, delta `-0.2129`
- tick `62498`, seconds `28.00`, LSTM `0.5759`, delta `+0.1463`
- tick `62978`, seconds `35.50`, LSTM `0.0321`, delta `-0.0932`
- tick `62754`, seconds `32.00`, LSTM `0.4961`, delta `-0.0910`
- tick `62658`, seconds `30.50`, LSTM `0.6371`, delta `+0.0845`
- tick `62082`, seconds `21.50`, LSTM `0.4602`, delta `-0.0706`
- tick `62786`, seconds `32.50`, LSTM `0.4280`, delta `-0.0681`
- tick `62722`, seconds `31.50`, LSTM `0.5871`, delta `-0.0501`
- tick `62818`, seconds `33.00`, LSTM `0.3789`, delta `-0.0491`
- tick `62850`, seconds `33.50`, LSTM `0.3328`, delta `-0.0461`

## Top 15 local ridge features

- `lag_08__CT_place_SHOP`: coefficient `-0.001643`, |coef| `0.001643`
- `lag_12__CT_shots_fired_sum`: coefficient `0.001496`, |coef| `0.001496`
- `lag_06__CT_place_SHOP`: coefficient `-0.001352`, |coef| `0.001352`
- `lag_13__CT_shots_fired_sum`: coefficient `-0.001242`, |coef| `0.001242`
- `lag_13__T5__shots_fired`: coefficient `-0.001231`, |coef| `0.001231`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001216`, |coef| `0.001216`
- `lag_12__CT_place_JUNGLE`: coefficient `0.001198`, |coef| `0.001198`
- `lag_13__T_place_BACKALLEY`: coefficient `0.001187`, |coef| `0.001187`
- `lag_01__CT3__flash_duration`: coefficient `0.001175`, |coef| `0.001175`
- `lag_15__T_place_HOUSE`: coefficient `-0.001119`, |coef| `0.001119`
- `lag_15__T_place_BACKALLEY`: coefficient `0.001064`, |coef| `0.001064`
- `lag_00__kill_diff_last_3s`: coefficient `0.001031`, |coef| `0.001031`
- `lag_03__CT_place_JUNGLE`: coefficient `-0.001016`, |coef| `0.001016`
- `lag_00__damage_diff_last_5s`: coefficient `0.001013`, |coef| `0.001013`
- `lag_05__CT3__flash_duration`: coefficient `0.001008`, |coef| `0.001008`

## Top 10 utility ridge features

- `lag_01__CT3__flash_duration`: coefficient `0.001175` (raises CT win probability)
- `lag_05__CT3__flash_duration`: coefficient `0.001008` (raises CT win probability)
- `lag_07__CT2__flash_duration`: coefficient `0.000960` (raises CT win probability)
- `lag_00__CT3__flash_duration`: coefficient `0.000789` (raises CT win probability)
- `lag_04__T_A_site_active_infernos`: coefficient `0.000770` (raises CT win probability)
- `lag_10__CT_utility_damage_last_5s`: coefficient `-0.000765` (lowers CT win probability)
- `lag_13__T5__flash_duration`: coefficient `0.000752` (raises CT win probability)
- `lag_15__CT2__flash_duration`: coefficient `0.000720` (raises CT win probability)
- `lag_14__CT3__flash_duration`: coefficient `-0.000710` (lowers CT win probability)
- `lag_00__CT2__flash_duration`: coefficient `-0.000692` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_08__CT_place_SHOP`: coefficient `-0.001643` (lowers CT win probability)
- `lag_12__CT_shots_fired_sum`: coefficient `0.001496` (raises CT win probability)
- `lag_06__CT_place_SHOP`: coefficient `-0.001352` (lowers CT win probability)
- `lag_13__CT_shots_fired_sum`: coefficient `-0.001242` (lowers CT win probability)
- `lag_13__T5__shots_fired`: coefficient `-0.001231` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001216` (raises CT win probability)
- `lag_12__CT_place_JUNGLE`: coefficient `0.001198` (raises CT win probability)
- `lag_13__T_place_BACKALLEY`: coefficient `0.001187` (raises CT win probability)
- `lag_15__T_place_HOUSE`: coefficient `-0.001119` (lowers CT win probability)
- `lag_15__T_place_BACKALLEY`: coefficient `0.001064` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `62914`, seconds `34.50`, LSTM delta `-0.2129`

Top all feature movements:
- `lag_12__CT_shots_fired_sum`: contribution `-0.017669`
- `lag_08__T_place_TRUCK`: contribution `-0.012625`
- `lag_13__CT_shots_fired_sum`: contribution `-0.008630`
- `lag_08__CT_place_SHOP`: contribution `-0.008243`
- `lag_06__CT_place_SHOP`: contribution `-0.006779`

Top utility-only movements:
- `lag_05__CT3__flash_duration`: contribution `-0.005113`
- `lag_14__CT3__flash_duration`: contribution `-0.003601`
- `lag_13__T5__flash_duration`: contribution `-0.003468`
- `lag_14__T5__flash_duration`: contribution `-0.003121`
- `lag_04__T_A_site_active_infernos`: contribution `-0.002291`

### tick `62498`, seconds `28.00`, LSTM delta `+0.1463`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.008450`
- `lag_03__CT_place_JUNGLE`: contribution `+0.006520`
- `lag_13__CT2__is_scoped`: contribution `+0.006153`
- `lag_01__CT3__flash_duration`: contribution `+0.005960`
- `lag_09__CT_place_UNDERPASS`: contribution `+0.004866`

Top utility-only movements:
- `lag_01__CT3__flash_duration`: contribution `+0.005960`
- `lag_10__CT_utility_damage_last_5s`: contribution `+0.003957`
- `lag_01__T5__flash_duration`: contribution `+0.002978`

### tick `62978`, seconds `35.50`, LSTM delta `-0.0932`

Top all feature movements:
- `lag_10__T_place_TRUCK`: contribution `-0.014318`
- `lag_08__T_place_TRUCK`: contribution `+0.012625`
- `lag_08__CT_place_SHOP`: contribution `-0.008243`
- `lag_15__T_place_BACKALLEY`: contribution `-0.003220`
- `lag_15__T5__shots_fired`: contribution `-0.003107`

Top utility-only movements:
- `lag_07__CT3__flash_duration`: contribution `-0.002056`

### tick `62754`, seconds `32.00`, LSTM delta `-0.0910`

Top all feature movements:
- `lag_07__CT_shots_fired_sum`: contribution `-0.008937`
- `lag_03__T_place_TRUCK`: contribution `-0.004630`
- `lag_11__CT_place_JUNGLE`: contribution `-0.004147`
- `lag_00__CT3__flash_duration`: contribution `-0.004002`
- `lag_03__T4__is_scoped`: contribution `-0.003576`

Top utility-only movements:
- `lag_00__CT3__flash_duration`: contribution `-0.004002`

### tick `62658`, seconds `30.50`, LSTM delta `+0.0845`

Top all feature movements:
- `lag_04__CT_shots_fired_sum`: contribution `+0.008473`
- `lag_12__CT_place_JUNGLE`: contribution `+0.007687`
- `lag_00__T_place_TRUCK`: contribution `+0.004419`
- `lag_11__CT_place_JUNGLE`: contribution `+0.004147`
- `lag_05__CT_shots_fired_sum`: contribution `+0.003934`

Top utility-only movements:
- `lag_06__CT3__flash_duration`: contribution `+0.002695`
- `lag_15__CT_utility_damage_last_5s`: contribution `+0.002444`
- `lag_05__T5__flash_duration`: contribution `+0.002196`
- `lag_06__T5__flash_duration`: contribution `+0.001940`
- `lag_15__utility_damage_diff_last_5s`: contribution `+0.001800`
