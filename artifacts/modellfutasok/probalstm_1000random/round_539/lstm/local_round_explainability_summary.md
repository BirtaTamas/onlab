# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-imperial-vs-legacy-bo3-GRvbnL5Q4zT_JzAd-0AXgo/imperial-vs-legacy-m1-inferno.csv`
- round_num: `10`

## Largest probability jumps

- tick `81999`, seconds `21.00`, LSTM `0.3329`, delta `-0.2746`
- tick `82511`, seconds `29.00`, LSTM `0.0647`, delta `-0.2147`
- tick `82415`, seconds `27.50`, LSTM `0.3120`, delta `-0.0831`
- tick `82031`, seconds `21.50`, LSTM `0.2580`, delta `-0.0749`
- tick `82351`, seconds `26.50`, LSTM `0.3621`, delta `+0.0429`
- tick `82319`, seconds `26.00`, LSTM `0.3191`, delta `+0.0427`
- tick `82383`, seconds `27.00`, LSTM `0.3952`, delta `+0.0331`
- tick `81007`, seconds `5.50`, LSTM `0.5170`, delta `+0.0290`
- tick `82191`, seconds `24.00`, LSTM `0.2433`, delta `+0.0254`
- tick `82479`, seconds `28.50`, LSTM `0.2795`, delta `-0.0246`

## Top 15 local ridge features

- `lag_13__CT_place_QUAD`: coefficient `-0.002423`, |coef| `0.002423`
- `lag_00__CT_place_QUAD`: coefficient `-0.002064`, |coef| `0.002064`
- `lag_00__T_kills_last_3s`: coefficient `-0.001893`, |coef| `0.001893`
- `lag_03__CT2__flash_duration`: coefficient `-0.001874`, |coef| `0.001874`
- `lag_03__T_flashed_players`: coefficient `-0.001840`, |coef| `0.001840`
- `lag_00__T1__flash_duration`: coefficient `-0.001608`, |coef| `0.001608`
- `lag_03__T5__flash_duration`: coefficient `-0.001472`, |coef| `0.001472`
- `lag_00__T_damage_last_5s`: coefficient `-0.001458`, |coef| `0.001458`
- `lag_00__kill_diff_last_3s`: coefficient `0.001405`, |coef| `0.001405`
- `lag_00__CT2__flash_duration`: coefficient `-0.001300`, |coef| `0.001300`
- `lag_01__CT2__flash_duration`: coefficient `-0.001259`, |coef| `0.001259`
- `lag_02__CT1__duck_amount`: coefficient `0.001219`, |coef| `0.001219`
- `lag_13__CT_A_site_active_infernos`: coefficient `-0.001200`, |coef| `0.001200`
- `lag_00__damage_diff_last_5s`: coefficient `0.001195`, |coef| `0.001195`
- `lag_12__CT3__is_walking`: coefficient `0.001188`, |coef| `0.001188`

## Top 10 utility ridge features

- `lag_03__CT2__flash_duration`: coefficient `-0.001874` (lowers CT win probability)
- `lag_00__T1__flash_duration`: coefficient `-0.001608` (lowers CT win probability)
- `lag_03__T5__flash_duration`: coefficient `-0.001472` (lowers CT win probability)
- `lag_00__CT2__flash_duration`: coefficient `-0.001300` (lowers CT win probability)
- `lag_01__CT2__flash_duration`: coefficient `-0.001259` (lowers CT win probability)
- `lag_13__CT_A_site_active_infernos`: coefficient `-0.001200` (lowers CT win probability)
- `lag_11__T1__flash_duration`: coefficient `0.001170` (raises CT win probability)
- `lag_02__CT_A_site_active_infernos`: coefficient `0.001152` (raises CT win probability)
- `lag_01__CT_utility_damage_last_5s`: coefficient `-0.001137` (lowers CT win probability)
- `lag_08__T_A_site_active_infernos`: coefficient `0.001133` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_13__CT_place_QUAD`: coefficient `-0.002423` (lowers CT win probability)
- `lag_00__CT_place_QUAD`: coefficient `-0.002064` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001893` (lowers CT win probability)
- `lag_03__T_flashed_players`: coefficient `-0.001840` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001458` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001405` (raises CT win probability)
- `lag_02__CT1__duck_amount`: coefficient `0.001219` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001195` (raises CT win probability)
- `lag_12__CT3__is_walking`: coefficient `0.001188` (raises CT win probability)
- `lag_03__CT3__duck_amount`: coefficient `0.001185` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `81999`, seconds `21.00`, LSTM delta `-0.2746`

Top all feature movements:
- `lag_13__CT_place_QUAD`: contribution `-0.019099`
- `lag_00__CT_place_QUAD`: contribution `-0.016269`
- `lag_00__T1__flash_duration`: contribution `-0.010042`
- `lag_00__CT2__flash_duration`: contribution `-0.007685`
- `lag_00__T_kills_last_3s`: contribution `-0.005996`

Top utility-only movements:
- `lag_00__T1__flash_duration`: contribution `-0.010042`
- `lag_00__CT2__flash_duration`: contribution `-0.007685`
- `lag_13__CT_A_site_active_infernos`: contribution `-0.004234`
- `lag_02__CT_A_site_active_infernos`: contribution `-0.004067`
- `lag_01__CT_utility_damage_last_5s`: contribution `-0.003754`

### tick `82511`, seconds `29.00`, LSTM delta `-0.2147`

Top all feature movements:
- `lag_03__T_flashed_players`: contribution `-0.017756`
- `lag_03__CT2__flash_duration`: contribution `-0.012677`
- `lag_03__T5__flash_duration`: contribution `-0.009000`
- `lag_00__CT2__flash_duration`: contribution `+0.008793`
- `lag_00__T_kills_last_3s`: contribution `-0.005996`

Top utility-only movements:
- `lag_03__CT2__flash_duration`: contribution `-0.012677`
- `lag_03__T5__flash_duration`: contribution `-0.009000`
- `lag_00__CT2__flash_duration`: contribution `+0.008793`
- `lag_03__T_flash_duration_sum`: contribution `-0.005820`
- `lag_11__T1__flash_duration`: contribution `-0.004122`

### tick `82415`, seconds `27.50`, LSTM delta `-0.0831`

Top all feature movements:
- `lag_13__CT_place_QUAD`: contribution `-0.019099`
- `lag_00__CT2__flash_duration`: contribution `-0.008793`
- `lag_00__T_flashed_players`: contribution `-0.006170`
- `lag_00__T1__flash_duration`: contribution `-0.004727`
- `lag_13__CT2__flash_duration`: contribution `-0.003318`

Top utility-only movements:
- `lag_00__CT2__flash_duration`: contribution `-0.008793`
- `lag_00__T1__flash_duration`: contribution `-0.004727`
- `lag_13__CT2__flash_duration`: contribution `-0.003318`
- `lag_00__T_flash_duration_sum`: contribution `-0.002902`
- `lag_02__CT2__flash_duration`: contribution `+0.002274`

### tick `82031`, seconds `21.50`, LSTM delta `-0.0749`

Top all feature movements:
- `lag_14__CT_place_QUAD`: contribution `-0.007660`
- `lag_01__CT2__flash_duration`: contribution `-0.007448`
- `lag_00__T_shots_fired_sum`: contribution `+0.005176`
- `lag_01__CT_place_QUAD`: contribution `-0.004803`
- `lag_08__T_A_site_active_infernos`: contribution `-0.003373`

Top utility-only movements:
- `lag_01__CT2__flash_duration`: contribution `-0.007448`
- `lag_08__T_A_site_active_infernos`: contribution `-0.003373`
- `lag_01__T1__flash_duration`: contribution `-0.003357`
- `lag_05__T_B_site_active_infernos`: contribution `-0.002029`
- `lag_14__CT_A_site_active_infernos`: contribution `-0.001881`

### tick `82351`, seconds `26.50`, LSTM delta `+0.0429`

Top all feature movements:
- `lag_11__T1__flash_duration`: contribution `+0.007309`
- `lag_13__CT_A_site_active_infernos`: contribution `+0.004234`
- `lag_11__CT_place_QUAD`: contribution `-0.003670`
- `lag_11__CT2__flash_duration`: contribution `+0.003171`
- `lag_00__CT2__flash_duration`: contribution `+0.002991`

Top utility-only movements:
- `lag_11__T1__flash_duration`: contribution `+0.007309`
- `lag_13__CT_A_site_active_infernos`: contribution `+0.004234`
- `lag_11__CT2__flash_duration`: contribution `+0.003171`
- `lag_00__CT2__flash_duration`: contribution `+0.002991`
- `lag_06__CT2__flash_duration`: contribution `+0.002503`
