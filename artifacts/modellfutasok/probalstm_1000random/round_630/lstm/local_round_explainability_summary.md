# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-mouz-vs-pain-bo3-Ao8EIC0rxvFkpkJ5bGImFu/mouz-vs-pain-m1-nuke.csv`
- round_num: `7`

## Largest probability jumps

- tick `60950`, seconds `96.00`, LSTM `0.8283`, delta `+0.4257`
- tick `61046`, seconds `97.50`, LSTM `0.5947`, delta `-0.2322`
- tick `61398`, seconds `103.00`, LSTM `0.3672`, delta `-0.2241`
- tick `61110`, seconds `98.50`, LSTM `0.6135`, delta `+0.1542`
- tick `61078`, seconds `98.00`, LSTM `0.4593`, delta `-0.1354`
- tick `61494`, seconds `104.50`, LSTM `0.1796`, delta `-0.0841`
- tick `61302`, seconds `101.50`, LSTM `0.5437`, delta `-0.0837`
- tick `60918`, seconds `95.50`, LSTM `0.4026`, delta `+0.0769`
- tick `59958`, seconds `80.50`, LSTM `0.3637`, delta `+0.0664`
- tick `59574`, seconds `74.50`, LSTM `0.3100`, delta `+0.0591`

## Top 15 local ridge features

- `lag_01__T_place_RAFTERS`: coefficient `0.004182`, |coef| `0.004182`
- `lag_03__T_place_RAFTERS`: coefficient `0.002490`, |coef| `0.002490`
- `lag_01__T_place_MINI`: coefficient `0.002340`, |coef| `0.002340`
- `lag_02__T_place_RAFTERS`: coefficient `0.002182`, |coef| `0.002182`
- `lag_00__T_place_RAFTERS`: coefficient `-0.001810`, |coef| `0.001810`
- `lag_01__CT_utility_damage_last_5s`: coefficient `0.001772`, |coef| `0.001772`
- `lag_00__T_place_HELL`: coefficient `-0.001768`, |coef| `0.001768`
- `lag_00__T_place_HEAVEN`: coefficient `-0.001647`, |coef| `0.001647`
- `lag_07__CT_place_HUT`: coefficient `-0.001563`, |coef| `0.001563`
- `lag_05__T_place_RAFTERS`: coefficient `-0.001478`, |coef| `0.001478`
- `lag_01__utility_damage_diff_last_5s`: coefficient `0.001449`, |coef| `0.001449`
- `lag_04__T_place_MINI`: coefficient `-0.001413`, |coef| `0.001413`
- `lag_11__T1__is_walking`: coefficient `0.001388`, |coef| `0.001388`
- `lag_11__T4__shots_fired`: coefficient `-0.001281`, |coef| `0.001281`
- `lag_00__T4__shots_fired`: coefficient `-0.001251`, |coef| `0.001251`

## Top 10 utility ridge features

- `lag_01__CT_utility_damage_last_5s`: coefficient `0.001772` (raises CT win probability)
- `lag_01__utility_damage_diff_last_5s`: coefficient `0.001449` (raises CT win probability)
- `lag_04__CT_utility_damage_last_5s`: coefficient `-0.001027` (lowers CT win probability)
- `lag_04__utility_damage_diff_last_5s`: coefficient `-0.000856` (lowers CT win probability)
- `lag_00__CT1__flash`: coefficient `0.000734` (raises CT win probability)
- `lag_05__CT_utility_damage_last_5s`: coefficient `0.000624` (raises CT win probability)
- `lag_00__CT1__utility_total`: coefficient `0.000606` (raises CT win probability)
- `lag_06__CT_utility_damage_last_5s`: coefficient `0.000602` (raises CT win probability)
- `lag_14__T_smokes_last_5s`: coefficient `0.000546` (raises CT win probability)
- `lag_09__T_smokes_last_5s`: coefficient `0.000523` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_01__T_place_RAFTERS`: coefficient `0.004182` (raises CT win probability)
- `lag_03__T_place_RAFTERS`: coefficient `0.002490` (raises CT win probability)
- `lag_01__T_place_MINI`: coefficient `0.002340` (raises CT win probability)
- `lag_02__T_place_RAFTERS`: coefficient `0.002182` (raises CT win probability)
- `lag_00__T_place_RAFTERS`: coefficient `-0.001810` (lowers CT win probability)
- `lag_00__T_place_HELL`: coefficient `-0.001768` (lowers CT win probability)
- `lag_00__T_place_HEAVEN`: coefficient `-0.001647` (lowers CT win probability)
- `lag_07__CT_place_HUT`: coefficient `-0.001563` (lowers CT win probability)
- `lag_05__T_place_RAFTERS`: coefficient `-0.001478` (lowers CT win probability)
- `lag_04__T_place_MINI`: coefficient `-0.001413` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `60950`, seconds `96.00`, LSTM delta `+0.4257`

Top all feature movements:
- `lag_01__T_place_RAFTERS`: contribution `+0.109446`
- `lag_02__T_place_RAFTERS`: contribution `+0.057113`
- `lag_00__T_place_RAFTERS`: contribution `+0.047374`
- `lag_01__T_place_MINI`: contribution `+0.032551`
- `lag_00__T_place_HEAVEN`: contribution `+0.020203`

Top utility-only movements:
- `lag_01__CT_utility_damage_last_5s`: contribution `+0.019501`
- `lag_01__utility_damage_diff_last_5s`: contribution `+0.013080`

### tick `61046`, seconds `97.50`, LSTM delta `-0.2322`

Top all feature movements:
- `lag_03__T_place_RAFTERS`: contribution `-0.065158`
- `lag_00__T_place_RAFTERS`: contribution `+0.047374`
- `lag_05__T_place_RAFTERS`: contribution `-0.038679`
- `lag_00__T_place_HEAVEN`: contribution `-0.020203`
- `lag_04__T_place_MINI`: contribution `-0.019657`

Top utility-only movements:
- `lag_04__CT_utility_damage_last_5s`: contribution `-0.011304`
- `lag_04__utility_damage_diff_last_5s`: contribution `-0.007732`
- `lag_00__CT1__flash`: contribution `-0.002628`

### tick `61398`, seconds `103.00`, LSTM delta `-0.2241`

Top all feature movements:
- `lag_03__T_place_RAFTERS`: contribution `-0.065158`
- `lag_14__T_place_RAFTERS`: contribution `-0.027945`
- `lag_11__T_place_RAFTERS`: contribution `-0.026971`
- `lag_09__T_place_MINI`: contribution `-0.009344`
- `lag_05__CT_utility_damage_last_5s`: contribution `-0.006872`

Top utility-only movements:
- `lag_05__CT_utility_damage_last_5s`: contribution `-0.006872`
- `lag_05__utility_damage_diff_last_5s`: contribution `-0.004528`
- `lag_15__CT_utility_damage_last_5s`: contribution `-0.003937`
- `lag_15__utility_damage_diff_last_5s`: contribution `-0.002762`

### tick `61110`, seconds `98.50`, LSTM delta `+0.1542`

Top all feature movements:
- `lag_01__T_place_RAFTERS`: contribution `+0.109446`
- `lag_02__T_place_RAFTERS`: contribution `-0.057113`
- `lag_05__T_place_RAFTERS`: contribution `+0.038679`
- `lag_06__T_place_RAFTERS`: contribution `+0.028398`
- `lag_02__T_place_HEAVEN`: contribution `-0.013530`

Top utility-only movements:
- `lag_06__CT_utility_damage_last_5s`: contribution `+0.006622`
- `lag_06__utility_damage_diff_last_5s`: contribution `+0.004340`

### tick `61078`, seconds `98.00`, LSTM delta `-0.1354`

Top all feature movements:
- `lag_01__T_place_RAFTERS`: contribution `-0.109446`
- `lag_00__T_place_RAFTERS`: contribution `-0.047374`
- `lag_05__T_place_RAFTERS`: contribution `-0.038679`
- `lag_06__T_place_RAFTERS`: contribution `+0.028398`
- `lag_00__T_place_HEAVEN`: contribution `+0.020203`

Top utility-only movements:
- `lag_05__CT_utility_damage_last_5s`: contribution `+0.006872`
- `lag_05__utility_damage_diff_last_5s`: contribution `+0.004528`
