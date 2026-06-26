# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-legacy-vs-vitality-bo3-43WNFDazpfbmBN3Sj5hWmP/vitality-vs-legacy-m2-dust2.csv`
- round_num: `14`

## Largest probability jumps

- tick `111703`, seconds `47.00`, LSTM `0.4252`, delta `+0.2604`
- tick `112503`, seconds `59.50`, LSTM `0.8924`, delta `+0.2473`
- tick `112023`, seconds `52.00`, LSTM `0.5810`, delta `+0.1689`
- tick `112311`, seconds `56.50`, LSTM `0.7070`, delta `+0.0935`
- tick `112215`, seconds `55.00`, LSTM `0.5542`, delta `+0.0855`
- tick `112055`, seconds `52.50`, LSTM `0.4976`, delta `-0.0834`
- tick `112087`, seconds `53.00`, LSTM `0.4204`, delta `-0.0772`
- tick `111991`, seconds `51.50`, LSTM `0.4120`, delta `-0.0764`
- tick `109815`, seconds `17.50`, LSTM `0.2748`, delta `-0.0569`
- tick `112151`, seconds `54.00`, LSTM `0.4691`, delta `+0.0537`

## Top 15 local ridge features

- `lag_00__CT2__is_scoped`: coefficient `0.002187`, |coef| `0.002187`
- `lag_06__CT3__duck_amount`: coefficient `-0.001837`, |coef| `0.001837`
- `lag_07__CT_place_HOLE`: coefficient `-0.001787`, |coef| `0.001787`
- `lag_05__CT1__flash_duration`: coefficient `0.001676`, |coef| `0.001676`
- `lag_02__T_shots_fired_sum`: coefficient `-0.001639`, |coef| `0.001639`
- `lag_11__T_shots_fired_sum`: coefficient `-0.001623`, |coef| `0.001623`
- `lag_11__T5__shots_fired`: coefficient `-0.001564`, |coef| `0.001564`
- `lag_12__CT2__is_scoped`: coefficient `0.001437`, |coef| `0.001437`
- `lag_00__kill_diff_last_3s`: coefficient `0.001421`, |coef| `0.001421`
- `lag_00__CT_scoped_count`: coefficient `0.001409`, |coef| `0.001409`
- `lag_11__CT_place_HOLE`: coefficient `0.001388`, |coef| `0.001388`
- `lag_13__CT3__flash_duration`: coefficient `-0.001377`, |coef| `0.001377`
- `lag_00__CT_kills_last_3s`: coefficient `0.001370`, |coef| `0.001370`
- `lag_07__CT_place_BDOORS`: coefficient `0.001310`, |coef| `0.001310`
- `lag_12__T3__shots_fired`: coefficient `-0.001298`, |coef| `0.001298`

## Top 10 utility ridge features

- `lag_05__CT1__flash_duration`: coefficient `0.001676` (raises CT win probability)
- `lag_13__CT3__flash_duration`: coefficient `-0.001377` (lowers CT win probability)
- `lag_02__CT1__flash_duration`: coefficient `-0.000834` (lowers CT win probability)
- `lag_11__T_A_site_active_infernos`: coefficient `-0.000832` (lowers CT win probability)
- `lag_05__CT3__flash_duration`: coefficient `-0.000796` (lowers CT win probability)
- `lag_00__T1__smoke`: coefficient `-0.000777` (lowers CT win probability)
- `lag_13__CT_flash_duration_sum`: coefficient `-0.000747` (lowers CT win probability)
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.000740` (lowers CT win probability)
- `lag_14__CT1__flash_duration`: coefficient `-0.000725` (lowers CT win probability)
- `lag_00__T1__utility_total`: coefficient `-0.000679` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT2__is_scoped`: coefficient `0.002187` (raises CT win probability)
- `lag_06__CT3__duck_amount`: coefficient `-0.001837` (lowers CT win probability)
- `lag_07__CT_place_HOLE`: coefficient `-0.001787` (lowers CT win probability)
- `lag_02__T_shots_fired_sum`: coefficient `-0.001639` (lowers CT win probability)
- `lag_11__T_shots_fired_sum`: coefficient `-0.001623` (lowers CT win probability)
- `lag_11__T5__shots_fired`: coefficient `-0.001564` (lowers CT win probability)
- `lag_12__CT2__is_scoped`: coefficient `0.001437` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001421` (raises CT win probability)
- `lag_00__CT_scoped_count`: coefficient `0.001409` (raises CT win probability)
- `lag_11__CT_place_HOLE`: coefficient `0.001388` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `111703`, seconds `47.00`, LSTM delta `+0.2604`

Top all feature movements:
- `lag_07__CT_place_HOLE`: contribution `+0.019946`
- `lag_11__CT_place_HOLE`: contribution `+0.015501`
- `lag_05__CT1__flash_duration`: contribution `+0.012257`
- `lag_12__CT2__is_scoped`: contribution `+0.008795`
- `lag_13__CT3__flash_duration`: contribution `+0.008071`

Top utility-only movements:
- `lag_05__CT1__flash_duration`: contribution `+0.012257`
- `lag_13__CT3__flash_duration`: contribution `+0.008071`

### tick `112503`, seconds `59.50`, LSTM delta `+0.2473`

Top all feature movements:
- `lag_11__T_shots_fired_sum`: contribution `+0.031629`
- `lag_11__T5__shots_fired`: contribution `+0.024992`
- `lag_12__T3__shots_fired`: contribution `+0.017288`
- `lag_12__T_shots_fired_sum`: contribution `+0.013956`
- `lag_00__CT2__is_scoped`: contribution `+0.013386`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `112023`, seconds `52.00`, LSTM delta `+0.1689`

Top all feature movements:
- `lag_00__CT2__is_scoped`: contribution `+0.013386`
- `lag_00__T_shots_fired_sum`: contribution `-0.009329`
- `lag_05__CT2__is_scoped`: contribution `+0.007442`
- `lag_02__T_shots_fired_sum`: contribution `+0.007374`
- `lag_06__CT3__duck_amount`: contribution `+0.006178`

Top utility-only movements:
- `lag_02__CT1__flash_duration`: contribution `+0.006103`
- `lag_15__CT1__flash_duration`: contribution `+0.004128`

### tick `112311`, seconds `56.50`, LSTM delta `+0.0935`

Top all feature movements:
- `lag_06__T_shots_fired_sum`: contribution `+0.010713`
- `lag_12__CT2__is_scoped`: contribution `-0.008795`
- `lag_07__T_shots_fired_sum`: contribution `+0.008073`
- `lag_05__T5__shots_fired`: contribution `+0.007449`
- `lag_11__T_shots_fired_sum`: contribution `+0.007299`

Top utility-only movements:
- `lag_11__CT1__flash_duration`: contribution `+0.002325`

### tick `112215`, seconds `55.00`, LSTM delta `+0.0855`

Top all feature movements:
- `lag_02__T_shots_fired_sum`: contribution `+0.031953`
- `lag_02__T5__shots_fired`: contribution `+0.012925`
- `lag_03__T3__shots_fired`: contribution `+0.008812`
- `lag_06__T_shots_fired_sum`: contribution `-0.006932`
- `lag_07__T_shots_fired_sum`: contribution `+0.006728`

Top utility-only movements:
- No utility movement among the top local contributors.
