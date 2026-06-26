# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-legacy-vs-vitality-bo3-43WNFDazpfbmBN3Sj5hWmP/vitality-vs-legacy-m2-dust2.csv`
- round_num: `18`

## Largest probability jumps

- tick `137693`, seconds `96.00`, LSTM `0.2015`, delta `-0.3187`
- tick `137469`, seconds `92.50`, LSTM `0.5962`, delta `-0.1759`
- tick `137789`, seconds `97.50`, LSTM `0.2374`, delta `+0.1316`
- tick `136253`, seconds `73.50`, LSTM `0.7567`, delta `-0.1067`
- tick `136221`, seconds `73.00`, LSTM `0.8634`, delta `+0.1014`
- tick `136349`, seconds `75.00`, LSTM `0.7860`, delta `+0.0601`
- tick `137437`, seconds `92.00`, LSTM `0.7721`, delta `+0.0595`
- tick `136029`, seconds `70.00`, LSTM `0.8793`, delta `-0.0553`
- tick `137757`, seconds `97.00`, LSTM `0.1058`, delta `-0.0517`
- tick `131741`, seconds `3.00`, LSTM `0.7167`, delta `-0.0513`

## Top 15 local ridge features

- `lag_00__CT_place_UPPERTUNNEL`: coefficient `0.002851`, |coef| `0.002851`
- `lag_14__CT_place_HOLE`: coefficient `-0.002287`, |coef| `0.002287`
- `lag_10__CT_place_HOLE`: coefficient `-0.002211`, |coef| `0.002211`
- `lag_00__T_damage_last_5s`: coefficient `-0.002039`, |coef| `0.002039`
- `lag_00__damage_diff_last_5s`: coefficient `0.002023`, |coef| `0.002023`
- `lag_08__CT_B_site_active_infernos`: coefficient `-0.001818`, |coef| `0.001818`
- `lag_13__CT_place_HOLE`: coefficient `-0.001803`, |coef| `0.001803`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001723`, |coef| `0.001723`
- `lag_00__T_flashes_last_5s`: coefficient `-0.001712`, |coef| `0.001712`
- `lag_00__T_kills_last_3s`: coefficient `-0.001606`, |coef| `0.001606`
- `lag_07__CT_place_BDOORS`: coefficient `0.001584`, |coef| `0.001584`
- `lag_00__kill_diff_last_3s`: coefficient `0.001568`, |coef| `0.001568`
- `lag_08__CT1__is_scoped`: coefficient `-0.001566`, |coef| `0.001566`
- `lag_07__T_kills_last_3s`: coefficient `-0.001527`, |coef| `0.001527`
- `lag_05__CT_B_site_active_infernos`: coefficient `0.001336`, |coef| `0.001336`

## Top 10 utility ridge features

- `lag_08__CT_B_site_active_infernos`: coefficient `-0.001818` (lowers CT win probability)
- `lag_00__T_flashes_last_5s`: coefficient `-0.001712` (lowers CT win probability)
- `lag_05__CT_B_site_active_infernos`: coefficient `0.001336` (raises CT win probability)
- `lag_09__CT_B_site_active_infernos`: coefficient `-0.001207` (lowers CT win probability)
- `lag_12__CT1__molly`: coefficient `0.001089` (raises CT win probability)
- `lag_08__CT_active_infernos`: coefficient `-0.000977` (lowers CT win probability)
- `lag_05__CT_active_infernos`: coefficient `0.000907` (raises CT win probability)
- `lag_01__T_flashes_last_5s`: coefficient `-0.000861` (lowers CT win probability)
- `lag_08__T_flashes_last_5s`: coefficient `-0.000858` (lowers CT win probability)
- `lag_15__CT_B_site_active_infernos`: coefficient `-0.000800` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_UPPERTUNNEL`: coefficient `0.002851` (raises CT win probability)
- `lag_14__CT_place_HOLE`: coefficient `-0.002287` (lowers CT win probability)
- `lag_10__CT_place_HOLE`: coefficient `-0.002211` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002039` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002023` (raises CT win probability)
- `lag_13__CT_place_HOLE`: coefficient `-0.001803` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001723` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001606` (lowers CT win probability)
- `lag_07__CT_place_BDOORS`: coefficient `0.001584` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001568` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `137693`, seconds `96.00`, LSTM delta `-0.3187`

Top all feature movements:
- `lag_10__CT_place_HOLE`: contribution `-0.024688`
- `lag_00__CT_place_UPPERTUNNEL`: contribution `-0.021868`
- `lag_13__CT_place_HOLE`: contribution `-0.020130`
- `lag_07__CT_place_BDOORS`: contribution `-0.007622`
- `lag_08__CT1__is_scoped`: contribution `-0.006708`

Top utility-only movements:
- `lag_08__CT_B_site_active_infernos`: contribution `-0.006245`
- `lag_05__CT_B_site_active_infernos`: contribution `-0.004590`

### tick `137469`, seconds `92.50`, LSTM delta `-0.1759`

Top all feature movements:
- `lag_03__CT_place_HOLE`: contribution `-0.014310`
- `lag_06__CT_place_HOLE`: contribution `-0.013963`
- `lag_04__CT_place_HOLE`: contribution `-0.009599`
- `lag_00__T_shots_fired_sum`: contribution `-0.006460`
- `lag_06__CT_place_BDOORS`: contribution `-0.006219`

Top utility-only movements:
- `lag_09__CT_B_site_active_infernos`: contribution `-0.004146`
- `lag_13__T2__flash_duration`: contribution `-0.004141`

### tick `137789`, seconds `97.50`, LSTM delta `+0.1316`

Top all feature movements:
- `lag_14__CT_place_HOLE`: contribution `+0.025531`
- `lag_13__CT_place_HOLE`: contribution `-0.020130`
- `lag_03__CT_place_UPPERTUNNEL`: contribution `+0.006610`
- `lag_00__T_shots_fired_sum`: contribution `+0.006460`
- `lag_08__CT_B_site_active_infernos`: contribution `+0.006245`

Top utility-only movements:
- `lag_08__CT_B_site_active_infernos`: contribution `+0.006245`
- `lag_08__CT_active_infernos`: contribution `+0.002252`

### tick `136253`, seconds `73.50`, LSTM delta `-0.1067`

Top all feature movements:
- `lag_08__T_place_BDOORS`: contribution `-0.005542`
- `lag_00__T_kills_last_3s`: contribution `-0.005087`
- `lag_07__T_kills_last_3s`: contribution `-0.004838`
- `lag_08__CT_place_OUTSIDELONG`: contribution `-0.004490`
- `lag_00__kill_diff_last_3s`: contribution `-0.003774`

Top utility-only movements:
- `lag_13__T2__flash_duration`: contribution `+0.001955`
- `lag_08__CT5__flash_duration`: contribution `-0.001811`

### tick `136221`, seconds `73.00`, LSTM delta `+0.1014`

Top all feature movements:
- `lag_12__CT_place_OUTSIDELONG`: contribution `+0.009709`
- `lag_07__T_place_BDOORS`: contribution `+0.009210`
- `lag_00__kill_diff_last_3s`: contribution `+0.007548`
- `lag_07__CT_place_OUTSIDELONG`: contribution `+0.006588`
- `lag_00__T_kills_last_3s`: contribution `+0.005087`

Top utility-only movements:
- `lag_07__CT5__flash_duration`: contribution `+0.001614`
- `lag_00__T3__flash_duration`: contribution `+0.001276`
