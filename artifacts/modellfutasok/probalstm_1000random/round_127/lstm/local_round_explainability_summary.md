# Local Round Explainability

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-flyquest-vs-legacy-bo3-FlEa8e0vdBrf1ft_mNbThh/flyquest-vs-legacy-m2-nuke.csv`
- round_num: `8`

## Largest probability jumps

- tick `83717`, seconds `91.00`, LSTM `0.1181`, delta `-0.3189`
- tick `83653`, seconds `90.00`, LSTM `0.4198`, delta `+0.2810`
- tick `79269`, seconds `21.50`, LSTM `0.1351`, delta `-0.2291`
- tick `79173`, seconds `20.00`, LSTM `0.3972`, delta `+0.0821`
- tick `77925`, seconds `0.50`, LSTM `0.1465`, delta `-0.0762`
- tick `83525`, seconds `88.00`, LSTM `0.1194`, delta `+0.0638`
- tick `84453`, seconds `102.50`, LSTM `0.1028`, delta `+0.0625`
- tick `84645`, seconds `105.50`, LSTM `0.0803`, delta `-0.0569`
- tick `79237`, seconds `21.00`, LSTM `0.3642`, delta `-0.0512`
- tick `78629`, seconds `11.50`, LSTM `0.2079`, delta `+0.0509`

## Top 15 local ridge features

- `lag_02__T_shots_fired_sum`: coefficient `0.004135`, |coef| `0.004135`
- `lag_05__CT_shots_fired_sum`: coefficient `0.004010`, |coef| `0.004010`
- `lag_05__CT2__shots_fired`: coefficient `0.002843`, |coef| `0.002843`
- `lag_03__CT_shots_fired_sum`: coefficient `-0.002278`, |coef| `0.002278`
- `lag_00__T_shots_fired_sum`: coefficient `-0.002258`, |coef| `0.002258`
- `lag_09__CT_place_CONTROL`: coefficient `-0.001796`, |coef| `0.001796`
- `lag_15__T_place_CONTROL`: coefficient `0.001672`, |coef| `0.001672`
- `lag_03__CT2__shots_fired`: coefficient `-0.001644`, |coef| `0.001644`
- `lag_13__CT_place_MINI`: coefficient `0.001628`, |coef| `0.001628`
- `lag_06__CT_place_HUT`: coefficient `0.001424`, |coef| `0.001424`
- `lag_13__T_place_CONTROL`: coefficient `-0.001313`, |coef| `0.001313`
- `lag_00__CT_place_LOBBY`: coefficient `0.001228`, |coef| `0.001228`
- `lag_12__T_place_CONTROL`: coefficient `-0.001227`, |coef| `0.001227`
- `lag_04__CT_shots_fired_sum`: coefficient `0.001161`, |coef| `0.001161`
- `lag_06__T3__flash_duration`: coefficient `-0.001159`, |coef| `0.001159`

## Top 10 utility ridge features

- `lag_06__T3__flash_duration`: coefficient `-0.001159` (lowers CT win probability)
- `lag_06__T2__flash_duration`: coefficient `-0.000997` (lowers CT win probability)
- `lag_06__T_flash_duration_sum`: coefficient `-0.000889` (lowers CT win probability)
- `lag_03__T2__flash_duration`: coefficient `0.000651` (raises CT win probability)
- `lag_02__T_A_site_active_infernos`: coefficient `0.000632` (raises CT win probability)
- `lag_03__T3__flash_duration`: coefficient `0.000613` (raises CT win probability)
- `lag_02__T_B_site_active_infernos`: coefficient `0.000603` (raises CT win probability)
- `lag_03__CT_utility_damage_last_5s`: coefficient `-0.000548` (lowers CT win probability)
- `lag_03__T_flash_duration_sum`: coefficient `0.000522` (raises CT win probability)
- `lag_08__T1__smoke`: coefficient `-0.000519` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_02__T_shots_fired_sum`: coefficient `0.004135` (raises CT win probability)
- `lag_05__CT_shots_fired_sum`: coefficient `0.004010` (raises CT win probability)
- `lag_05__CT2__shots_fired`: coefficient `0.002843` (raises CT win probability)
- `lag_03__CT_shots_fired_sum`: coefficient `-0.002278` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.002258` (lowers CT win probability)
- `lag_09__CT_place_CONTROL`: coefficient `-0.001796` (lowers CT win probability)
- `lag_15__T_place_CONTROL`: coefficient `0.001672` (raises CT win probability)
- `lag_03__CT2__shots_fired`: coefficient `-0.001644` (lowers CT win probability)
- `lag_13__CT_place_MINI`: coefficient `0.001628` (raises CT win probability)
- `lag_06__CT_place_HUT`: coefficient `0.001424` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `83717`, seconds `91.00`, LSTM delta `-0.3189`

Top all feature movements:
- `lag_02__T_shots_fired_sum`: contribution `-0.068212`
- `lag_05__CT_shots_fired_sum`: contribution `-0.050148`
- `lag_05__CT2__shots_fired`: contribution `-0.025433`
- `lag_15__T_place_CONTROL`: contribution `-0.011883`
- `lag_04__T_shots_fired_sum`: contribution `-0.005718`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `83653`, seconds `90.00`, LSTM delta `+0.2810`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.037250`
- `lag_03__CT_shots_fired_sum`: contribution `+0.028487`
- `lag_02__T_shots_fired_sum`: contribution `+0.024804`
- `lag_05__CT_shots_fired_sum`: contribution `+0.019502`
- `lag_03__CT2__shots_fired`: contribution `+0.014713`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `79269`, seconds `21.50`, LSTM delta `-0.2291`

Top all feature movements:
- `lag_09__CT_place_CONTROL`: contribution `-0.018642`
- `lag_06__CT_place_HUT`: contribution `-0.013885`
- `lag_13__CT_place_MINI`: contribution `-0.009979`
- `lag_06__CT_place_LOBBY`: contribution `-0.008976`
- `lag_06__T3__flash_duration`: contribution `-0.008187`

Top utility-only movements:
- `lag_06__T3__flash_duration`: contribution `-0.008187`
- `lag_06__T2__flash_duration`: contribution `-0.006280`
- `lag_06__T_flash_duration_sum`: contribution `-0.004907`

### tick `79173`, seconds `20.00`, LSTM delta `+0.0821`

Top all feature movements:
- `lag_00__CT_place_LOBBY`: contribution `+0.010051`
- `lag_00__CT_place_HUT`: contribution `+0.008157`
- `lag_01__CT_place_LOBBY`: contribution `-0.007330`
- `lag_01__CT_place_HUT`: contribution `-0.005931`
- `lag_03__T3__flash_duration`: contribution `+0.004328`

Top utility-only movements:
- `lag_03__T3__flash_duration`: contribution `+0.004328`
- `lag_03__T2__flash_duration`: contribution `+0.004099`
- `lag_03__T_flash_duration_sum`: contribution `+0.002882`

### tick `77925`, seconds `0.50`, LSTM delta `-0.0762`

Top all feature movements:
- `lag_00__T_velocity_mean`: contribution `-0.003621`
- `lag_01__T_closest_enemy_dist`: contribution `-0.002511`
- `lag_01__CT_place_CTSPAWN`: contribution `-0.002496`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.002453`
- `lag_01__T_place_TSPAWN`: contribution `-0.002409`

Top utility-only movements:
- `lag_01__flash_inv_diff`: contribution `-0.000779`
- `lag_01__molly_inv_diff`: contribution `-0.000708`
- `lag_01__T5__flash`: contribution `-0.000704`
- `lag_01__utility_inv_diff`: contribution `-0.000699`
- `lag_01__T1__molly`: contribution `-0.000672`
