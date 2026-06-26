# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-nrg-vs-vitality-bo3-7fVRmFXct71SEzAlz8IKvC/nrg-vs-vitality-m2-dust2.csv`
- round_num: `3`

## Largest probability jumps

- tick `19012`, seconds `7.00`, LSTM `0.8899`, delta `+0.0386`
- tick `19364`, seconds `12.50`, LSTM `0.9478`, delta `+0.0298`
- tick `21668`, seconds `48.50`, LSTM `0.9684`, delta `+0.0253`
- tick `19972`, seconds `22.00`, LSTM `0.9294`, delta `-0.0243`
- tick `19044`, seconds `7.50`, LSTM `0.9099`, delta `+0.0201`
- tick `21316`, seconds `43.00`, LSTM `0.9772`, delta `+0.0191`
- tick `20420`, seconds `29.00`, LSTM `0.9295`, delta `+0.0166`
- tick `18596`, seconds `0.50`, LSTM `0.8974`, delta `+0.0158`
- tick `21380`, seconds `44.00`, LSTM `0.9593`, delta `-0.0155`
- tick `18660`, seconds `1.50`, LSTM `0.8691`, delta `-0.0144`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.000413`, |coef| `0.000413`
- `lag_12__CT_place_HOLE`: coefficient `-0.000400`, |coef| `0.000400`
- `lag_00__CT_place_SHORTSTAIRS`: coefficient `0.000388`, |coef| `0.000388`
- `lag_00__damage_diff_last_5s`: coefficient `0.000367`, |coef| `0.000367`
- `lag_00__CT_place_EXTENDEDA`: coefficient `-0.000342`, |coef| `0.000342`
- `lag_00__CT_kills_last_3s`: coefficient `0.000314`, |coef| `0.000314`
- `lag_10__CT_place_EXTENDEDA`: coefficient `-0.000265`, |coef| `0.000265`
- `lag_06__CT_place_HOLE`: coefficient `-0.000253`, |coef| `0.000253`
- `lag_08__CT_place_MIDDOORS`: coefficient `0.000253`, |coef| `0.000253`
- `lag_00__CT_damage_last_5s`: coefficient `0.000249`, |coef| `0.000249`
- `lag_00__CT_shots_fired_sum`: coefficient `0.000241`, |coef| `0.000241`
- `lag_07__CT1__is_scoped`: coefficient `0.000227`, |coef| `0.000227`
- `lag_05__T_place_TSPAWN`: coefficient `-0.000227`, |coef| `0.000227`
- `lag_05__CT_B_site_active_infernos`: coefficient `0.000223`, |coef| `0.000223`
- `lag_11__CT_place_HOLE`: coefficient `-0.000222`, |coef| `0.000222`

## Top 10 utility ridge features

- `lag_05__CT_B_site_active_infernos`: coefficient `0.000223` (raises CT win probability)
- `lag_06__CT_B_site_active_infernos`: coefficient `0.000167` (raises CT win probability)
- `lag_07__T4__flash_duration`: coefficient `0.000157` (raises CT win probability)
- `lag_02__CT3__smoke`: coefficient `-0.000154` (lowers CT win probability)
- `lag_14__utility_inv_diff`: coefficient `0.000153` (raises CT win probability)
- `lag_14__smoke_inv_diff`: coefficient `0.000138` (raises CT win probability)
- `lag_14__molly_inv_diff`: coefficient `0.000135` (raises CT win probability)
- `lag_03__CT_molly_inv`: coefficient `-0.000134` (lowers CT win probability)
- `lag_03__molly_inv_diff`: coefficient `-0.000124` (lowers CT win probability)
- `lag_00__T4__flash_duration`: coefficient `0.000121` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.000413` (raises CT win probability)
- `lag_12__CT_place_HOLE`: coefficient `-0.000400` (lowers CT win probability)
- `lag_00__CT_place_SHORTSTAIRS`: coefficient `0.000388` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000367` (raises CT win probability)
- `lag_00__CT_place_EXTENDEDA`: coefficient `-0.000342` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000314` (raises CT win probability)
- `lag_10__CT_place_EXTENDEDA`: coefficient `-0.000265` (lowers CT win probability)
- `lag_06__CT_place_HOLE`: coefficient `-0.000253` (lowers CT win probability)
- `lag_08__CT_place_MIDDOORS`: coefficient `0.000253` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.000249` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `19012`, seconds `7.00`, LSTM delta `+0.0386`

Top all feature movements:
- `lag_08__CT_place_MIDDOORS`: contribution `+0.001460`
- `lag_00__kill_diff_last_3s`: contribution `+0.000993`
- `lag_02__CT_place_EXTENDEDA`: contribution `+0.000939`
- `lag_00__CT1__is_scoped`: contribution `+0.000932`
- `lag_00__CT_kills_last_3s`: contribution `+0.000908`

Top utility-only movements:
- `lag_14__utility_inv_diff`: contribution `+0.000540`
- `lag_14__smoke_inv_diff`: contribution `+0.000444`
- `lag_14__molly_inv_diff`: contribution `+0.000444`

### tick `19364`, seconds `12.50`, LSTM delta `+0.0298`

Top all feature movements:
- `lag_10__CT_place_EXTENDEDA`: contribution `+0.001490`
- `lag_02__T_place_TUNNELSTAIRS`: contribution `+0.001098`
- `lag_13__CT_place_EXTENDEDA`: contribution `+0.001003`
- `lag_00__kill_diff_last_3s`: contribution `+0.000993`
- `lag_00__CT1__is_scoped`: contribution `+0.000932`

Top utility-only movements:
- `lag_06__CT_B_site_active_infernos`: contribution `+0.000572`

### tick `21668`, seconds `48.50`, LSTM delta `+0.0253`

Top all feature movements:
- `lag_09__CT_shots_fired_sum`: contribution `+0.001153`
- `lag_00__kill_diff_last_3s`: contribution `+0.000993`
- `lag_00__CT_kills_last_3s`: contribution `+0.000908`
- `lag_00__damage_diff_last_5s`: contribution `+0.000721`
- `lag_02__CT_place_UNDERA`: contribution `+0.000643`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `19972`, seconds `22.00`, LSTM delta `-0.0243`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.000993`
- `lag_07__CT1__is_scoped`: contribution `-0.000974`
- `lag_00__damage_diff_last_5s`: contribution `-0.000829`
- `lag_05__CT_B_site_active_infernos`: contribution `-0.000766`
- `lag_07__T_place_LOWERTUNNEL`: contribution `-0.000724`

Top utility-only movements:
- `lag_05__CT_B_site_active_infernos`: contribution `-0.000766`
- `lag_02__CT3__smoke`: contribution `-0.000341`

### tick `19044`, seconds `7.50`, LSTM delta `+0.0201`

Top all feature movements:
- `lag_00__CT_place_SHORTSTAIRS`: contribution `+0.002164`
- `lag_00__CT_place_EXTENDEDA`: contribution `+0.001922`
- `lag_09__CT_place_MIDDOORS`: contribution `+0.000928`
- `lag_03__CT_place_EXTENDEDA`: contribution `+0.000855`
- `lag_01__CT_place_BDOORS`: contribution `-0.000797`

Top utility-only movements:
- `lag_15__utility_inv_diff`: contribution `+0.000357`
- `lag_15__smoke_inv_diff`: contribution `+0.000328`
