# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-faze-vs-heroic-dust2-PtQF8ASKD1754yZQHk6148/faze-vs-heroic-dust2.csv`
- round_num: `2`

## Largest probability jumps

- tick `14882`, seconds `47.00`, LSTM `0.9575`, delta `+0.0478`
- tick `13730`, seconds `29.00`, LSTM `0.8846`, delta `-0.0187`
- tick `14754`, seconds `45.00`, LSTM `0.8984`, delta `-0.0176`
- tick `14306`, seconds `38.00`, LSTM `0.9000`, delta `+0.0154`
- tick `14626`, seconds `43.00`, LSTM `0.9028`, delta `-0.0136`
- tick `13954`, seconds `32.50`, LSTM `0.8958`, delta `+0.0133`
- tick `11906`, seconds `0.50`, LSTM `0.9223`, delta `+0.0131`
- tick `14914`, seconds `47.50`, LSTM `0.9703`, delta `+0.0127`
- tick `14178`, seconds `36.00`, LSTM `0.8833`, delta `-0.0127`
- tick `14370`, seconds `39.00`, LSTM `0.9074`, delta `+0.0124`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.000481`, |coef| `0.000481`
- `lag_00__T_place_EXTENDEDA`: coefficient `-0.000459`, |coef| `0.000459`
- `lag_04__T_place_EXTENDEDA`: coefficient `0.000455`, |coef| `0.000455`
- `lag_00__CT_damage_last_5s`: coefficient `0.000447`, |coef| `0.000447`
- `lag_00__damage_diff_last_5s`: coefficient `0.000428`, |coef| `0.000428`
- `lag_00__T_walking_count`: coefficient `-0.000421`, |coef| `0.000421`
- `lag_12__T_place_LOWERTUNNEL`: coefficient `-0.000417`, |coef| `0.000417`
- `lag_00__kill_diff_last_3s`: coefficient `0.000367`, |coef| `0.000367`
- `lag_00__CT_walking_count`: coefficient `-0.000355`, |coef| `0.000355`
- `lag_00__T4__is_walking`: coefficient `-0.000341`, |coef| `0.000341`
- `lag_05__T_place_TUNNELSTAIRS`: coefficient `-0.000328`, |coef| `0.000328`
- `lag_00__T1__alive`: coefficient `-0.000322`, |coef| `0.000322`
- `lag_00__T1__hp`: coefficient `-0.000317`, |coef| `0.000317`
- `lag_00__CT5__is_walking`: coefficient `-0.000317`, |coef| `0.000317`
- `lag_08__T3__is_walking`: coefficient `-0.000312`, |coef| `0.000312`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.000117` (lowers CT win probability)
- `lag_15__CT_B_site_active_infernos`: coefficient `0.000114` (raises CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `-0.000110` (lowers CT win probability)
- `lag_07__CT_active_infernos`: coefficient `0.000110` (raises CT win probability)
- `lag_12__T_flash_alpha_mean`: coefficient `-0.000092` (lowers CT win probability)
- `lag_11__CT_active_infernos`: coefficient `0.000092` (raises CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000090` (raises CT win probability)
- `lag_03__CT3__molly`: coefficient `-0.000086` (lowers CT win probability)
- `lag_07__CT_B_site_active_infernos`: coefficient `0.000086` (raises CT win probability)
- `lag_15__CT_active_infernos`: coefficient `0.000084` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.000481` (raises CT win probability)
- `lag_00__T_place_EXTENDEDA`: coefficient `-0.000459` (lowers CT win probability)
- `lag_04__T_place_EXTENDEDA`: coefficient `0.000455` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.000447` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000428` (raises CT win probability)
- `lag_00__T_walking_count`: coefficient `-0.000421` (lowers CT win probability)
- `lag_12__T_place_LOWERTUNNEL`: coefficient `-0.000417` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000367` (raises CT win probability)
- `lag_00__CT_walking_count`: coefficient `-0.000355` (lowers CT win probability)
- `lag_00__T4__is_walking`: coefficient `-0.000341` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `14882`, seconds `47.00`, LSTM delta `+0.0478`

Top all feature movements:
- `lag_00__T_place_EXTENDEDA`: contribution `+0.002277`
- `lag_04__T_place_EXTENDEDA`: contribution `+0.002257`
- `lag_00__CT_damage_last_5s`: contribution `+0.001802`
- `lag_12__T_place_LOWERTUNNEL`: contribution `+0.001801`
- `lag_00__damage_diff_last_5s`: contribution `+0.001788`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `13730`, seconds `29.00`, LSTM delta `-0.0187`

Top all feature movements:
- `lag_00__CT_place_BDOORS`: contribution `-0.001225`
- `lag_06__T2__duck_amount`: contribution `-0.001029`
- `lag_03__CT3__duck_amount`: contribution `-0.000844`
- `lag_00__CT5__is_walking`: contribution `-0.000759`
- `lag_02__CT2__is_walking`: contribution `-0.000716`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `14754`, seconds `45.00`, LSTM delta `-0.0176`

Top all feature movements:
- `lag_00__T_place_EXTENDEDA`: contribution `-0.002277`
- `lag_15__T_place_SHORTSTAIRS`: contribution `+0.001295`
- `lag_14__T1__duck_amount`: contribution `-0.000841`
- `lag_12__CT3__duck_amount`: contribution `-0.000830`
- `lag_04__T_place_MIDDOORS`: contribution `-0.000780`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `14306`, seconds `38.00`, LSTM delta `+0.0154`

Top all feature movements:
- `lag_05__T_place_TUNNELSTAIRS`: contribution `+0.002293`
- `lag_02__CT_place_BDOORS`: contribution `+0.001250`
- `lag_00__T_walking_count`: contribution `+0.001006`
- `lag_00__T4__is_walking`: contribution `+0.000787`
- `lag_00__CT5__is_walking`: contribution `+0.000759`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `14626`, seconds `43.00`, LSTM delta `-0.0136`

Top all feature movements:
- `lag_15__T_place_TUNNELSTAIRS`: contribution `-0.000911`
- `lag_00__CT5__is_walking`: contribution `-0.000759`
- `lag_02__CT2__is_walking`: contribution `-0.000716`
- `lag_03__T3__is_walking`: contribution `-0.000681`
- `lag_11__T3__is_walking`: contribution `-0.000662`

Top utility-only movements:
- No utility movement among the top local contributors.
