# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-lynn-vision-vs-vitality-bo3-ZpOL0o26IrRvvgFRbFxVou/lynn-vision-vs-vitality-m1-dust2.csv`
- round_num: `14`

## Largest probability jumps

- tick `115631`, seconds `96.00`, LSTM `0.9269`, delta `+0.0477`
- tick `110287`, seconds `12.50`, LSTM `0.9530`, delta `+0.0406`
- tick `115151`, seconds `88.50`, LSTM `0.8818`, delta `-0.0380`
- tick `111183`, seconds `26.50`, LSTM `0.9413`, delta `-0.0343`
- tick `115823`, seconds `99.00`, LSTM `0.9681`, delta `+0.0298`
- tick `115503`, seconds `94.00`, LSTM `0.8870`, delta `+0.0246`
- tick `115471`, seconds `93.50`, LSTM `0.8624`, delta `-0.0238`
- tick `111983`, seconds `39.00`, LSTM `0.9518`, delta `+0.0221`
- tick `112207`, seconds `42.50`, LSTM `0.9001`, delta `-0.0207`
- tick `115695`, seconds `97.00`, LSTM `0.9472`, delta `+0.0195`

## Top 15 local ridge features

- `lag_04__T_place_ARAMP`: coefficient `-0.000897`, |coef| `0.000897`
- `lag_00__T_bomb_zone_count`: coefficient `-0.000877`, |coef| `0.000877`
- `lag_07__CT_place_SHORTSTAIRS`: coefficient `0.000838`, |coef| `0.000838`
- `lag_00__T1__is_scoped`: coefficient `0.000728`, |coef| `0.000728`
- `lag_00__kill_diff_last_3s`: coefficient `0.000656`, |coef| `0.000656`
- `lag_00__T_place_ARAMP`: coefficient `-0.000635`, |coef| `0.000635`
- `lag_05__T_place_ARAMP`: coefficient `0.000571`, |coef| `0.000571`
- `lag_00__CT_kills_last_3s`: coefficient `0.000562`, |coef| `0.000562`
- `lag_00__CT3__is_walking`: coefficient `-0.000543`, |coef| `0.000543`
- `lag_15__T_bomb_zone_count`: coefficient `0.000536`, |coef| `0.000536`
- `lag_01__CT_place_HOLE`: coefficient `0.000535`, |coef| `0.000535`
- `lag_11__CT_place_SIDE`: coefficient `-0.000504`, |coef| `0.000504`
- `lag_09__CT_place_SIDE`: coefficient `-0.000500`, |coef| `0.000500`
- `lag_07__CT_place_CATWALK`: coefficient `-0.000481`, |coef| `0.000481`
- `lag_06__CT_place_LOWERTUNNEL`: coefficient `0.000447`, |coef| `0.000447`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.000234` (lowers CT win probability)
- `lag_03__CT4__smoke`: coefficient `-0.000144` (lowers CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000141` (raises CT win probability)
- `lag_01__CT4__smoke`: coefficient `0.000139` (raises CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `-0.000114` (lowers CT win probability)
- `lag_01__CT2__molly`: coefficient `0.000114` (raises CT win probability)
- `lag_06__CT2__flash`: coefficient `-0.000112` (lowers CT win probability)
- `lag_01__CT2__utility_total`: coefficient `0.000112` (raises CT win probability)
- `lag_01__CT2__smoke`: coefficient `0.000110` (raises CT win probability)
- `lag_02__CT_active_smokes`: coefficient `-0.000106` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_04__T_place_ARAMP`: coefficient `-0.000897` (lowers CT win probability)
- `lag_00__T_bomb_zone_count`: coefficient `-0.000877` (lowers CT win probability)
- `lag_07__CT_place_SHORTSTAIRS`: coefficient `0.000838` (raises CT win probability)
- `lag_00__T1__is_scoped`: coefficient `0.000728` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000656` (raises CT win probability)
- `lag_00__T_place_ARAMP`: coefficient `-0.000635` (lowers CT win probability)
- `lag_05__T_place_ARAMP`: coefficient `0.000571` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000562` (raises CT win probability)
- `lag_00__CT3__is_walking`: coefficient `-0.000543` (lowers CT win probability)
- `lag_15__T_bomb_zone_count`: coefficient `0.000536` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `115631`, seconds `96.00`, LSTM delta `+0.0477`

Top all feature movements:
- `lag_04__T_place_ARAMP`: contribution `+0.008118`
- `lag_05__T_place_ARAMP`: contribution `+0.005165`
- `lag_07__CT_place_SHORTSTAIRS`: contribution `+0.004669`
- `lag_15__T_bomb_zone_count`: contribution `+0.003121`
- `lag_07__CT_place_CATWALK`: contribution `+0.001914`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `110287`, seconds `12.50`, LSTM delta `+0.0406`

Top all feature movements:
- `lag_07__CT_place_SHORTSTAIRS`: contribution `+0.004669`
- `lag_02__CT_place_TUNNELSTAIRS`: contribution `+0.004251`
- `lag_09__CT_place_HOLE`: contribution `+0.003079`
- `lag_11__CT_place_HOLE`: contribution `+0.002793`
- `lag_07__CT_place_EXTENDEDA`: contribution `+0.001670`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `115151`, seconds `88.50`, LSTM delta `-0.0380`

Top all feature movements:
- `lag_01__CT_place_HOLE`: contribution `-0.005971`
- `lag_00__T_bomb_zone_count`: contribution `-0.005105`
- `lag_06__CT_place_HOLE`: contribution `-0.004159`
- `lag_10__CT_place_OUTSIDELONG`: contribution `-0.001916`
- `lag_09__CT1__duck_amount`: contribution `-0.001414`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `111183`, seconds `26.50`, LSTM delta `-0.0343`

Top all feature movements:
- `lag_07__CT_place_SHORTSTAIRS`: contribution `-0.004669`
- `lag_06__CT_place_LOWERTUNNEL`: contribution `-0.003289`
- `lag_12__T2__is_scoped`: contribution `-0.002249`
- `lag_07__CT_place_EXTENDEDA`: contribution `-0.001670`
- `lag_00__kill_diff_last_3s`: contribution `-0.001580`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `115823`, seconds `99.00`, LSTM delta `+0.0298`

Top all feature movements:
- `lag_01__CT_place_HOLE`: contribution `+0.005971`
- `lag_10__T_place_ARAMP`: contribution `+0.003167`
- `lag_04__T_bomb_zone_count`: contribution `+0.001152`
- `lag_07__T4__duck_amount`: contribution `+0.001106`
- `lag_08__CT4__is_walking`: contribution `+0.000993`

Top utility-only movements:
- No utility movement among the top local contributors.
