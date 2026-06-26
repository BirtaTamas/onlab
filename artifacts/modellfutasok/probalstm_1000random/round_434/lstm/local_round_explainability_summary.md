# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-spirit-vs-vitality-bo3-KtWhzrlsNkWCCS0U9BIlr3/spirit-vs-vitality-m2-nuke.csv`
- round_num: `27`

## Largest probability jumps

- tick `243626`, seconds `30.50`, LSTM `0.9294`, delta `+0.1931`
- tick `243370`, seconds `26.50`, LSTM `0.8260`, delta `+0.1396`
- tick `243274`, seconds `25.00`, LSTM `0.6539`, delta `+0.1121`
- tick `243434`, seconds `27.50`, LSTM `0.7372`, delta `-0.1108`
- tick `243306`, seconds `25.50`, LSTM `0.6811`, delta `+0.0273`
- tick `243402`, seconds `27.00`, LSTM `0.8480`, delta `+0.0219`
- tick `242282`, seconds `9.50`, LSTM `0.5337`, delta `+0.0218`
- tick `241994`, seconds `5.00`, LSTM `0.5323`, delta `-0.0132`
- tick `243466`, seconds `28.00`, LSTM `0.7500`, delta `+0.0127`
- tick `242218`, seconds `8.50`, LSTM `0.5237`, delta `-0.0125`

## Top 15 local ridge features

- `lag_14__CT3__shots_fired`: coefficient `0.001212`, |coef| `0.001212`
- `lag_03__T_shots_fired_sum`: coefficient `-0.001074`, |coef| `0.001074`
- `lag_05__T_place_HUT`: coefficient `0.001035`, |coef| `0.001035`
- `lag_14__CT_shots_fired_sum`: coefficient `0.000974`, |coef| `0.000974`
- `lag_00__kill_diff_last_3s`: coefficient `0.000954`, |coef| `0.000954`
- `lag_04__T_place_HUT`: coefficient `0.000940`, |coef| `0.000940`
- `lag_00__T3__is_scoped`: coefficient `0.000921`, |coef| `0.000921`
- `lag_08__T5__flash_duration`: coefficient `-0.000909`, |coef| `0.000909`
- `lag_02__T_burning_players`: coefficient `0.000892`, |coef| `0.000892`
- `lag_02__T_place_HUT`: coefficient `0.000846`, |coef| `0.000846`
- `lag_11__CT3__shots_fired`: coefficient `0.000812`, |coef| `0.000812`
- `lag_00__CT_kills_last_3s`: coefficient `0.000801`, |coef| `0.000801`
- `lag_03__T2__shots_fired`: coefficient `-0.000785`, |coef| `0.000785`
- `lag_13__T_place_HUT`: coefficient `0.000778`, |coef| `0.000778`
- `lag_01__T_place_HUT`: coefficient `0.000771`, |coef| `0.000771`

## Top 10 utility ridge features

- `lag_08__T5__flash_duration`: coefficient `-0.000909` (lowers CT win probability)
- `lag_14__T5__flash_duration`: coefficient `0.000630` (raises CT win probability)
- `lag_06__T5__flash_duration`: coefficient `0.000622` (raises CT win probability)
- `lag_02__T5__flash_duration`: coefficient `0.000609` (raises CT win probability)
- `lag_03__T5__flash_duration`: coefficient `0.000542` (raises CT win probability)
- `lag_00__T5__flash_duration`: coefficient `-0.000516` (lowers CT win probability)
- `lag_02__CT4__flash_duration`: coefficient `0.000460` (raises CT win probability)
- `lag_07__CT4__flash_duration`: coefficient `-0.000415` (lowers CT win probability)
- `lag_08__T_flash_duration_sum`: coefficient `-0.000385` (lowers CT win probability)
- `lag_00__T_molly_inv`: coefficient `-0.000378` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_14__CT3__shots_fired`: coefficient `0.001212` (raises CT win probability)
- `lag_03__T_shots_fired_sum`: coefficient `-0.001074` (lowers CT win probability)
- `lag_05__T_place_HUT`: coefficient `0.001035` (raises CT win probability)
- `lag_14__CT_shots_fired_sum`: coefficient `0.000974` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000954` (raises CT win probability)
- `lag_04__T_place_HUT`: coefficient `0.000940` (raises CT win probability)
- `lag_00__T3__is_scoped`: coefficient `0.000921` (raises CT win probability)
- `lag_02__T_burning_players`: coefficient `0.000892` (raises CT win probability)
- `lag_02__T_place_HUT`: coefficient `0.000846` (raises CT win probability)
- `lag_11__CT3__shots_fired`: coefficient `0.000812` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `243626`, seconds `30.50`, LSTM delta `+0.1931`

Top all feature movements:
- `lag_03__T_shots_fired_sum`: contribution `+0.015296`
- `lag_03__T2__shots_fired`: contribution `+0.008772`
- `lag_02__T_place_HUT`: contribution `-0.007886`
- `lag_08__T5__flash_duration`: contribution `+0.007401`
- `lag_13__T_place_HUT`: contribution `+0.007250`

Top utility-only movements:
- `lag_08__T5__flash_duration`: contribution `+0.007401`
- `lag_14__T5__flash_duration`: contribution `+0.005126`

### tick `243370`, seconds `26.50`, LSTM delta `+0.1396`

Top all feature movements:
- `lag_05__T_place_HUT`: contribution `+0.009643`
- `lag_04__T_place_HUT`: contribution `+0.008766`
- `lag_12__CT_shots_fired_sum`: contribution `+0.006475`
- `lag_06__T5__flash_duration`: contribution `+0.005062`
- `lag_02__T_burning_players`: contribution `+0.004521`

Top utility-only movements:
- `lag_06__T5__flash_duration`: contribution `+0.005062`
- `lag_00__T5__flash_duration`: contribution `+0.004196`
- `lag_05__CT4__flash_duration`: contribution `+0.001925`

### tick `243274`, seconds `25.00`, LSTM delta `+0.1121`

Top all feature movements:
- `lag_09__CT_shots_fired_sum`: contribution `+0.009997`
- `lag_02__T_place_HUT`: contribution `+0.007886`
- `lag_01__T_place_HUT`: contribution `+0.007188`
- `lag_03__T5__flash_duration`: contribution `+0.004413`
- `lag_15__T_place_SQUEAKY`: contribution `+0.003465`

Top utility-only movements:
- `lag_03__T5__flash_duration`: contribution `+0.004413`
- `lag_02__CT4__flash_duration`: contribution `+0.002833`
- `lag_11__CT4__flash_duration`: contribution `+0.001498`

### tick `243434`, seconds `27.50`, LSTM delta `-0.1108`

Top all feature movements:
- `lag_14__CT_shots_fired_sum`: contribution `-0.012862`
- `lag_14__CT3__shots_fired`: contribution `-0.011840`
- `lag_02__T_place_HUT`: contribution `-0.007886`
- `lag_08__T5__flash_duration`: contribution `-0.007401`
- `lag_00__CT_shots_fired_sum`: contribution `-0.005064`

Top utility-only movements:
- `lag_08__T5__flash_duration`: contribution `-0.007401`
- `lag_02__T5__flash_duration`: contribution `-0.004961`
- `lag_07__CT4__flash_duration`: contribution `-0.002551`

### tick `243306`, seconds `25.50`, LSTM delta `+0.0273`

Top all feature movements:
- `lag_02__T_place_HUT`: contribution `+0.007886`
- `lag_00__CT_shots_fired_sum`: contribution `-0.003039`
- `lag_10__CT3__shots_fired`: contribution `-0.002722`
- `lag_04__T5__flash_duration`: contribution `+0.002257`
- `lag_11__CT3__shots_fired`: contribution `+0.002088`

Top utility-only movements:
- `lag_04__T5__flash_duration`: contribution `+0.002257`
- `lag_03__CT4__flash_duration`: contribution `+0.001191`
