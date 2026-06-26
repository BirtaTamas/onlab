# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-lynn-vision-vs-vitality-bo3-ZpOL0o26IrRvvgFRbFxVou/lynn-vision-vs-vitality-m1-dust2.csv`
- round_num: `15`

## Largest probability jumps

- tick `123695`, seconds `81.00`, LSTM `0.1437`, delta `-0.2592`
- tick `124207`, seconds `89.00`, LSTM `0.0428`, delta `-0.2076`
- tick `123823`, seconds `83.00`, LSTM `0.2716`, delta `+0.1642`
- tick `126575`, seconds `126.00`, LSTM `0.1362`, delta `-0.1436`
- tick `126223`, seconds `120.50`, LSTM `0.1405`, delta `+0.1247`
- tick `124175`, seconds `88.50`, LSTM `0.2504`, delta `-0.0926`
- tick `123887`, seconds `84.00`, LSTM `0.3942`, delta `+0.0785`
- tick `124143`, seconds `88.00`, LSTM `0.3430`, delta `-0.0759`
- tick `120111`, seconds `25.00`, LSTM `0.3117`, delta `-0.0710`
- tick `121455`, seconds `46.00`, LSTM `0.3456`, delta `+0.0635`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002400`, |coef| `0.002400`
- `lag_13__T_place_MIDDOORS`: coefficient `-0.002218`, |coef| `0.002218`
- `lag_11__T_place_HOLE`: coefficient `0.002217`, |coef| `0.002217`
- `lag_00__damage_diff_last_5s`: coefficient `0.002079`, |coef| `0.002079`
- `lag_00__T_kills_last_3s`: coefficient `-0.001987`, |coef| `0.001987`
- `lag_00__T_flashed_players`: coefficient `0.001933`, |coef| `0.001933`
- `lag_00__CT5__flash_duration`: coefficient `0.001721`, |coef| `0.001721`
- `lag_01__CT_place_ARAMP`: coefficient `0.001707`, |coef| `0.001707`
- `lag_01__damage_diff_last_5s`: coefficient `0.001612`, |coef| `0.001612`
- `lag_00__T1__duck_amount`: coefficient `-0.001599`, |coef| `0.001599`
- `lag_02__CT5__flash_duration`: coefficient `0.001567`, |coef| `0.001567`
- `lag_05__CT_place_ARAMP`: coefficient `0.001525`, |coef| `0.001525`
- `lag_00__CT3__is_walking`: coefficient `-0.001474`, |coef| `0.001474`
- `lag_14__T_place_TUNNELSTAIRS`: coefficient `0.001458`, |coef| `0.001458`
- `lag_05__CT1__flash_duration`: coefficient `0.001402`, |coef| `0.001402`

## Top 10 utility ridge features

- `lag_00__CT5__flash_duration`: coefficient `0.001721` (raises CT win probability)
- `lag_02__CT5__flash_duration`: coefficient `0.001567` (raises CT win probability)
- `lag_05__CT1__flash_duration`: coefficient `0.001402` (raises CT win probability)
- `lag_03__CT4__flash_duration`: coefficient `-0.001335` (lowers CT win probability)
- `lag_00__CT_flash_duration_sum`: coefficient `0.001272` (raises CT win probability)
- `lag_00__CT1__flash_duration`: coefficient `0.001144` (raises CT win probability)
- `lag_00__T1__flash_duration`: coefficient `0.001123` (raises CT win probability)
- `lag_11__CT_flashes_last_5s`: coefficient `-0.001014` (lowers CT win probability)
- `lag_05__T_active_infernos`: coefficient `0.001004` (raises CT win probability)
- `lag_12__CT5__flash_duration`: coefficient `-0.000949` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002400` (raises CT win probability)
- `lag_13__T_place_MIDDOORS`: coefficient `-0.002218` (lowers CT win probability)
- `lag_11__T_place_HOLE`: coefficient `0.002217` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002079` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001987` (lowers CT win probability)
- `lag_00__T_flashed_players`: coefficient `0.001933` (raises CT win probability)
- `lag_01__CT_place_ARAMP`: coefficient `0.001707` (raises CT win probability)
- `lag_01__damage_diff_last_5s`: coefficient `0.001612` (raises CT win probability)
- `lag_00__T1__duck_amount`: coefficient `-0.001599` (lowers CT win probability)
- `lag_05__CT_place_ARAMP`: coefficient `0.001525` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `123695`, seconds `81.00`, LSTM delta `-0.2592`

Top all feature movements:
- `lag_13__T_place_MIDDOORS`: contribution `-0.018850`
- `lag_14__CT_place_HOLE`: contribution `-0.012226`
- `lag_00__T_flashed_players`: contribution `-0.011191`
- `lag_01__CT_place_ARAMP`: contribution `-0.010632`
- `lag_14__T_place_TUNNELSTAIRS`: contribution `-0.010178`

Top utility-only movements:
- `lag_03__CT4__flash_duration`: contribution `-0.007306`
- `lag_00__CT4__flash_duration`: contribution `-0.003932`

### tick `124207`, seconds `89.00`, LSTM delta `-0.2076`

Top all feature movements:
- `lag_05__CT1__flash_duration`: contribution `-0.009456`
- `lag_02__CT5__flash_duration`: contribution `-0.008987`
- `lag_11__CT_shots_fired_sum`: contribution `-0.006572`
- `lag_00__T_kills_last_3s`: contribution `-0.006295`
- `lag_12__CT1__flash_duration`: contribution `-0.005933`

Top utility-only movements:
- `lag_05__CT1__flash_duration`: contribution `-0.009456`
- `lag_02__CT5__flash_duration`: contribution `-0.008987`
- `lag_12__CT1__flash_duration`: contribution `-0.005933`
- `lag_12__CT5__flash_duration`: contribution `-0.005440`
- `lag_12__CT_flash_duration_sum`: contribution `-0.004030`

### tick `123823`, seconds `83.00`, LSTM delta `+0.1642`

Top all feature movements:
- `lag_00__CT5__flash_duration`: contribution `+0.009869`
- `lag_05__CT_place_ARAMP`: contribution `-0.009498`
- `lag_00__CT_place_ARAMP`: contribution `-0.008239`
- `lag_00__T_flashed_players`: contribution `+0.007461`
- `lag_00__CT1__flash_duration`: contribution `+0.007458`

Top utility-only movements:
- `lag_00__CT5__flash_duration`: contribution `+0.009869`
- `lag_00__CT1__flash_duration`: contribution `+0.007458`
- `lag_00__CT_flash_duration_sum`: contribution `+0.007088`
- `lag_00__T1__flash_duration`: contribution `+0.005613`
- `lag_04__CT4__flash_duration`: contribution `+0.004999`

### tick `126575`, seconds `126.00`, LSTM delta `-0.1436`

Top all feature movements:
- `lag_11__T_place_HOLE`: contribution `-0.057161`
- `lag_00__CT_place_UPPERTUNNEL`: contribution `-0.007638`
- `lag_00__T_kills_last_3s`: contribution `-0.006295`
- `lag_00__kill_diff_last_3s`: contribution `-0.005776`
- `lag_02__T_duck_amount_mean`: contribution `-0.005510`

Top utility-only movements:
- `lag_02__T_B_site_active_infernos`: contribution `-0.001532`
- `lag_02__T_active_infernos`: contribution `-0.001439`

### tick `126223`, seconds `120.50`, LSTM delta `+0.1247`

Top all feature movements:
- `lag_00__T_place_HOLE`: contribution `+0.032165`
- `lag_05__T_place_HOLE`: contribution `+0.026114`
- `lag_14__CT_place_OUTSIDETUNNEL`: contribution `+0.013177`
- `lag_14__CT_place_UPPERTUNNEL`: contribution `+0.006330`
- `lag_00__kill_diff_last_3s`: contribution `+0.005776`

Top utility-only movements:
- `lag_05__T_active_infernos`: contribution `+0.002090`
