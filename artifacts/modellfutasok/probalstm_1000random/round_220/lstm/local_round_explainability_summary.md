# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-mouz-vs-falcons-bo3-xBECUqZMcQ8GCwi-GUyz8e/mouz-vs-falcons-m1-train.csv`
- round_num: `14`

## Largest probability jumps

- tick `146518`, seconds `67.50`, LSTM `0.8008`, delta `-0.0408`
- tick `142550`, seconds `5.50`, LSTM `0.8708`, delta `-0.0399`
- tick `144342`, seconds `33.50`, LSTM `0.8610`, delta `+0.0389`
- tick `144534`, seconds `36.50`, LSTM `0.8204`, delta `-0.0341`
- tick `146806`, seconds `72.00`, LSTM `0.8865`, delta `+0.0327`
- tick `144310`, seconds `33.00`, LSTM `0.8221`, delta `-0.0314`
- tick `145846`, seconds `57.00`, LSTM `0.8296`, delta `+0.0314`
- tick `146998`, seconds `75.00`, LSTM `0.9491`, delta `+0.0312`
- tick `146582`, seconds `68.50`, LSTM `0.8232`, delta `+0.0272`
- tick `144630`, seconds `38.00`, LSTM `0.7905`, delta `-0.0252`

## Top 15 local ridge features

- `lag_00__CT_walking_count`: coefficient `-0.001519`, |coef| `0.001519`
- `lag_00__CT3__is_walking`: coefficient `-0.001407`, |coef| `0.001407`
- `lag_00__CT5__is_walking`: coefficient `-0.001026`, |coef| `0.001026`
- `lag_00__T_walking_count`: coefficient `-0.000962`, |coef| `0.000962`
- `lag_00__T4__is_walking`: coefficient `-0.000834`, |coef| `0.000834`
- `lag_00__CT_damage_last_5s`: coefficient `0.000830`, |coef| `0.000830`
- `lag_00__T_place_BACKOFB`: coefficient `-0.000812`, |coef| `0.000812`
- `lag_01__T_place_LONGDOG`: coefficient `-0.000810`, |coef| `0.000810`
- `lag_13__CT5__is_walking`: coefficient `-0.000777`, |coef| `0.000777`
- `lag_00__T2__is_walking`: coefficient `-0.000777`, |coef| `0.000777`
- `lag_00__damage_diff_last_5s`: coefficient `0.000775`, |coef| `0.000775`
- `lag_00__T_place_LONGDOG`: coefficient `-0.000761`, |coef| `0.000761`
- `lag_05__T1__is_walking`: coefficient `-0.000729`, |coef| `0.000729`
- `lag_00__CT2__is_walking`: coefficient `-0.000710`, |coef| `0.000710`
- `lag_03__T1__is_walking`: coefficient `-0.000706`, |coef| `0.000706`

## Top 10 utility ridge features

- `lag_11__CT_A_site_active_infernos`: coefficient `0.000399` (raises CT win probability)
- `lag_10__CT_A_site_active_infernos`: coefficient `0.000399` (raises CT win probability)
- `lag_12__CT_A_site_active_infernos`: coefficient `0.000374` (raises CT win probability)
- `lag_13__CT_A_site_active_infernos`: coefficient `0.000333` (raises CT win probability)
- `lag_09__CT_A_site_active_infernos`: coefficient `0.000298` (raises CT win probability)
- `lag_11__CT_active_infernos`: coefficient `0.000264` (raises CT win probability)
- `lag_10__CT_active_infernos`: coefficient `0.000259` (raises CT win probability)
- `lag_15__CT_A_site_active_smokes`: coefficient `-0.000258` (lowers CT win probability)
- `lag_12__CT_active_infernos`: coefficient `0.000243` (raises CT win probability)
- `lag_14__CT_A_site_active_infernos`: coefficient `0.000237` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_walking_count`: coefficient `-0.001519` (lowers CT win probability)
- `lag_00__CT3__is_walking`: coefficient `-0.001407` (lowers CT win probability)
- `lag_00__CT5__is_walking`: coefficient `-0.001026` (lowers CT win probability)
- `lag_00__T_walking_count`: coefficient `-0.000962` (lowers CT win probability)
- `lag_00__T4__is_walking`: coefficient `-0.000834` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.000830` (raises CT win probability)
- `lag_00__T_place_BACKOFB`: coefficient `-0.000812` (lowers CT win probability)
- `lag_01__T_place_LONGDOG`: coefficient `-0.000810` (lowers CT win probability)
- `lag_13__CT5__is_walking`: coefficient `-0.000777` (lowers CT win probability)
- `lag_00__T2__is_walking`: coefficient `-0.000777` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `146518`, seconds `67.50`, LSTM delta `-0.0408`

Top all feature movements:
- `lag_00__CT_walking_count`: contribution `-0.004090`
- `lag_00__CT3__is_walking`: contribution `-0.003358`
- `lag_08__CT2__duck_amount`: contribution `-0.002149`
- `lag_05__CT1__duck_amount`: contribution `-0.002086`
- `lag_00__T4__is_walking`: contribution `-0.001926`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `142550`, seconds `5.50`, LSTM delta `-0.0399`

Top all feature movements:
- `lag_06__CT_place_ENTRANCE`: contribution `-0.008245`
- `lag_00__T_place_TSTAIRS`: contribution `-0.007731`
- `lag_04__CT_place_ENTRANCE`: contribution `-0.005104`
- `lag_07__CT_place_ENTRANCE`: contribution `-0.002022`
- `lag_10__T_velocity_mean`: contribution `-0.000782`

Top utility-only movements:
- `lag_11__CT3__molly`: contribution `-0.000215`

### tick `144342`, seconds `33.50`, LSTM delta `+0.0389`

Top all feature movements:
- `lag_00__CT3__is_walking`: contribution `+0.003358`
- `lag_09__CT_place_LONGDOG`: contribution `+0.003325`
- `lag_00__CT_walking_count`: contribution `+0.002727`
- `lag_00__T2__is_walking`: contribution `+0.001785`
- `lag_00__CT2__is_walking`: contribution `+0.001675`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `144534`, seconds `36.50`, LSTM delta `-0.0341`

Top all feature movements:
- `lag_00__T_place_LONGDOG`: contribution `-0.007084`
- `lag_02__T_place_LONGDOG`: contribution `-0.005722`
- `lag_00__T_place_BACKOFB`: contribution `+0.004359`
- `lag_01__T_place_LONGDOG`: contribution `-0.003770`
- `lag_00__CT5__is_walking`: contribution `-0.002460`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `146806`, seconds `72.00`, LSTM delta `+0.0327`

Top all feature movements:
- `lag_05__CT1__duck_amount`: contribution `+0.002086`
- `lag_00__T4__is_walking`: contribution `-0.001926`
- `lag_05__T1__is_walking`: contribution `+0.001662`
- `lag_00__CT_shots_fired_sum`: contribution `+0.001592`
- `lag_14__CT1__duck_amount`: contribution `+0.001351`

Top utility-only movements:
- No utility movement among the top local contributors.
