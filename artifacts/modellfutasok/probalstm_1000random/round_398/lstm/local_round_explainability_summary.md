# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-lynn-vision-bo3-6KVULP2-Gxo12lI67V9ZfV/chinggis-warriors-vs-lynn-vision-m3-ancient.csv`
- round_num: `11`

## Largest probability jumps

- tick `87419`, seconds `73.00`, LSTM `0.8365`, delta `+0.1491`
- tick `87355`, seconds `72.00`, LSTM `0.6752`, delta `+0.1299`
- tick `87451`, seconds `73.50`, LSTM `0.9332`, delta `+0.0968`
- tick `87483`, seconds `74.00`, LSTM `0.9717`, delta `+0.0385`
- tick `83259`, seconds `8.00`, LSTM `0.6170`, delta `-0.0377`
- tick `85531`, seconds `43.50`, LSTM `0.6164`, delta `+0.0242`
- tick `82811`, seconds `1.00`, LSTM `0.6756`, delta `+0.0238`
- tick `85819`, seconds `48.00`, LSTM `0.6064`, delta `-0.0238`
- tick `83323`, seconds `9.00`, LSTM `0.5935`, delta `-0.0209`
- tick `86075`, seconds `52.00`, LSTM `0.5825`, delta `+0.0200`

## Top 15 local ridge features

- `lag_14__T_place_HOUSE`: coefficient `0.002626`, |coef| `0.002626`
- `lag_00__CT_kills_last_3s`: coefficient `0.001730`, |coef| `0.001730`
- `lag_12__T_place_HOUSE`: coefficient `0.001658`, |coef| `0.001658`
- `lag_00__kill_diff_last_3s`: coefficient `0.001442`, |coef| `0.001442`
- `lag_00__T_place_HOUSE`: coefficient `-0.001406`, |coef| `0.001406`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001401`, |coef| `0.001401`
- `lag_00__CT_damage_last_5s`: coefficient `0.001348`, |coef| `0.001348`
- `lag_14__T_place_TOPOFMID`: coefficient `-0.001250`, |coef| `0.001250`
- `lag_00__damage_diff_last_5s`: coefficient `0.001151`, |coef| `0.001151`
- `lag_01__T_place_HOUSE`: coefficient `-0.001088`, |coef| `0.001088`
- `lag_00__T2__shots_fired`: coefficient `0.001088`, |coef| `0.001088`
- `lag_15__T_place_HOUSE`: coefficient `0.001058`, |coef| `0.001058`
- `lag_00__T_place_CTSPAWN`: coefficient `-0.001049`, |coef| `0.001049`
- `lag_01__CT_damage_last_5s`: coefficient `0.000907`, |coef| `0.000907`
- `lag_11__T2__duck_amount`: coefficient `-0.000862`, |coef| `0.000862`

## Top 10 utility ridge features

- `lag_11__CT_A_site_active_infernos`: coefficient `0.000783` (raises CT win probability)
- `lag_13__CT_A_site_active_infernos`: coefficient `0.000762` (raises CT win probability)
- `lag_02__CT_A_site_active_infernos`: coefficient `-0.000752` (lowers CT win probability)
- `lag_11__CT3__smoke`: coefficient `-0.000640` (lowers CT win probability)
- `lag_00__CT_A_site_active_infernos`: coefficient `-0.000635` (lowers CT win probability)
- `lag_14__CT2__molly`: coefficient `-0.000622` (lowers CT win probability)
- `lag_13__CT3__smoke`: coefficient `-0.000620` (lowers CT win probability)
- `lag_00__T3__flash`: coefficient `-0.000529` (lowers CT win probability)
- `lag_00__T2__flash`: coefficient `-0.000515` (lowers CT win probability)
- `lag_13__CT_B_site_active_infernos`: coefficient `-0.000499` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_14__T_place_HOUSE`: coefficient `0.002626` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001730` (raises CT win probability)
- `lag_12__T_place_HOUSE`: coefficient `0.001658` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001442` (raises CT win probability)
- `lag_00__T_place_HOUSE`: coefficient `-0.001406` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001401` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001348` (raises CT win probability)
- `lag_14__T_place_TOPOFMID`: coefficient `-0.001250` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001151` (raises CT win probability)
- `lag_01__T_place_HOUSE`: coefficient `-0.001088` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `87419`, seconds `73.00`, LSTM delta `+0.1491`

Top all feature movements:
- `lag_14__T_place_HOUSE`: contribution `+0.023098`
- `lag_00__T_place_HOUSE`: contribution `+0.006182`
- `lag_14__T_place_TOPOFMID`: contribution `+0.005091`
- `lag_00__CT_kills_last_3s`: contribution `+0.004995`
- `lag_00__CT_shots_fired_sum`: contribution `+0.004867`

Top utility-only movements:
- `lag_13__CT_A_site_active_infernos`: contribution `+0.002690`
- `lag_02__CT_A_site_active_infernos`: contribution `+0.002655`

### tick `87355`, seconds `72.00`, LSTM delta `+0.1299`

Top all feature movements:
- `lag_12__T_place_HOUSE`: contribution `+0.014585`
- `lag_14__T_place_HOUSE`: contribution `+0.011549`
- `lag_00__CT_shots_fired_sum`: contribution `+0.006814`
- `lag_00__T_place_CTSPAWN`: contribution `+0.005003`
- `lag_00__CT_kills_last_3s`: contribution `+0.004995`

Top utility-only movements:
- `lag_11__CT_A_site_active_infernos`: contribution `+0.002765`
- `lag_00__CT_A_site_active_infernos`: contribution `+0.002242`

### tick `87451`, seconds `73.50`, LSTM delta `+0.0968`

Top all feature movements:
- `lag_15__T_place_HOUSE`: contribution `+0.009300`
- `lag_00__T_place_HOUSE`: contribution `+0.006182`
- `lag_00__CT_kills_last_3s`: contribution `+0.004995`
- `lag_00__CT_shots_fired_sum`: contribution `+0.004867`
- `lag_01__T_place_HOUSE`: contribution `+0.004785`

Top utility-only movements:
- `lag_00__T2__flash`: contribution `+0.001517`

### tick `87483`, seconds `74.00`, LSTM delta `+0.0385`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.010707`
- `lag_00__T_place_HOUSE`: contribution `+0.006182`
- `lag_00__CT_kills_last_3s`: contribution `+0.004995`
- `lag_01__T_place_HOUSE`: contribution `+0.004785`
- `lag_00__CT1__shots_fired`: contribution `-0.004516`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `83259`, seconds `8.00`, LSTM delta `-0.0377`

Top all feature movements:
- `lag_11__T_place_TUNNEL`: contribution `-0.004849`
- `lag_04__T_place_WATER`: contribution `-0.003328`
- `lag_06__T_place_TUNNEL`: contribution `-0.002082`
- `lag_06__T_place_WATER`: contribution `-0.001947`
- `lag_05__CT_place_HOUSE`: contribution `-0.001799`

Top utility-only movements:
- `lag_02__CT_active_infernos`: contribution `-0.000985`
