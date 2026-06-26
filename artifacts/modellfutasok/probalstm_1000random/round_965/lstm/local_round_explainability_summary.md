# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-the-mongolz-vs-natus-vincere-bo3-PG4ywdeF4kSxWHc10zCBZ3/the-mongolz-vs-natus-vincere-m1-nuke.csv`
- round_num: `3`

## Largest probability jumps

- tick `18451`, seconds `30.00`, LSTM `0.1181`, delta `-0.3082`
- tick `18099`, seconds `24.50`, LSTM `0.1159`, delta `-0.2467`
- tick `18387`, seconds `29.00`, LSTM `0.3573`, delta `+0.2386`
- tick `18131`, seconds `25.00`, LSTM `0.2166`, delta `+0.1007`
- tick `18163`, seconds `25.50`, LSTM `0.1322`, delta `-0.0844`
- tick `18035`, seconds `23.50`, LSTM `0.3749`, delta `-0.0808`
- tick `17267`, seconds `11.50`, LSTM `0.2777`, delta `-0.0801`
- tick `18419`, seconds `29.50`, LSTM `0.4263`, delta `+0.0690`
- tick `17363`, seconds `13.00`, LSTM `0.3955`, delta `+0.0499`
- tick `18579`, seconds `32.00`, LSTM `0.0453`, delta `-0.0415`

## Top 15 local ridge features

- `lag_04__T_place_HUT`: coefficient `-0.002457`, |coef| `0.002457`
- `lag_00__T_place_HUT`: coefficient `-0.001660`, |coef| `0.001660`
- `lag_09__CT_place_HEAVEN`: coefficient `-0.001385`, |coef| `0.001385`
- `lag_00__T1__flash_duration`: coefficient `-0.001370`, |coef| `0.001370`
- `lag_13__T_place_SQUEAKY`: coefficient `0.001370`, |coef| `0.001370`
- `lag_08__T_place_ROOF`: coefficient `0.001304`, |coef| `0.001304`
- `lag_14__CT3__flash_duration`: coefficient `0.001271`, |coef| `0.001271`
- `lag_00__kill_diff_last_3s`: coefficient `0.001205`, |coef| `0.001205`
- `lag_05__T_bomb_zone_count`: coefficient `-0.001182`, |coef| `0.001182`
- `lag_00__T_kills_last_3s`: coefficient `-0.001181`, |coef| `0.001181`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001166`, |coef| `0.001166`
- `lag_06__CT_place_RAFTERS`: coefficient `-0.001132`, |coef| `0.001132`
- `lag_07__CT_place_SECRET`: coefficient `0.001113`, |coef| `0.001113`
- `lag_10__T1__shots_fired`: coefficient `0.001036`, |coef| `0.001036`
- `lag_09__CT_place_RAFTERS`: coefficient `0.001031`, |coef| `0.001031`

## Top 10 utility ridge features

- `lag_00__T1__flash_duration`: coefficient `-0.001370` (lowers CT win probability)
- `lag_14__CT3__flash_duration`: coefficient `0.001271` (raises CT win probability)
- `lag_09__CT4__flash_duration`: coefficient `-0.000930` (lowers CT win probability)
- `lag_12__CT3__flash_duration`: coefficient `0.000928` (raises CT win probability)
- `lag_09__T1__flash_duration`: coefficient `-0.000908` (lowers CT win probability)
- `lag_09__T2__flash_duration`: coefficient `-0.000877` (lowers CT win probability)
- `lag_11__T1__flash_duration`: coefficient `-0.000801` (lowers CT win probability)
- `lag_01__T_A_site_active_infernos`: coefficient `-0.000677` (lowers CT win probability)
- `lag_00__CT1__flash`: coefficient `0.000652` (raises CT win probability)
- `lag_00__CT4__flash_duration`: coefficient `-0.000642` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_04__T_place_HUT`: coefficient `-0.002457` (lowers CT win probability)
- `lag_00__T_place_HUT`: coefficient `-0.001660` (lowers CT win probability)
- `lag_09__CT_place_HEAVEN`: coefficient `-0.001385` (lowers CT win probability)
- `lag_13__T_place_SQUEAKY`: coefficient `0.001370` (raises CT win probability)
- `lag_08__T_place_ROOF`: coefficient `0.001304` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001205` (raises CT win probability)
- `lag_05__T_bomb_zone_count`: coefficient `-0.001182` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001181` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001166` (lowers CT win probability)
- `lag_06__CT_place_RAFTERS`: coefficient `-0.001132` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `18451`, seconds `30.00`, LSTM delta `-0.3082`

Top all feature movements:
- `lag_04__T_place_HUT`: contribution `-0.022901`
- `lag_07__CT_place_SECRET`: contribution `-0.011456`
- `lag_00__T_shots_fired_sum`: contribution `-0.009614`
- `lag_09__T_place_HUT`: contribution `-0.009247`
- `lag_13__T_place_SQUEAKY`: contribution `-0.008530`

Top utility-only movements:
- `lag_09__T2__flash_duration`: contribution `-0.006389`
- `lag_09__T1__flash_duration`: contribution `-0.005745`
- `lag_09__CT4__flash_duration`: contribution `-0.004808`

### tick `18099`, seconds `24.50`, LSTM delta `-0.2467`

Top all feature movements:
- `lag_04__T_place_HUT`: contribution `-0.022901`
- `lag_14__CT3__flash_duration`: contribution `-0.008704`
- `lag_00__T_shots_fired_sum`: contribution `-0.006118`
- `lag_05__T_place_HUT`: contribution `-0.005790`
- `lag_04__T_place_SQUEAKY`: contribution `-0.005475`

Top utility-only movements:
- `lag_14__CT3__flash_duration`: contribution `-0.008704`
- `lag_00__T1__flash_duration`: contribution `-0.003910`

### tick `18387`, seconds `29.00`, LSTM delta `+0.2386`

Top all feature movements:
- `lag_00__T_place_HUT`: contribution `+0.015478`
- `lag_13__T_place_SQUEAKY`: contribution `+0.008530`
- `lag_09__CT_place_HEAVEN`: contribution `+0.007479`
- `lag_14__T_place_HUT`: contribution `+0.007378`
- `lag_08__T_place_HUT`: contribution `+0.006094`

Top utility-only movements:
- `lag_07__T2__flash_duration`: contribution `+0.004491`

### tick `18131`, seconds `25.00`, LSTM delta `+0.1007`

Top all feature movements:
- `lag_00__T_place_HUT`: contribution `+0.015478`
- `lag_00__T_shots_fired_sum`: contribution `+0.008740`
- `lag_00__T1__shots_fired`: contribution `+0.005835`
- `lag_05__T_place_HUT`: contribution `-0.005790`
- `lag_00__T1__flash_duration`: contribution `+0.003910`

Top utility-only movements:
- `lag_00__T1__flash_duration`: contribution `+0.003910`

### tick `18163`, seconds `25.50`, LSTM delta `-0.0844`

Top all feature movements:
- `lag_00__T_place_HUT`: contribution `-0.015478`
- `lag_00__T1__flash_duration`: contribution `-0.008665`
- `lag_04__T_place_SQUEAKY`: contribution `+0.005475`
- `lag_01__T_place_HUT`: contribution `+0.005110`
- `lag_02__T_shots_fired_sum`: contribution `-0.004110`

Top utility-only movements:
- `lag_00__T1__flash_duration`: contribution `-0.008665`
- `lag_00__CT4__flash_duration`: contribution `-0.003317`
- `lag_00__T2__flash_duration`: contribution `-0.002416`
