# Random LSTM Round Suite

- rounds: `10`
- LSTM round wins by MAE: `3`
- XGBoost round wins by MAE: `7`
- LSTM closer ticks total: `781`
- XGBoost closer ticks total: `929`

## Selected Rounds

| idx | rows | round_num | csv |
|---:|---:|---:|---|
| 1 | 193 | 1 | `processed_full\esl_pro_league_season_22\esl-pro-league-season-22-the-mongolz-vs-natus-vincere-bo3-PG4ywdeF4kSxWHc10zCBZ3\the-mongolz-vs-natus-vincere-m1-nuke.csv` |
| 2 | 145 | 3 | `processed_full\esl_pro_league_season_22\esl-pro-league-season-22-the-mongolz-vs-natus-vincere-bo3-PG4ywdeF4kSxWHc10zCBZ3\the-mongolz-vs-natus-vincere-m1-nuke.csv` |
| 3 | 144 | 6 | `processed_full\esl_pro_league_season_21\esl-pro-league-season-21-vitality-vs-the-mongolz-bo3-7VmOOQFfF_Xgx4vOG4cYIY\vitality-vs-the-mongolz-m1-anubis.csv` |
| 4 | 180 | 7 | `processed_full\esl_pro_league_season_22\esl-pro-league-season-22-the-mongolz-vs-natus-vincere-bo3-PG4ywdeF4kSxWHc10zCBZ3\the-mongolz-vs-natus-vincere-m1-nuke.csv` |
| 5 | 250 | 2 | `processed_full\blast_austin_major\blasttv-austin-major-2025-the-mongolz-vs-faze-bo3-HypmoQ2OL2Ts_Mqj1_9ELG\the-mongolz-vs-faze-m2-anubis.csv` |
| 6 | 176 | 12 | `processed_full\esl_pro_league_season_21\esl-pro-league-season-21-vitality-vs-the-mongolz-bo3-7VmOOQFfF_Xgx4vOG4cYIY\vitality-vs-the-mongolz-m1-anubis.csv` |
| 7 | 231 | 8 | `processed_full\blast_austin_major_stage_1\blasttv-austin-major-2025-stage-1-flyquest-vs-fluxo-ancient-YrTVvYzgDXauKEykMAFJPX\flyquest-vs-fluxo-ancient.csv` |
| 8 | 121 | 2 | `processed_full\blast_austin_major_stage_1\blasttv-austin-major-2025-stage-1-flyquest-vs-fluxo-ancient-YrTVvYzgDXauKEykMAFJPX\flyquest-vs-fluxo-ancient.csv` |
| 9 | 144 | 4 | `processed_full\blast_austin_major\blasttv-austin-major-2025-the-mongolz-vs-faze-bo3-HypmoQ2OL2Ts_Mqj1_9ELG\the-mongolz-vs-faze-m2-anubis.csv` |
| 10 | 126 | 3 | `processed_full\esl_pro_league_season_22\esl-pro-league-season-22-vitality-vs-hotu-bo3-g2oB3RySVGugmKq6yJcHo9\vitality-vs-hotu-m2-dust2.csv` |

## Model Comparison

| idx | true_ct_win | rows | winner | lstm_mae | xgb_mae | lstm_logloss | xgb_logloss | lstm_closer | xgb_closer |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 193 | xgboost | 0.213603 | 0.173077 | 0.286851 | 0.234388 | 16 | 177 |
| 2 | 0 | 145 | lstm | 0.162637 | 0.221342 | 0.207698 | 0.291741 | 139 | 6 |
| 3 | 1 | 144 | xgboost | 0.664391 | 0.533656 | 1.375004 | 0.855990 | 0 | 144 |
| 4 | 0 | 180 | lstm | 0.244666 | 0.368705 | 0.318801 | 0.499529 | 178 | 2 |
| 5 | 1 | 250 | xgboost | 0.500604 | 0.431580 | 0.800927 | 0.627174 | 46 | 204 |
| 6 | 0 | 176 | lstm | 0.234598 | 0.297838 | 0.304695 | 0.392724 | 155 | 21 |
| 7 | 1 | 231 | xgboost | 0.404426 | 0.394552 | 0.630482 | 0.588797 | 108 | 123 |
| 8 | 1 | 121 | xgboost | 0.059622 | 0.021819 | 0.061566 | 0.022072 | 0 | 121 |
| 9 | 1 | 144 | xgboost | 0.537320 | 0.524128 | 0.957511 | 0.843193 | 62 | 82 |
| 10 | 0 | 126 | xgboost | 0.144102 | 0.128175 | 0.188423 | 0.156515 | 77 | 49 |

## Utility Cohorts Across Random Rounds

| cohort | rows | rounds | lstm_mean_prob | xgb_mean_prob | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 1710 | 10 | 0.450298 | 0.504462 | 781 | 929 | 0.704094 | 0.715205 |
| active/recent utility | 1710 | 10 | 0.450298 | 0.504462 | 781 | 929 | 0.704094 | 0.715205 |
| strong utility action | 1201 | 10 | 0.431188 | 0.483190 | 596 | 605 | 0.679434 | 0.678601 |
| utility damage | 84 | 6 | 0.530463 | 0.558056 | 42 | 42 | 0.511905 | 0.595238 |
| active smoke/inferno | 1190 | 10 | 0.431606 | 0.483188 | 585 | 605 | 0.676471 | 0.675630 |
| recent utility last 5s | 19 | 2 | 0.343607 | 0.413637 | 17 | 2 | 1.000000 | 1.000000 |
| flash effect present | 1710 | 10 | 0.450298 | 0.504462 | 781 | 929 | 0.704094 | 0.715205 |

## Frequent Top Ridge Features

| feature | utility | top10_count | mean_abs_coef | max_abs_coef |
|---|---:|---:|---:|---:|
| `kill_diff_last_3s` | False | 8 | 0.002706 | 0.005277 |
| `T_kills_last_3s` | False | 6 | 0.002266 | 0.003933 |
| `CT_place_OUTSIDELONG` | False | 5 | 0.002263 | 0.002787 |
| `damage_diff_last_5s` | False | 4 | 0.002330 | 0.003895 |
| `T_place_HELL` | False | 4 | 0.002052 | 0.002553 |
| `T_place_SILO` | False | 4 | 0.001684 | 0.002237 |
| `CT_shots_fired_sum` | False | 3 | 0.002955 | 0.004733 |
| `T_bomb_zone_count` | False | 3 | 0.002673 | 0.003542 |
| `CT_kills_last_3s` | False | 3 | 0.002181 | 0.004124 |
| `CT_place_BRIDGE` | False | 3 | 0.002058 | 0.002161 |
| `CT_place_TSIDELOWER` | False | 3 | 0.000391 | 0.000640 |
| `CT_place_HOLE` | False | 2 | 0.002684 | 0.002740 |
| `CT_place_TUNNEL` | False | 2 | 0.002588 | 0.002684 |
| `CT_place_WALKWAY` | False | 2 | 0.002435 | 0.003064 |
| `CT_place_CTSIDEUPPER` | False | 2 | 0.002405 | 0.003124 |
| `T4__flash_duration` | True | 2 | 0.002401 | 0.003175 |
| `CT5__flash_duration` | True | 2 | 0.002295 | 0.002396 |
| `CT_place_HEAVEN` | False | 2 | 0.002111 | 0.002837 |
| `T1__flash_duration` | True | 2 | 0.002065 | 0.002760 |
| `T_place_HUT` | False | 2 | 0.002059 | 0.002457 |

## Final Conclusion Draft

A random round mintan az XGBoost tobb roundban volt jobb MAE szerint.

A lokalis ridge surrogate-ok celja nem uj prediktiv modell tanitasa, hanem az LSTM roundon beluli valoszinuseg-mozgasanak ertelmezheto kozelitese. A suite riportban ezert kulon erdemes kezelni a prediktiv osszehasonlitast es az explainability eredmenyeket.

A utility cohort tabla azt mutatja, hogy az aktiv smoke/inferno, utility damage es recent utility helyzetekben melyik modell valoszinusege volt kozelebb a valos roundkimenethez. Ez lokalis parja a korabbi XGBoost utility ablation elemzesnek, de itt roundon beluli tick-szintu viselkedest mer.
