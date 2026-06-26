# EDA összefoglaló

- Mintában szereplő CSV fájlok száma: 1000
- Mintában szereplő sorok száma: 1176164
- Oszlopok száma: 32
- CT győzelmi arány a mintában: 0.4877

## Legerősebb korrelációk
                feature    corr  abs_corr
             equip_diff  0.5502    0.5502
             armor_diff  0.5332    0.5332
             alive_diff  0.5030    0.5030
                hp_diff  0.4952    0.4952
           CT_armor_sum  0.3538    0.3538
     CT_equip_value_sum  0.3417    0.3417
       utility_inv_diff  0.3380    0.3380
        CT_helmet_count  0.3364    0.3364
      T_equip_value_sum -0.3251    0.3251
         T_helmet_count -0.3189    0.3189
            T_armor_sum -0.3158    0.3158
closest_enemy_dist_diff  0.3075    0.3075
         flash_inv_diff  0.3054    0.3054
            spread_diff  0.2905    0.2905
         smoke_inv_diff  0.2785    0.2785
               CT_alive  0.2773    0.2773
        CT1__has_helmet  0.2714    0.2714
         T3__has_helmet -0.2599    0.2599
        CT4__has_helmet  0.2598    0.2598
CT_cash_spent_round_sum  0.2594    0.2594

## Fázisonkénti átlagok
             alive_diff  hp_diff  equip_diff  utility_inv_diff  damage_diff_last_5s  active_smokes_total  active_infernos_total
round_phase                                                                                                                    
korai            0.0084   2.1876   2230.4674           -1.3318               1.6489               1.0801                 0.5931
középső          0.0658  14.8513   2094.1687           -1.1280               3.0295               2.1125                 0.2806
késői           -0.0300  12.8730    691.4872           -0.5158               2.1243               0.9969                 0.1703

## Site utility átlagok
                            átlag
T_A_site_active_smokes     0.3710
CT_A_site_active_smokes    0.3014
T_B_site_active_smokes     0.3525
CT_B_site_active_smokes    0.3411
T_A_site_active_infernos   0.0882
CT_A_site_active_infernos  0.0561
T_B_site_active_infernos   0.1002
CT_B_site_active_infernos  0.0638
T_A_site_smokes_last_5s    0.0000
CT_A_site_smokes_last_5s   0.0000
T_B_site_smokes_last_5s    0.0000
CT_B_site_smokes_last_5s   0.0000
T_A_site_mollies_last_5s   0.0000
CT_A_site_mollies_last_5s  0.0000
T_B_site_mollies_last_5s   0.0000
CT_B_site_mollies_last_5s  0.0000

## Elkészült EDA ábrák
- eda_target_distribution.png
- eda_diff_histograms.png
- eda_boxplots_by_target.png
- eda_alive_equip_heatmap.png
- eda_top_feature_correlations.png
- eda_round_phase_summary.png
- eda_phase_feature_lines.png
- eda_site_utility_heatmap.png
- eda_alive_diff_curve.png
- eda_equip_diff_curve.png