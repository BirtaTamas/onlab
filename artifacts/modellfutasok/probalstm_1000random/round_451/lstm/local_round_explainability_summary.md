# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-wildcard-vs-spirit-bo3-VLdaQLy-otUvCLBOl-LFGy/wildcard-vs-spirit-m2-dust2.csv`
- round_num: `9`

## Largest probability jumps

- tick `80249`, seconds `119.00`, LSTM `0.5485`, delta `+0.3406`
- tick `76601`, seconds `62.00`, LSTM `0.3570`, delta `+0.2162`
- tick `74329`, seconds `26.50`, LSTM `0.2746`, delta `+0.1824`
- tick `77017`, seconds `68.50`, LSTM `0.3148`, delta `-0.1590`
- tick `73497`, seconds `13.50`, LSTM `0.2684`, delta `-0.1535`
- tick `73593`, seconds `15.00`, LSTM `0.0463`, delta `-0.1511`
- tick `77273`, seconds `72.50`, LSTM `0.2533`, delta `-0.0965`
- tick `77689`, seconds `79.00`, LSTM `0.3023`, delta `+0.0875`
- tick `76665`, seconds `63.00`, LSTM `0.4635`, delta `+0.0871`
- tick `74393`, seconds `27.50`, LSTM `0.3820`, delta `+0.0842`

## Top 15 local ridge features

- `lag_00__T_place_SIDE`: coefficient `-0.004696`, |coef| `0.004696`
- `lag_13__T_place_SIDE`: coefficient `0.004454`, |coef| `0.004454`
- `lag_00__kill_diff_last_3s`: coefficient `0.003930`, |coef| `0.003930`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.003229`, |coef| `0.003229`
- `lag_00__T_place_ARAMP`: coefficient `-0.002948`, |coef| `0.002948`
- `lag_00__CT_kills_last_3s`: coefficient `0.002912`, |coef| `0.002912`
- `lag_00__T_place_PIT`: coefficient `-0.002871`, |coef| `0.002871`
- `lag_01__T_place_LONGA`: coefficient `-0.002441`, |coef| `0.002441`
- `lag_15__CT5__is_walking`: coefficient `0.002388`, |coef| `0.002388`
- `lag_00__damage_diff_last_5s`: coefficient `0.002289`, |coef| `0.002289`
- `lag_03__CT_place_LOWERTUNNEL`: coefficient `0.002117`, |coef| `0.002117`
- `lag_00__CT_damage_last_5s`: coefficient `0.002067`, |coef| `0.002067`
- `lag_04__T_place_LONGA`: coefficient `-0.002029`, |coef| `0.002029`
- `lag_00__T_kills_last_3s`: coefficient `-0.001977`, |coef| `0.001977`
- `lag_05__T_place_LONGA`: coefficient `-0.001922`, |coef| `0.001922`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.003229` (lowers CT win probability)
- `lag_15__CT_A_site_active_infernos`: coefficient `-0.001624` (lowers CT win probability)
- `lag_15__CT_active_infernos`: coefficient `-0.001511` (lowers CT win probability)
- `lag_11__T_flash_alpha_mean`: coefficient `-0.001405` (lowers CT win probability)
- `lag_05__T_flash_alpha_mean`: coefficient `-0.001347` (lowers CT win probability)
- `lag_11__CT_utility_damage_last_5s`: coefficient `-0.000932` (lowers CT win probability)
- `lag_03__CT_A_site_active_infernos`: coefficient `-0.000878` (lowers CT win probability)
- `lag_01__T1__smoke`: coefficient `-0.000869` (lowers CT win probability)
- `lag_00__T2__flash`: coefficient `-0.000844` (lowers CT win probability)
- `lag_09__CT_he_last_5s`: coefficient `0.000841` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_SIDE`: coefficient `-0.004696` (lowers CT win probability)
- `lag_13__T_place_SIDE`: coefficient `0.004454` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003930` (raises CT win probability)
- `lag_00__T_place_ARAMP`: coefficient `-0.002948` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002912` (raises CT win probability)
- `lag_00__T_place_PIT`: coefficient `-0.002871` (lowers CT win probability)
- `lag_01__T_place_LONGA`: coefficient `-0.002441` (lowers CT win probability)
- `lag_15__CT5__is_walking`: coefficient `0.002388` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002289` (raises CT win probability)
- `lag_03__CT_place_LOWERTUNNEL`: coefficient `0.002117` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `80249`, seconds `119.00`, LSTM delta `+0.3406`

Top all feature movements:
- `lag_00__T_place_SIDE`: contribution `+0.090857`
- `lag_13__T_place_SIDE`: contribution `+0.086179`
- `lag_00__T_flash_alpha_mean`: contribution `+0.019589`
- `lag_13__T_place_PIT`: contribution `+0.010583`
- `lag_14__CT_place_EXTENDEDA`: contribution `+0.010198`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.019589`
- `lag_15__CT_A_site_active_infernos`: contribution `+0.005732`
- `lag_15__CT_active_infernos`: contribution `+0.003482`

### tick `76601`, seconds `62.00`, LSTM delta `+0.2162`

Top all feature movements:
- `lag_03__CT_place_LOWERTUNNEL`: contribution `+0.015558`
- `lag_01__T_place_TUNNELSTAIRS`: contribution `+0.012653`
- `lag_00__T_place_TUNNELSTAIRS`: contribution `+0.012348`
- `lag_07__CT_place_LOWERTUNNEL`: contribution `+0.011676`
- `lag_07__T_place_PIT`: contribution `+0.010327`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `74329`, seconds `26.50`, LSTM delta `+0.1824`

Top all feature movements:
- `lag_01__T_place_LONGA`: contribution `+0.010402`
- `lag_00__kill_diff_last_3s`: contribution `+0.009460`
- `lag_00__CT_kills_last_3s`: contribution `+0.008408`
- `lag_00__CT3__is_scoped`: contribution `+0.007739`
- `lag_02__CT1__duck_amount`: contribution `+0.005987`

Top utility-only movements:
- `lag_00__T2__utility_total`: contribution `+0.002695`

### tick `77017`, seconds `68.50`, LSTM delta `-0.1590`

Top all feature movements:
- `lag_04__CT_place_LOWERTUNNEL`: contribution `-0.014031`
- `lag_00__kill_diff_last_3s`: contribution `-0.009460`
- `lag_13__T_place_TUNNELSTAIRS`: contribution `-0.008004`
- `lag_14__T_place_TUNNELSTAIRS`: contribution `-0.007300`
- `lag_00__T_kills_last_3s`: contribution `-0.006265`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `73497`, seconds `13.50`, LSTM delta `-0.1535`

Top all feature movements:
- `lag_01__T_place_LONGA`: contribution `-0.010402`
- `lag_00__kill_diff_last_3s`: contribution `-0.009460`
- `lag_00__T_kills_last_3s`: contribution `-0.006265`
- `lag_06__CT_shots_fired_sum`: contribution `-0.005446`
- `lag_04__T_place_PIT`: contribution `-0.004971`

Top utility-only movements:
- `lag_12__CT4__flash_duration`: contribution `-0.004219`
- `lag_00__CT4__flash_duration`: contribution `-0.003681`
