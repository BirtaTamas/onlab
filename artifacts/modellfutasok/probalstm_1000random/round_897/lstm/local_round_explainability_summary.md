# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-rare-atom-vs-astralis-bo3-2mbRF781jI0kkV-FX6ZCr7/rare-atom-vs-astralis-m1-ancient.csv`
- round_num: `2`

## Largest probability jumps

- tick `23539`, seconds `102.00`, LSTM `0.6289`, delta `-0.2051`
- tick `20851`, seconds `60.00`, LSTM `0.6883`, delta `-0.1913`
- tick `23699`, seconds `104.50`, LSTM `0.7410`, delta `+0.1620`
- tick `21267`, seconds `66.50`, LSTM `0.8772`, delta `+0.1466`
- tick `23795`, seconds `106.00`, LSTM `0.9100`, delta `+0.1134`
- tick `21171`, seconds `65.00`, LSTM `0.7271`, delta `+0.0604`
- tick `22835`, seconds `91.00`, LSTM `0.9074`, delta `-0.0572`
- tick `18067`, seconds `16.50`, LSTM `0.9443`, delta `+0.0560`
- tick `17971`, seconds `15.00`, LSTM `0.8640`, delta `+0.0534`
- tick `24115`, seconds `111.00`, LSTM `0.9103`, delta `-0.0495`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003646`, |coef| `0.003646`
- `lag_00__T_kills_last_3s`: coefficient `-0.002603`, |coef| `0.002603`
- `lag_00__damage_diff_last_5s`: coefficient `0.002513`, |coef| `0.002513`
- `lag_00__T_damage_last_5s`: coefficient `-0.002279`, |coef| `0.002279`
- `lag_00__CT_kills_last_3s`: coefficient `0.002002`, |coef| `0.002002`
- `lag_07__T_bomb_zone_count`: coefficient `0.001994`, |coef| `0.001994`
- `lag_06__CT_place_HOUSE`: coefficient `0.001959`, |coef| `0.001959`
- `lag_00__CT_defusing_count`: coefficient `0.001932`, |coef| `0.001932`
- `lag_13__T3__duck_amount`: coefficient `-0.001777`, |coef| `0.001777`
- `lag_03__T_place_SIDEHALL`: coefficient `-0.001739`, |coef| `0.001739`
- `lag_12__T_bomb_zone_count`: coefficient `-0.001708`, |coef| `0.001708`
- `lag_07__T3__has_bomb`: coefficient `0.001671`, |coef| `0.001671`
- `lag_08__T_place_SIDEHALL`: coefficient `0.001568`, |coef| `0.001568`
- `lag_09__CT4__duck_amount`: coefficient `-0.001497`, |coef| `0.001497`
- `lag_00__CT_duck_amount_mean`: coefficient `0.001495`, |coef| `0.001495`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.001062` (lowers CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `-0.000565` (lowers CT win probability)
- `lag_06__T_flash_alpha_mean`: coefficient `-0.000516` (lowers CT win probability)
- `lag_05__CT1__smoke`: coefficient `-0.000492` (lowers CT win probability)
- `lag_12__T_flash_alpha_mean`: coefficient `-0.000486` (lowers CT win probability)
- `lag_13__T_flash_alpha_mean`: coefficient `-0.000459` (lowers CT win probability)
- `lag_05__CT2__smoke`: coefficient `-0.000442` (lowers CT win probability)
- `lag_09__T_flash_alpha_mean`: coefficient `-0.000436` (lowers CT win probability)
- `lag_05__CT_flash_alpha_mean`: coefficient `0.000432` (raises CT win probability)
- `lag_03__T_flash_alpha_mean`: coefficient `-0.000396` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003646` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002603` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002513` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002279` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002002` (raises CT win probability)
- `lag_07__T_bomb_zone_count`: coefficient `0.001994` (raises CT win probability)
- `lag_06__CT_place_HOUSE`: coefficient `0.001959` (raises CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.001932` (raises CT win probability)
- `lag_13__T3__duck_amount`: coefficient `-0.001777` (lowers CT win probability)
- `lag_03__T_place_SIDEHALL`: coefficient `-0.001739` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `23539`, seconds `102.00`, LSTM delta `-0.2051`

Top all feature movements:
- `lag_07__T_bomb_zone_count`: contribution `-0.011610`
- `lag_03__T_place_SIDEHALL`: contribution `-0.011269`
- `lag_00__kill_diff_last_3s`: contribution `-0.008776`
- `lag_00__T_kills_last_3s`: contribution `-0.008247`
- `lag_06__CT_place_HOUSE`: contribution `-0.006922`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `20851`, seconds `60.00`, LSTM delta `-0.1913`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.008776`
- `lag_00__T_kills_last_3s`: contribution `-0.008247`
- `lag_06__CT_place_HOUSE`: contribution `-0.006922`
- `lag_03__T_place_SIDEENTRANCE`: contribution `-0.005899`
- `lag_00__damage_diff_last_5s`: contribution `-0.005669`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `23699`, seconds `104.50`, LSTM delta `+0.1620`

Top all feature movements:
- `lag_08__T_place_SIDEHALL`: contribution `+0.010161`
- `lag_12__T_bomb_zone_count`: contribution `+0.009943`
- `lag_04__T_place_SIDEHALL`: contribution `+0.008919`
- `lag_00__kill_diff_last_3s`: contribution `+0.008776`
- `lag_00__CT_kills_last_3s`: contribution `+0.005779`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `21267`, seconds `66.50`, LSTM delta `+0.1466`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.008776`
- `lag_00__T_place_TUNNEL`: contribution `+0.005797`
- `lag_00__CT_kills_last_3s`: contribution `+0.005779`
- `lag_00__T_place_WATER`: contribution `+0.005620`
- `lag_01__T_place_SIDEENTRANCE`: contribution `+0.005059`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `23795`, seconds `106.00`, LSTM delta `+0.1134`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.008776`
- `lag_15__T_bomb_zone_count`: contribution `+0.007849`
- `lag_00__T_flash_alpha_mean`: contribution `+0.006446`
- `lag_00__CT_kills_last_3s`: contribution `+0.005779`
- `lag_00__CT_shots_fired_sum`: contribution `-0.004871`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.006446`
