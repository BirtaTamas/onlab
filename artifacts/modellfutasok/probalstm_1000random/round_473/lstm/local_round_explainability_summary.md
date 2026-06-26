# Local Round Explainability

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-tyloo-vs-rare-atom-bo3-8GB1HWZtKOlh9_707n2A62/tyloo-vs-rare-atom-m2-inferno.csv`
- round_num: `16`

## Largest probability jumps

- tick `140438`, seconds `82.50`, LSTM `0.0646`, delta `-0.1889`
- tick `140182`, seconds `78.50`, LSTM `0.3377`, delta `-0.1879`
- tick `140822`, seconds `88.50`, LSTM `0.0148`, delta `-0.0986`
- tick `140790`, seconds `88.00`, LSTM `0.1134`, delta `-0.0700`
- tick `140630`, seconds `85.50`, LSTM `0.0826`, delta `+0.0602`
- tick `135958`, seconds `12.50`, LSTM `0.3958`, delta `+0.0596`
- tick `138070`, seconds `45.50`, LSTM `0.4489`, delta `+0.0577`
- tick `140214`, seconds `79.00`, LSTM `0.2864`, delta `-0.0513`
- tick `135446`, seconds `4.50`, LSTM `0.3359`, delta `+0.0505`
- tick `135478`, seconds `5.00`, LSTM `0.3811`, delta `+0.0453`

## Top 15 local ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.002984`, |coef| `0.002984`
- `lag_02__T_place_ARCH`: coefficient `-0.002880`, |coef| `0.002880`
- `lag_00__kill_diff_last_3s`: coefficient `0.002409`, |coef| `0.002409`
- `lag_00__CT3__is_walking`: coefficient `-0.002248`, |coef| `0.002248`
- `lag_00__damage_diff_last_5s`: coefficient `0.002164`, |coef| `0.002164`
- `lag_00__T_damage_last_5s`: coefficient `-0.002130`, |coef| `0.002130`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001972`, |coef| `0.001972`
- `lag_14__T_place_APARTMENTS`: coefficient `-0.001796`, |coef| `0.001796`
- `lag_14__T_place_BACKALLEY`: coefficient `0.001728`, |coef| `0.001728`
- `lag_00__CT_place_ARCH`: coefficient `0.001645`, |coef| `0.001645`
- `lag_01__T_kills_last_3s`: coefficient `-0.001636`, |coef| `0.001636`
- `lag_00__CT5__alive`: coefficient `0.001532`, |coef| `0.001532`
- `lag_00__CT5__hp`: coefficient `0.001513`, |coef| `0.001513`
- `lag_01__CT_place_TOPOFMID`: coefficient `0.001504`, |coef| `0.001504`
- `lag_00__T_place_BALCONY`: coefficient `-0.001501`, |coef| `0.001501`

## Top 10 utility ridge features

- `lag_15__T4__smoke`: coefficient `0.001267` (raises CT win probability)
- `lag_00__CT3__flash_duration`: coefficient `0.001084` (raises CT win probability)
- `lag_00__CT2__flash_duration`: coefficient `0.000999` (raises CT win probability)
- `lag_14__CT_B_site_active_smokes`: coefficient `-0.000990` (lowers CT win probability)
- `lag_00__CT_flash_duration_sum`: coefficient `0.000986` (raises CT win probability)
- `lag_12__CT3__flash_duration`: coefficient `0.000848` (raises CT win probability)
- `lag_11__T_B_site_active_infernos`: coefficient `-0.000732` (lowers CT win probability)
- `lag_03__T_A_site_active_smokes`: coefficient `-0.000660` (lowers CT win probability)
- `lag_14__T4__smoke`: coefficient `0.000658` (raises CT win probability)
- `lag_10__T_B_site_active_infernos`: coefficient `-0.000644` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.002984` (lowers CT win probability)
- `lag_02__T_place_ARCH`: coefficient `-0.002880` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002409` (raises CT win probability)
- `lag_00__CT3__is_walking`: coefficient `-0.002248` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002164` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002130` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001972` (lowers CT win probability)
- `lag_14__T_place_APARTMENTS`: coefficient `-0.001796` (lowers CT win probability)
- `lag_14__T_place_BACKALLEY`: coefficient `0.001728` (raises CT win probability)
- `lag_00__CT_place_ARCH`: coefficient `0.001645` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `140438`, seconds `82.50`, LSTM delta `-0.1889`

Top all feature movements:
- `lag_02__T_place_ARCH`: contribution `-0.026790`
- `lag_00__T_kills_last_3s`: contribution `-0.009453`
- `lag_00__T_shots_fired_sum`: contribution `-0.008870`
- `lag_00__kill_diff_last_3s`: contribution `-0.005798`
- `lag_00__CT3__is_walking`: contribution `-0.005367`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `140182`, seconds `78.50`, LSTM delta `-0.1879`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.009453`
- `lag_00__T_shots_fired_sum`: contribution `-0.007392`
- `lag_00__CT_place_ARCH`: contribution `-0.006710`
- `lag_00__kill_diff_last_3s`: contribution `-0.005798`
- `lag_15__T4__duck_amount`: contribution `-0.005535`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `140822`, seconds `88.50`, LSTM delta `-0.0986`

Top all feature movements:
- `lag_00__T_place_BALCONY`: contribution `-0.020640`
- `lag_07__T_place_BALCONY`: contribution `-0.016585`
- `lag_06__T_place_BALCONY`: contribution `+0.013866`
- `lag_00__kill_diff_last_3s`: contribution `-0.011595`
- `lag_00__T_kills_last_3s`: contribution `-0.009453`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `140790`, seconds `88.00`, LSTM delta `-0.0700`

Top all feature movements:
- `lag_00__T_place_BALCONY`: contribution `-0.020640`
- `lag_06__T_place_BALCONY`: contribution `-0.013866`
- `lag_13__T_place_ARCH`: contribution `-0.007390`
- `lag_05__T4__duck_amount`: contribution `-0.005246`
- `lag_01__CT_place_PIT`: contribution `+0.004938`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `140630`, seconds `85.50`, LSTM delta `+0.0602`

Top all feature movements:
- `lag_00__T_place_BALCONY`: contribution `+0.020640`
- `lag_00__kill_diff_last_3s`: contribution `+0.011595`
- `lag_00__T_kills_last_3s`: contribution `+0.009453`
- `lag_00__CT_place_ARCH`: contribution `+0.006710`
- `lag_01__T_place_BALCONY`: contribution `-0.005077`

Top utility-only movements:
- No utility movement among the top local contributors.
