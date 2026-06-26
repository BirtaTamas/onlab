# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2_finals/blast-bounty-2025-season-2-finals-vitality-vs-the-mongolz-bo3-JVS9HKMAkaZTRHkoiRSMP6/vitality-vs-the-mongolz-m1-mirage.csv`
- round_num: `4`

## Largest probability jumps

- tick `21148`, seconds `54.50`, LSTM `0.4263`, delta `+0.3194`
- tick `20956`, seconds `51.50`, LSTM `0.2175`, delta `-0.3183`
- tick `20828`, seconds `49.50`, LSTM `0.5425`, delta `-0.2267`
- tick `21820`, seconds `65.00`, LSTM `0.3980`, delta `-0.2056`
- tick `21852`, seconds `65.50`, LSTM `0.5857`, delta `+0.1877`
- tick `21212`, seconds `55.50`, LSTM `0.5953`, delta `+0.1864`
- tick `20316`, seconds `41.50`, LSTM `0.7106`, delta `+0.1311`
- tick `18140`, seconds `7.50`, LSTM `0.6184`, delta `-0.1243`
- tick `22172`, seconds `70.50`, LSTM `0.5036`, delta `-0.0610`
- tick `20636`, seconds `46.50`, LSTM `0.7272`, delta `-0.0585`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005728`, |coef| `0.005728`
- `lag_05__T_place_STAIRS`: coefficient `-0.005430`, |coef| `0.005430`
- `lag_00__T_kills_last_3s`: coefficient `-0.004284`, |coef| `0.004284`
- `lag_00__damage_diff_last_5s`: coefficient `0.003974`, |coef| `0.003974`
- `lag_03__T_place_STAIRS`: coefficient `-0.003160`, |coef| `0.003160`
- `lag_00__CT_kills_last_3s`: coefficient `0.002967`, |coef| `0.002967`
- `lag_08__CT4__is_walking`: coefficient `-0.002954`, |coef| `0.002954`
- `lag_04__T2__duck_amount`: coefficient `-0.002888`, |coef| `0.002888`
- `lag_00__T_damage_last_5s`: coefficient `-0.002614`, |coef| `0.002614`
- `lag_04__T_duck_amount_mean`: coefficient `-0.002595`, |coef| `0.002595`
- `lag_11__T_place_STAIRS`: coefficient `0.002400`, |coef| `0.002400`
- `lag_01__CT_shots_fired_sum`: coefficient `-0.002132`, |coef| `0.002132`
- `lag_00__CT_velocity_mean`: coefficient `-0.002093`, |coef| `0.002093`
- `lag_03__T2__duck_amount`: coefficient `0.002063`, |coef| `0.002063`
- `lag_06__T2__is_walking`: coefficient `-0.002038`, |coef| `0.002038`

## Top 10 utility ridge features

- `lag_01__CT_A_site_active_infernos`: coefficient `0.001982` (raises CT win probability)
- `lag_04__CT3__flash_duration`: coefficient `0.001793` (raises CT win probability)
- `lag_11__CT1__flash_duration`: coefficient `-0.001770` (lowers CT win probability)
- `lag_00__CT4__smoke`: coefficient `0.001722` (raises CT win probability)
- `lag_03__CT5__molly`: coefficient `0.001668` (raises CT win probability)
- `lag_10__CT3__flash_duration`: coefficient `-0.001499` (lowers CT win probability)
- `lag_04__CT5__molly`: coefficient `-0.001382` (lowers CT win probability)
- `lag_00__CT_A_site_active_infernos`: coefficient `-0.001332` (lowers CT win probability)
- `lag_02__CT3__flash_duration`: coefficient `-0.001316` (lowers CT win probability)
- `lag_12__CT1__flash_duration`: coefficient `-0.001302` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005728` (raises CT win probability)
- `lag_05__T_place_STAIRS`: coefficient `-0.005430` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.004284` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003974` (raises CT win probability)
- `lag_03__T_place_STAIRS`: coefficient `-0.003160` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002967` (raises CT win probability)
- `lag_08__CT4__is_walking`: coefficient `-0.002954` (lowers CT win probability)
- `lag_04__T2__duck_amount`: coefficient `-0.002888` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002614` (lowers CT win probability)
- `lag_04__T_duck_amount_mean`: coefficient `-0.002595` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `21148`, seconds `54.50`, LSTM delta `+0.3194`

Top all feature movements:
- `lag_03__T_place_STAIRS`: contribution `+0.060492`
- `lag_11__T_place_STAIRS`: contribution `+0.045955`
- `lag_00__kill_diff_last_3s`: contribution `+0.027575`
- `lag_00__T_kills_last_3s`: contribution `+0.013572`
- `lag_00__CT_kills_last_3s`: contribution `+0.008566`

Top utility-only movements:
- `lag_10__CT3__flash_duration`: contribution `+0.008396`
- `lag_12__CT1__flash_duration`: contribution `+0.004913`
- `lag_12__CT3__flash_duration`: contribution `+0.004496`
- `lag_06__T3__flash_duration`: contribution `+0.003600`

### tick `20956`, seconds `51.50`, LSTM delta `-0.3183`

Top all feature movements:
- `lag_05__T_place_STAIRS`: contribution `-0.103958`
- `lag_00__kill_diff_last_3s`: contribution `-0.013787`
- `lag_11__CT1__flash_duration`: contribution `-0.013601`
- `lag_00__T_kills_last_3s`: contribution `-0.013572`
- `lag_04__CT3__flash_duration`: contribution `-0.010047`

Top utility-only movements:
- `lag_11__CT1__flash_duration`: contribution `-0.013601`
- `lag_04__CT3__flash_duration`: contribution `-0.010047`
- `lag_06__CT3__flash_duration`: contribution `-0.004314`
- `lag_06__T3__flash_duration`: contribution `-0.003600`
- `lag_11__CT_flash_duration_sum`: contribution `-0.003495`

### tick `20828`, seconds `49.50`, LSTM delta `-0.2267`

Top all feature movements:
- `lag_01__T_place_STAIRS`: contribution `-0.036332`
- `lag_00__kill_diff_last_3s`: contribution `-0.013787`
- `lag_00__T_kills_last_3s`: contribution `-0.013572`
- `lag_01__CT_shots_fired_sum`: contribution `-0.008888`
- `lag_00__CT_shots_fired_sum`: contribution `-0.008205`

Top utility-only movements:
- `lag_02__CT3__flash_duration`: contribution `-0.007370`
- `lag_07__CT1__flash_duration`: contribution `-0.007361`
- `lag_00__CT3__flash_duration`: contribution `-0.005860`

### tick `21820`, seconds `65.00`, LSTM delta `-0.2056`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.013787`
- `lag_00__T_kills_last_3s`: contribution `-0.013572`
- `lag_04__T2__duck_amount`: contribution `-0.011043`
- `lag_03__T2__duck_amount`: contribution `-0.007888`
- `lag_04__T_duck_amount_mean`: contribution `-0.007546`

Top utility-only movements:
- `lag_00__CT_A_site_active_infernos`: contribution `-0.004700`
- `lag_03__CT5__molly`: contribution `-0.004137`
- `lag_00__CT4__smoke`: contribution `-0.003758`

### tick `21852`, seconds `65.50`, LSTM delta `+0.1877`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.013787`
- `lag_04__T2__duck_amount`: contribution `+0.011043`
- `lag_00__CT_kills_last_3s`: contribution `+0.008566`
- `lag_04__T_duck_amount_mean`: contribution `+0.007546`
- `lag_08__CT4__is_walking`: contribution `+0.007044`

Top utility-only movements:
- `lag_01__CT_A_site_active_infernos`: contribution `+0.006993`
- `lag_04__CT5__molly`: contribution `+0.003429`
- `lag_01__CT_active_infernos`: contribution `+0.002895`
