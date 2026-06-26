# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-vitality-vs-gamerlegion-bo3-HfAhqHTEhpe_HlObeToa76/vitality-vs-gamerlegion-m2-dust2.csv`
- round_num: `10`

## Largest probability jumps

- tick `94877`, seconds `95.00`, LSTM `0.1266`, delta `-0.1806`
- tick `91229`, seconds `38.00`, LSTM `0.4171`, delta `+0.0635`
- tick `95517`, seconds `105.00`, LSTM `0.0108`, delta `-0.0557`
- tick `94749`, seconds `93.00`, LSTM `0.3002`, delta `+0.0387`
- tick `89437`, seconds `10.00`, LSTM `0.3611`, delta `+0.0358`
- tick `92413`, seconds `56.50`, LSTM `0.3559`, delta `-0.0357`
- tick `94173`, seconds `84.00`, LSTM `0.3344`, delta `+0.0330`
- tick `93053`, seconds `66.50`, LSTM `0.3591`, delta `+0.0312`
- tick `89373`, seconds `9.00`, LSTM `0.3189`, delta `-0.0298`
- tick `94717`, seconds `92.50`, LSTM `0.2615`, delta `-0.0296`

## Top 15 local ridge features

- `lag_03__CT_place_HOLE`: coefficient `-0.001234`, |coef| `0.001234`
- `lag_05__T_place_EXTENDEDA`: coefficient `-0.001007`, |coef| `0.001007`
- `lag_04__CT_place_ARAMP`: coefficient `0.000969`, |coef| `0.000969`
- `lag_00__T_place_EXTENDEDA`: coefficient `-0.000968`, |coef| `0.000968`
- `lag_03__T_place_EXTENDEDA`: coefficient `-0.000864`, |coef| `0.000864`
- `lag_08__T3__flash_duration`: coefficient `0.000795`, |coef| `0.000795`
- `lag_00__T_flashed_players`: coefficient `0.000786`, |coef| `0.000786`
- `lag_02__T_flashed_players`: coefficient `-0.000775`, |coef| `0.000775`
- `lag_00__CT_place_ARAMP`: coefficient `-0.000766`, |coef| `0.000766`
- `lag_14__T_place_MIDDOORS`: coefficient `-0.000746`, |coef| `0.000746`
- `lag_02__CT5__flash_duration`: coefficient `-0.000729`, |coef| `0.000729`
- `lag_04__CT2__is_walking`: coefficient `0.000716`, |coef| `0.000716`
- `lag_00__CT1__alive`: coefficient `0.000711`, |coef| `0.000711`
- `lag_00__CT1__hp`: coefficient `0.000701`, |coef| `0.000701`
- `lag_01__T_burning_players`: coefficient `-0.000696`, |coef| `0.000696`

## Top 10 utility ridge features

- `lag_08__T3__flash_duration`: coefficient `0.000795` (raises CT win probability)
- `lag_02__CT5__flash_duration`: coefficient `-0.000729` (lowers CT win probability)
- `lag_09__T2__flash_duration`: coefficient `0.000671` (raises CT win probability)
- `lag_09__T1__flash_duration`: coefficient `0.000657` (raises CT win probability)
- `lag_07__T_A_site_active_smokes`: coefficient `-0.000621` (lowers CT win probability)
- `lag_00__T4__flash_duration`: coefficient `0.000618` (raises CT win probability)
- `lag_00__T_flash_duration_sum`: coefficient `0.000610` (raises CT win probability)
- `lag_02__CT_A_site_active_infernos`: coefficient `-0.000581` (lowers CT win probability)
- `lag_00__T2__flash`: coefficient `0.000574` (raises CT win probability)
- `lag_03__T3__flash_duration`: coefficient `0.000566` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_03__CT_place_HOLE`: coefficient `-0.001234` (lowers CT win probability)
- `lag_05__T_place_EXTENDEDA`: coefficient `-0.001007` (lowers CT win probability)
- `lag_04__CT_place_ARAMP`: coefficient `0.000969` (raises CT win probability)
- `lag_00__T_place_EXTENDEDA`: coefficient `-0.000968` (lowers CT win probability)
- `lag_03__T_place_EXTENDEDA`: coefficient `-0.000864` (lowers CT win probability)
- `lag_00__T_flashed_players`: coefficient `0.000786` (raises CT win probability)
- `lag_02__T_flashed_players`: coefficient `-0.000775` (lowers CT win probability)
- `lag_00__CT_place_ARAMP`: coefficient `-0.000766` (lowers CT win probability)
- `lag_14__T_place_MIDDOORS`: coefficient `-0.000746` (lowers CT win probability)
- `lag_04__CT2__is_walking`: coefficient `0.000716` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `94877`, seconds `95.00`, LSTM delta `-0.1806`

Top all feature movements:
- `lag_03__CT_place_HOLE`: contribution `-0.013781`
- `lag_08__T3__flash_duration`: contribution `-0.006182`
- `lag_04__CT_place_ARAMP`: contribution `-0.006039`
- `lag_02__T_flashed_players`: contribution `-0.005983`
- `lag_05__T_place_EXTENDEDA`: contribution `-0.004990`

Top utility-only movements:
- `lag_08__T3__flash_duration`: contribution `-0.006182`
- `lag_09__T2__flash_duration`: contribution `-0.004810`
- `lag_09__T1__flash_duration`: contribution `-0.004729`
- `lag_02__T2__flash_duration`: contribution `-0.003347`
- `lag_02__CT5__flash_duration`: contribution `-0.003060`

### tick `91229`, seconds `38.00`, LSTM delta `+0.0635`

Top all feature movements:
- `lag_00__T_flashed_players`: contribution `+0.006064`
- `lag_00__T_flash_duration_sum`: contribution `+0.005508`
- `lag_00__T4__flash_duration`: contribution `+0.003385`
- `lag_15__T_place_TUNNELSTAIRS`: contribution `+0.003156`
- `lag_00__T2__flash_duration`: contribution `+0.002652`

Top utility-only movements:
- `lag_00__T_flash_duration_sum`: contribution `+0.005508`
- `lag_00__T4__flash_duration`: contribution `+0.003385`
- `lag_00__T2__flash_duration`: contribution `+0.002652`
- `lag_00__T1__flash_duration`: contribution `+0.002048`

### tick `95517`, seconds `105.00`, LSTM delta `-0.0557`

Top all feature movements:
- `lag_04__T_place_EXTENDEDA`: contribution `+0.002199`
- `lag_00__T_kills_last_3s`: contribution `-0.002093`
- `lag_05__T3__flash_duration`: contribution `-0.002089`
- `lag_08__CT_place_LONGDOORS`: contribution `-0.001837`
- `lag_03__CT4__flash_duration`: contribution `-0.001577`

Top utility-only movements:
- `lag_05__T3__flash_duration`: contribution `-0.002089`
- `lag_03__CT4__flash_duration`: contribution `-0.001577`

### tick `94749`, seconds `93.00`, LSTM delta `+0.0387`

Top all feature movements:
- `lag_00__CT_place_ARAMP`: contribution `+0.004772`
- `lag_15__T_place_SHORTSTAIRS`: contribution `+0.001867`
- `lag_04__CT2__is_walking`: contribution `+0.001690`
- `lag_13__T3__duck_amount`: contribution `+0.001648`
- `lag_13__CT2__duck_amount`: contribution `+0.001567`

Top utility-only movements:
- `lag_05__T1__flash_duration`: contribution `+0.001341`
- `lag_05__T2__flash_duration`: contribution `+0.000830`

### tick `89437`, seconds `10.00`, LSTM delta `+0.0358`

Top all feature movements:
- `lag_00__CT_place_BDOORS`: contribution `+0.002444`
- `lag_04__CT4__duck_amount`: contribution `+0.002407`
- `lag_10__T1__is_scoped`: contribution `+0.002198`
- `lag_03__CT_place_EXTENDEDA`: contribution `+0.001452`
- `lag_03__T_place_OUTSIDETUNNEL`: contribution `+0.001446`

Top utility-only movements:
- `lag_01__T4__smoke`: contribution `+0.000587`
- `lag_14__CT1__smoke`: contribution `+0.000529`
