# Feature Quality Report

- run: `xgboost_streaming_final_with_utility_100pct`
- split scanned: `train`
- feature count in metrics: `531`
- scanned feature count: `531`
- forbidden hits: `0`
- leak-suspicious hits: `0`
- constant features: `43`
- rare nonzero features: `20`

## Block Counts

- `position`: `249`
- `utility`: `135`
- `state`: `41`
- `player_slot_numeric`: `30`
- `objective`: `28`
- `combat`: `18`
- `other`: `16`
- `economy`: `12`
- `time`: `2`

## Forbidden Hits

- none

## Leak Suspicious Hits

- none

## Constant Features

- `CT1__has_bomb`
- `CT1__he`
- `CT2__has_bomb`
- `CT2__he`
- `CT3__has_bomb`
- `CT3__he`
- `CT4__has_bomb`
- `CT4__he`
- `CT5__has_bomb`
- `CT5__he`
- `CT_A_site_flashes_last_5s`
- `CT_A_site_he_last_5s`
- `CT_A_site_mollies_last_5s`
- `CT_A_site_smokes_last_5s`
- `CT_B_site_flashes_last_5s`
- `CT_B_site_he_last_5s`
- `CT_B_site_mollies_last_5s`
- `CT_B_site_smokes_last_5s`
- `CT_bomb_carrier_alive`
- `CT_bomb_zone_count`
- `CT_he_inv`
- `T1__has_defuser`
- `T1__he`
- `T2__has_defuser`
- `T2__he`
- `T3__has_defuser`
- `T3__he`
- `T4__has_defuser`
- `T4__he`
- `T5__has_defuser`
- `T5__he`
- `T_A_site_flashes_last_5s`
- `T_A_site_he_last_5s`
- `T_A_site_mollies_last_5s`
- `T_A_site_smokes_last_5s`
- `T_B_site_flashes_last_5s`
- `T_B_site_he_last_5s`
- `T_B_site_mollies_last_5s`
- `T_B_site_smokes_last_5s`
- `T_defuser_count`
- `T_defusing_count`
- `T_he_inv`
- `T_place_UNKNOWN`

## Rare Nonzero Features

- `CT_mollies_last_5s` nonzero_ratio=`0.000812` block=`other`
- `CT_place_DECK` nonzero_ratio=`0.000056` block=`position`
- `CT_place_DUMPSTER` nonzero_ratio=`0.000354` block=`position`
- `CT_place_KITCHEN` nonzero_ratio=`0.000360` block=`utility`
- `CT_place_PIPE` nonzero_ratio=`0.000383` block=`position`
- `CT_place_PLAYGROUND` nonzero_ratio=`0.000371` block=`position`
- `CT_place_SIDE` nonzero_ratio=`0.000943` block=`position`
- `CT_place_SILO` nonzero_ratio=`0.000184` block=`position`
- `CT_place_STREET` nonzero_ratio=`0.000898` block=`position`
- `CT_place_UPSTAIRS` nonzero_ratio=`0.000562` block=`position`
- `T_place_BACKOFA` nonzero_ratio=`0.000493` block=`position`
- `T_place_BRICKS` nonzero_ratio=`0.000293` block=`position`
- `T_place_CRANE` nonzero_ratio=`0.000245` block=`position`
- `T_place_CTSIDEUPPER` nonzero_ratio=`0.000429` block=`position`
- `T_place_ENTRANCE` nonzero_ratio=`0.000834` block=`position`
- `T_place_HUTROOF` nonzero_ratio=`0.000903` block=`position`
- `T_place_KITCHEN` nonzero_ratio=`0.000769` block=`utility`
- `T_place_LOCKERROOM` nonzero_ratio=`0.000656` block=`position`
- `T_place_SCAFFOLDING` nonzero_ratio=`0.000831` block=`position`
- `T_place_STORAGEROOM` nonzero_ratio=`0.000290` block=`position`

## Rarest Utility Features

- `CT1__he` nonzero_ratio=`0.000000` unique_count=`1.0`
- `CT2__he` nonzero_ratio=`0.000000` unique_count=`1.0`
- `CT3__he` nonzero_ratio=`0.000000` unique_count=`1.0`
- `CT4__he` nonzero_ratio=`0.000000` unique_count=`1.0`
- `CT5__he` nonzero_ratio=`0.000000` unique_count=`1.0`
- `CT_A_site_flashes_last_5s` nonzero_ratio=`0.000000` unique_count=`1.0`
- `CT_A_site_he_last_5s` nonzero_ratio=`0.000000` unique_count=`1.0`
- `CT_A_site_smokes_last_5s` nonzero_ratio=`0.000000` unique_count=`1.0`
- `CT_B_site_flashes_last_5s` nonzero_ratio=`0.000000` unique_count=`1.0`
- `CT_B_site_he_last_5s` nonzero_ratio=`0.000000` unique_count=`1.0`
- `CT_B_site_smokes_last_5s` nonzero_ratio=`0.000000` unique_count=`1.0`
- `CT_he_inv` nonzero_ratio=`0.000000` unique_count=`1.0`
- `T1__he` nonzero_ratio=`0.000000` unique_count=`1.0`
- `T2__he` nonzero_ratio=`0.000000` unique_count=`1.0`
- `T3__he` nonzero_ratio=`0.000000` unique_count=`1.0`
- `T4__he` nonzero_ratio=`0.000000` unique_count=`1.0`
- `T5__he` nonzero_ratio=`0.000000` unique_count=`1.0`
- `T_A_site_flashes_last_5s` nonzero_ratio=`0.000000` unique_count=`1.0`
- `T_A_site_he_last_5s` nonzero_ratio=`0.000000` unique_count=`1.0`
- `T_A_site_smokes_last_5s` nonzero_ratio=`0.000000` unique_count=`1.0`
- `T_B_site_flashes_last_5s` nonzero_ratio=`0.000000` unique_count=`1.0`
- `T_B_site_he_last_5s` nonzero_ratio=`0.000000` unique_count=`1.0`
- `T_B_site_smokes_last_5s` nonzero_ratio=`0.000000` unique_count=`1.0`
- `T_he_inv` nonzero_ratio=`0.000000` unique_count=`1.0`
- `CT_place_KITCHEN` nonzero_ratio=`0.000360` unique_count=`3.0`
- `T_place_KITCHEN` nonzero_ratio=`0.000769` unique_count=`4.0`
- `T_place_HELL` nonzero_ratio=`0.001488` unique_count=`5.0`
- `CT_smokes_last_5s` nonzero_ratio=`0.002503` unique_count=`5.0`
- `CT_he_last_5s` nonzero_ratio=`0.002685` unique_count=`4.0`
- `T_smokes_last_5s` nonzero_ratio=`0.003072` unique_count=`5.0`
- `T_place_HEAVEN` nonzero_ratio=`0.004723` unique_count=`5.0`
- `T_he_last_5s` nonzero_ratio=`0.005196` unique_count=`4.0`
- `CT_flashes_last_5s` nonzero_ratio=`0.007338` unique_count=`5.0`
- `T_flashes_last_5s` nonzero_ratio=`0.010489` unique_count=`5.0`
- `CT_place_HELL` nonzero_ratio=`0.014822` unique_count=`6.0`
- `CT_place_HEAVEN` nonzero_ratio=`0.027557` unique_count=`5.0`
- `CT4__flash_duration` nonzero_ratio=`0.039799` unique_count=`nan`
- `CT3__flash_duration` nonzero_ratio=`0.040292` unique_count=`nan`
- `CT1__flash_duration` nonzero_ratio=`0.041362` unique_count=`nan`
- `T5__flash_duration` nonzero_ratio=`0.041485` unique_count=`nan`
- `CT2__flash_duration` nonzero_ratio=`0.041537` unique_count=`nan`
- `CT5__flash_duration` nonzero_ratio=`0.042287` unique_count=`nan`
- `T4__flash_duration` nonzero_ratio=`0.042407` unique_count=`nan`
- `T2__flash_duration` nonzero_ratio=`0.043385` unique_count=`nan`
- `T1__flash_duration` nonzero_ratio=`0.044458` unique_count=`nan`
- `T3__flash_duration` nonzero_ratio=`0.044676` unique_count=`nan`
- `T_utility_damage_last_5s` nonzero_ratio=`0.051445` unique_count=`195.0`
- `CT_A_site_active_infernos` nonzero_ratio=`0.053746` unique_count=`6.0`
- `CT_B_site_active_infernos` nonzero_ratio=`0.060292` unique_count=`5.0`
- `CT_utility_damage_last_5s` nonzero_ratio=`0.061220` unique_count=`231.0`
