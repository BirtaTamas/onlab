# Vegso utility konkluzio

Ez a dokumentum osszefoglalja a utility feature-okkal kapcsolatos vegso eredmenyeket.

## Kiindulo cel

A cel annak vizsgalata volt, hogy a CS2 snapshotokbol tanitott CT win probability modellben a utility feature-ok adnak-e plusz informaciot.

Ket fo modellt hasonlitottunk:

- `no-utility`: minden utility feature eldobva
- `with-utility`: utility feature-ok bent

Kesobb keszult egy harmadik modell is:

- `utility flash nelkul`: utility feature-ok bent, de a flash jellegu feature-ok kidobva

## Vegleges modellek globalis metrikai

| modell | feature count | accuracy | precision | recall | F1 | Brier | logloss | ROC AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no utility | 412 | 0.762628 | 0.733641 | 0.819891 | 0.774372 | 0.151830 | 0.450936 | 0.860441 |
| full utility | 531 | 0.762779 | 0.732923 | 0.822091 | 0.774951 | 0.151855 | 0.451270 | 0.860574 |
| utility flash nelkul | 498 | 0.763613 | 0.733686 | 0.822895 | 0.775734 | 0.151926 | 0.451494 | 0.860587 |

## Globalis metrikak ertelmezese

A full utility modell nem javitott egyertelmuen globalisan:

- accuracy, recall, F1 es ROC AUC picit javult
- Brier es logloss picit romlott

Ez azt jelenti, hogy a utility feature-ok nem adnak eros, egyertelmu globalis probability-javulast.

A flash nelkuli utility modell class metrikakban meg jobb lett, de logloss/Brier tovabb romlott. Ezert ezt sem szabad ugy allitani, hogy minden szempontbol jobb modell.

## Class flip elemzes

A class flip elemzes csak azokat a snapshotokat nezi, ahol a ket modell `0.5` threshold mellett mas class-t prediktal.

| osszehasonlitas | flip sorok | utility jo | no-utility jo | kulonbseg | utility win rate |
|---|---:|---:|---:|---:|---:|
| full utility vs no utility | 13781 | 6951 | 6830 | +121 | 50.44% |
| utility flash nelkul vs no utility | 14855 | 7822 | 7033 | +789 | 52.66% |

Ez azt mutatja, hogy a full utility modellben az atbillenesek majdnem szimmetrikusak voltak. Flash nelkul viszont a megmarado utility feature-ok tisztabb pozitiv jelet adtak.

## Aktiv utility helyzetek

| csoport | full utility win rate | flash nelkuli utility win rate |
|---|---:|---:|
| osszes flip | 50.44% | 52.66% |
| active/recent utility | 50.31% | 52.89% |
| strong utility action | 50.49% | 52.97% |
| utility damage | 51.35% | 51.92% |
| active smoke/inferno | 51.08% | 53.36% |
| recent utility last 5s | 50.20% | 51.95% |
| flash effect jelen van | 45.93% | 49.54% |

## Flash feature-ok konkluzioja

A flash feature-ok zajosabbnak tuntek.

Ez abbol latszott, hogy:

- a full utility modellben a `flash_effect` csoportban a utility win rate csak `45.93%` volt
- flash feature-ok eltavolitasa utan az osszes class flip win rate `50.44%`-rol `52.66%`-ra javult
- active smoke/inferno esetben `51.08%`-rol `53.36%`-ra javult
- active/recent utility esetben `50.31%`-rol `52.89%`-ra javult

A flash hatasa jateklogikailag is nehezebben ertelmezheto onmagaban. Egy flash lehet hasznos, de lehet rossz idozitesu, sajat csapattarsat vakito, vagy olyan kaotikus helyzetben torteno esemeny, amely nem alakul taktikai elonnye.

## Legjobban ertelmezheto utility feature-csoportok

A legerosebb, legjobban magyarazhato utility jelek:

- active smoke/inferno
- utility damage last 5s
- recent utility last 5s
- smoke/molly/he jellegu feature-ok
- utility inventory kulonbsegek, de ezek onmagukban zajosabbak

A flash jellegu feature-ok nem feltetlenul rosszak, de ebben az adathalmazban es modellben zajosabbnak bizonyultak.

## Place es weapon feature-ok

A korabbi ablation alapjan a nyers `place` es `weapon` kategorikus feature-ok inkabb rontottak:

- a train teljesitmeny javult, de a valid/test romlott
- ez overfitre utalt
- ezert a nyers place/weapon feature-ok kihagyasa indokolt

Az aggregalt numerikus pozicios feature-ok viszont maradhatnak.

## Vegso allitas

A utility feature-ok hatasa nem altalanos, nagy globalis javulaskent jelent meg. A teljes utility feature-keszletben a flash jellegu feature-ok zajosnak bizonyultak, es gyengitettek az atbilleneses elemzes tisztasagat.

A flash feature-ok eltavolitasa utan a megmarado utility feature-ok, kulonosen az active smoke/inferno es utility damage jellegu feature-ok, jobban ertelmezheto pozitiv jelet mutattak.

## Dolgozatba javasolt megfogalmazas

A utility feature-ok hasznossaga nem elsosorban a teljes teszthalmazon mert aggregalt metrikakban jelent meg, hanem bizonyos jatekhelyzetekben. A full utility modell globalis metrikakban csak minimalis es vegyes valtozast mutatott a no-utility modellhez kepest. Ugyanakkor az atbilleneses elemzes alapjan a utility feature-ok kepesek voltak megvaltoztatni a modell donteset hatarhelyzetekben.

A flash feature-ok eltavolitasa utan a class flip elemzes tisztabb pozitiv jelet adott: a utility modell az eltero class-predikciok `52.66%`-aban volt helyes, szemben a full utility modell `50.44%`-os aranyaval. Ez arra utal, hogy nem minden utility tipus egyforman hasznos; a flash feature-ok zajosabbak voltak, mig az active smoke/inferno es utility damage feature-ok jobban magyarazhato, kontextusfuggo plusz informaciot hordoztak.

## Kapcsolodo fajlok

- `artifacts/modellfutasok/delta_with_vs_no_utility_100pct_test/delta_summary.md`
- `artifacts/modellfutasok/delta_with_vs_no_utility_100pct_test/class_flip_only_summary.md`
- `artifacts/modellfutasok/delta_with_vs_no_utility_100pct_test/utility_active_flip_analysis/utility_active_flip_summary.md`
- `artifacts/modellfutasok/delta_utility_no_flash_vs_no_utility_100pct_test/flash_ablation_conclusion.md`
- `artifacts/modellfutasok/delta_utility_no_flash_vs_no_utility_100pct_test/utility_active_flip_analysis/utility_active_flip_summary.md`
