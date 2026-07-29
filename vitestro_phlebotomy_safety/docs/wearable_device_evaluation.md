# Vitestro Automated Phlebotomy: Real-Time Detector Evaluation

**Updated:** 2026-07-28

**Scope:** live presyncope/discomfort sensing during venipuncture

**Selection rule:** documented live sensor-to-host path; target hardware price ≤$400 per device

`✓` continuous live value or derivable live waveform · `△` on-demand, spot-check, optional, or partner/vendor-gated · `✕` no suitable live path confirmed

Prices were checked on 2026-07-28. Tax, shipping, host phone/computer, consumables, and non-public license fees are excluded unless shown.

## Metric key

| Metric | ECG | Raw PPG | HR | IBI / RR | Resp. | EDA | SpO2 | Temp. | Motion | Beat-to-beat BP |
|---|---|---|---|---|---|---|---|---|---|---|
| Meaning | Cardiac electrical waveform | Optical pulse waveform | Heart rate | Interbeat / R-R interval | Respiratory rate | Electrodermal activity / sweating response | Oxygen saturation | Skin/body temperature | Accelerometer/gyroscope | Systolic/mean/diastolic pressure for every beat |

## Budget live-feature matrix

| Device and placement | Vendor base | Public hardware price | ECG | Raw PPG | HR | IBI / RR | Resp. | EDA | SpO2 | Temp. | Motion | Beat-to-beat BP | Live host path |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| [Samsung Galaxy Watch8](https://developer.samsung.com/health/sensor/guide/data-specifications.html) — wrist | South Korea | [from $349.99](https://www.samsung.com/us/watches/galaxy-watch8/) | △ | ✓ | ✓ | ✓ | ✕ | ✓ | △ | ✓ | ✓ | ✕ | △ Partner SDK |
| [Polar H10](https://www.polar.com/en/science/research-tools) — chest | Finland | [$104.95](https://www.polar.com/us-en/sensors/h10) | ✓ | ✕ | ✓ | ✓ | ✕ | ✕ | ✕ | ✕ | ✓ | ✕ | ✓ Open BLE |
| [Polar Verity Sense](https://github.com/polarofficial/polar-ble-sdk/blob/master/documentation/products/PolarVeritySense.md) — upper arm | Finland | [$104.95](https://www.polar.com/us-en/products/accessories/polar-verity-sense/) | ✕ | ✓ | ✓ | ✓ | ✕ | ✕ | ✕ | ✕ | ✓ | ✕ | ✓ Open SDK |
| [BITalino (r)evolution Board Kit](https://www.pluxbiosignals.com/products/bitalino-revolution-board-kit-ble-bt) — adhesive electrodes / body | Portugal | $180 US storefront | ✓ | ✕ | ✓ | ✓ | ✕ | ✓ | ✕ | ✕ | ✓ | ✕ | ✓ Bluetooth + APIs |
| [Mindfield eSense Skin Response](https://mindfield-shop.com/en/product/esense-skin-response/) — two fingers | Germany | €149–159 + [OSC/LSL add-on](https://help.mindfield.de/en/skin-response-manual) (~$11.99 example) | ✕ | ✕ | ✕ | ✕ | ✕ | ✓ | ✕ | ✕ | ✕ | ✕ | ✓ OSC / LSL |
| [Garmin vívosmart 5](https://www.garmin.com/en-US/p/782585/pn/010-02645-04/) — wrist | United States | [$149.99](https://www.garmin.com/en-US/newsroom/press-release/featured/simple-and-streamlined-garmin-vivosmart-5-fitness-tracker-is-the-effortless-way-to-take-charge-of-your-everyday-health-wellness-and-activity/); commercial SDK fee or device MOQ not public | ✕ | ✕ | △ | △ | △ | ✕ | △ | ✕ | △ | ✕ | △ Partner SDK; confirm model |
| [Fitbit Charge 6](https://store.google.com/us/product/fitbit_charge_6?hl=en-US) — wrist | United States | $159.95 | △ | ✕ | ✓ | ✕ | ✕ | △ | ✕ | ✕ | ✕ | ✕ | △ BLE HR only |
| [Apple Watch SE 3](https://www.apple.com/us/shop/buy-watch/apple-watch-se) — wrist | United States | from $249; iPhone excluded | ✕ | ✕ | ✓ | ✕ | ✕ | ✕ | ✕ | ✕ | ✓ | ✕ | ✓ Public APIs |
| [Nonin 3230](https://www.nonin.com/products/3230/) — finger clip | United States | retail [$159](https://discountcardiology.com/products/nonin-3230-bluetooth-pulse-oximeter)–[$275](https://www.turnermedical.com/NONIN_CONNECT_BLUETOOTH_IPHONE_OXIMETER_p/nonin_connect.htm) | ✕ | ✕ | △ | ✕ | ✕ | ✕ | △ | ✕ | ✕ | ✕ | △ OEM BLE; spot-check only |

## Budget-device evidence

| Device | Quality evidence | Decision |
|---|---|---|
| Samsung Watch8 | PPG 25 Hz; HR/IBI and EDA 1 Hz; a Watch6 head-up-tilt study reported AUROC 0.91 for a five-minute presyncope window. [SDK](https://developer.samsung.com/health/sensor/guide/data-specifications.html) · [study](https://academic.oup.com/ehjdh/article/7/4/ztag053/8586837) | Primary watch |
| Polar H10 | Live raw ECG/RR; 2026 validation reported concordance ≥0.99 and HRV error <1%. [interface](https://www.polar.com/en/science/research-tools) · [study](https://pubmed.ncbi.nlm.nih.gov/42275859/) | ECG/RR reference |
| Polar Verity Sense | Open SDK streams PPG at 28–176 Hz, pulse intervals, and motion; a 2026 comparison found Verity and Apple Watch SE had the best HR agreement among four PPG wearables. [SDK](https://github.com/polarofficial/polar-ble-sdk/blob/master/documentation/products/PolarVeritySense.md) · [study](https://pubmed.ncbi.nlm.nih.gov/42438855/) | Raw-PPG candidate |
| BITalino | A peer-reviewed comparison found its EDA and ECG closely matched BIOPAC MP35 recordings. [study](https://doaj.org/article/53e98d1c65ed4f10a5de8e6b435c57ef) | Low-cost lab reference |
| Mindfield eSense | 5 Hz, 18-bit EDA with live OSC/LSL; peer-reviewed use in 63 participants established portable-study feasibility, not equivalence to BIOPAC. [specification](https://help.mindfield.de/en/skin-response-manual) · [study](https://pubmed.ncbi.nlm.nih.gov/28221710/) | Wearable EDA candidate |
| Garmin vívosmart 5 | A 2026 free-living Holter comparison reported systematic HR underestimation; exact real-time fields still require Garmin confirmation. [study](https://pubmed.ncbi.nlm.nih.gov/42099266/) · [SDK](https://developer.garmin.com/health-sdk/overview/) | Evaluate only if SDK approved |
| Fitbit Charge 6 | Standard Bluetooth Heart Rate Profile is documented; no Charge 6 device-specific peer-reviewed validation was identified in this search. [interface](https://support.google.com/googlehealth/answer/14236705?hl=en) | HR-only fallback |
| Apple Watch SE | A 2026 study found strong HR agreement with Polar H10 and better agreement than Galaxy Watch6 in that exercise cohort. [study](https://pubmed.ncbi.nlm.nih.gov/42438855/) | Consumer baseline |
| Nonin 3230 | FDA review reported ±2% SpO2 accuracy in non-motion hypoxia testing, but the indication is spot-checking rather than continuous monitoring. [FDA K131021](https://www.accessdata.fda.gov/cdrh_docs/pdf13/K131021.pdf) | Optional spot-check only |

## Higher-cost references: borrow, do not buy for phase 1

| Device | Public price / planning range | ECG | Raw PPG | HR | IBI / RR | Resp. | EDA | SpO2 | Temp. | Motion | Beat-to-beat BP | Live host path |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| [VitalConnect VitalPatch RTM / 4](https://vitalconnect.com/docs/ifu034/IFU-034_RevA_VitalPatch4_InstructionsforUse.pdf) | 2021 benchmark: [$150–250/patch + $100–200/month](https://www.dhs.gov/sites/default/files/2022-02/2425-01_SAVER_PhysMonitoring_MSR_12Jan2022-508.pdf) | ✓ | ✕ | ✓ | ✓ | ✓ | ✕ | ✕ | ✓ | ✓ | ✕ | △ Vendor API |
| [Vivalink VV330](https://www.vivalink.com/vivalink-sdk) | Planning: ≈$10,500/36 months = [$500 sensor](https://vtechworks.lib.vt.edu/server/api/core/bitstreams/8582911f-10f2-4485-888b-8a7825510ed6/content) + [$10,000 platform](https://aws.amazon.com/marketplace/pp/prodview-dbrmhy62ylw4g) | ✓ | ✕ | ✓ | ✓ | ✕ | ✕ | ✕ | ✕ | ✓ | ✕ | △ Vendor SDK |
| [Sibel ANNE Chest + Limb](https://accessgudid.nlm.nih.gov/devices/00860004541745) | Low-confidence planning: [$15,000–30,000/system](https://www.medindexer.com/knowledge/fetal-monitors-price-estimate) | ✓ | ✓ | ✓ | ✓ | ✓ | ✕ | ✓ | ✓ | ✓ | ✕ | △ Vendor SDK |
| [Nonin WristOx2 3150 BLE](https://www.nonin.com/products/wristox2-model-3150-with-ble/) | [$1,049.99 single; $1,689.99 kit](https://www.tri-anim.com/ths/diagnostics-monitoring/pulse-oximeters/wristox2-model-3150-oem-with-bluetooth-low-energy/p/group004755) | ✕ | ✕ | ✓ | ✕ | ✕ | ✕ | ✓ | ✕ | ✕ | ✕ | ✓ BLE |
| [BIOPAC BN-PPGED](https://www.biopac.com/product/bionomadix-ppg-and-eda-amplifier/) | Historical component subtotal ≈$7,381–10,681 | ✕ | ✓ | △ | △ | ✕ | ✓ | ✕ | ✕ | ✕ | ✕ | ✓ Receiver |
| [Caretaker VitalStream](https://www.smartmeddevices.com/vitalstream/) | FY25 procurement: [$6,468–7,143](https://www.uth.edu/buy/documents/transparency-reports-/NonOpenMarketPOs-FY25.xlsx) | ✕ | ✕ | ✓ | ✓ | ✓ | ✕ | ✕ | ✕ | ✕ | ✓ | △ Vendor API |
| [Finapres NOVA](https://www.finapres.com/products/hardware/finapres-nova/finapres-nova-basic) | FY20 procurement: [$40,865–58,555](https://www.uth.edu/buy/documents/transparency-reports-/FY20_NON_OPENMARKET_OVER15K_FOR%20POSTING84.xlsx) | △ | ✕ | ✓ | ✓ | ✕ | ✕ | △ | ✕ | ✕ | ✓ | ✓ Analog outputs |

## Near-budget EDA systems not selected

| Device | Complete price | Reason |
|---|---:|---|
| [EmotiBit bundle](https://www.emotibit.com/product/all-in-one-emotibit-bundle/) | $549.97 | Excellent open PPG + EDA + motion + temperature, but above the $400 complete-hardware gate |
| [Shimmer Consensys GSR](https://www.shimmersensing.com/product/consensys-gsr-development-kits/) | €723+ | Strong real-time EDA + PPG research platform, but BITalino and eSense are much cheaper |

## Phase-1 purchase

| Tier | Devices | Cost before tax/shipping/host | Action |
|---|---|---:|---|
| Core | Samsung Watch8 + Polar H10 + Mindfield eSense | $454.94 + €149–159 + OSC/LSL add-on | Buy and bench |
| Add raw PPG | Polar Verity Sense | +$104.95 | Buy |
| Add low-cost EDA/ECG reference | BITalino Board Kit | +$180 | Buy |
| Expanded total | All five devices above | $739.89 + €149–159 + OSC/LSL add-on | Preferred phase-1 stack |
| Hemodynamic reference | Finapres NOVA; Caretaker if unavailable | Borrowed | One validation session |
| Optional SpO2 | Nonin 3230 | +$159–275 | Spot-check only; not a primary detector |

U.S.-vendor devices dominate the consumer and SpO2 alternatives. The open, sub-$400 raw-PPG and EDA options that met the live-access gate are currently Polar, BITalino, and Mindfield.

## Phase-1 pass gates

| Gate | Pass condition |
|---|---|
| Sensor-to-process latency | P95 ≤2 s |
| Sample continuity | ≥99%; no unexplained primary-stream gap >2 s |
| Cross-device timing | Median absolute error ≤100 ms; P95 ≤250 ms |
| Signal fidelity | Pre-registered ECG, PPG, and EDA reference comparisons; Bland–Altman limits and coverage reported |
| Workflow | Pass screen-off, reconnect, motion, cold-finger, low-perfusion, skin-pigmentation, fit, and either-hand tests |

> Research and pre-procurement use only. Detector output must not trigger a patient alert or autonomous robot action before prospective clinical, human-factors, regulatory, and quality-system validation.
