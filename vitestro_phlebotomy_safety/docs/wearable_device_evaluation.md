# Vitestro Automated Phlebotomy: Real-Time Detector Evaluation

**Updated:** 2026-07-28

**Scope:** live presyncope/discomfort sensing during venipuncture

**Selection rule:** retain only devices with a documented real-time sensor-to-host path

Post-study CSV export, sleep scores, and app-only summaries are not scored.

## Live feature matrix

`✓` continuous live value or directly derivable from a live raw waveform · `△` on-demand, optional, processed-only, or vendor/partner-gated · `✕` no suitable live path confirmed

Price basis, checked 2026-07-28: new US list price for the minimum publicly
purchasable hardware. Tax, shipping, host phone/tablet/computer, SDK/API/cloud
licenses, consumables, and service contracts are excluded unless stated.
`Quote required` means no reliable public complete-system price is available.

| Metric | ECG | Raw PPG | HR | IBI / RR | Resp. | EDA | SpO2 | Temp. | Motion | Beat-to-beat BP |
|---|---|---|---|---|---|---|---|---|---|---|
| Meaning | Cardiac electrical waveform | Optical pulse waveform | Heart rate | Interbeat / R-R interval | Respiratory rate | Electrodermal activity / sweating | Oxygen saturation | Skin/body temperature | Accelerometer/gyroscope | Systolic/mean/diastolic pressure for every beat |

| Detector and placement | US price / complete-system quote | ECG | Raw PPG | HR | IBI / RR | Resp. | EDA | SpO2 | Temp. | Motion | Beat-to-beat BP | Live host path |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| [Samsung Galaxy Watch8](https://developer.samsung.com/health/sensor/guide/data-specifications.html) — wrist | [From $349.99](https://www.samsung.com/us/watches/galaxy-watch8/); phone and SDK approval excluded | △ | ✓ | ✓ | ✓ | ✕ | ✓ | △ | ✓ | ✓ | ✕ | △ Partner SDK |
| [Polar H10](https://www.polar.com/en/science/research-tools) — chest strap | [$104.95 list](https://www.polar.com/us-en/sensors/h10); collector excluded | ✓ | ✕ | ✓ | ✓ | ✕ | ✕ | ✕ | ✕ | ✓ | ✕ | ✓ Open BLE |
| [VitalConnect VitalPatch RTM / 4](https://vitalconnect.com/docs/ifu034/IFU-034_RevA_VitalPatch4_InstructionsforUse.pdf) — chest patch | [Partner quote required](https://vitalconnect.com/temporary-home-page/): patches + relay + API/platform | ✓ | ✕ | ✓ | ✓ | ✓ | ✕ | ✕ | ✓ | ✓ | ✕ | △ Relay / API |
| [Vivalink VV330](https://www.vivalink.com/vivalink-sdk) — chest patch | [Vendor quote required](https://www.vivalink.com/dev-app): sensor + adhesives + SDK/API license | ✓ | ✕ | ✓ | ✓ | ✕ | ✕ | ✕ | ✕ | ✓ | ✕ | △ Vendor SDK |
| [Sibel ANNE Chest + ANNE Limb](https://accessgudid.nlm.nih.gov/devices/00860004541745) — chest + finger/limb | [Vendor quote required](https://sibelhealth.com/request-demo/): both sensors + chargers + software/SDK | ✓ | ✓ | ✓ | ✓ | ✓ | ✕ | ✓ | ✓ | ✓ | ✕ | △ Vendor SDK |
| [Nonin WristOx2 3150 BLE](https://www.nonin.com/products/wristox2-model-3150-with-ble/) — wrist + finger probe | [$1,049.99 single; $1,689.99 starter kit](https://www.tri-anim.com/ths/diagnostics-monitoring/pulse-oximeters/wristox2-model-3150-oem-with-bluetooth-low-energy/p/group004755) | ✕ | ✕ | ✓ | ✕ | ✕ | ✕ | ✓ | ✕ | ✕ | ✕ | ✓ BLE |
| [BIOPAC BioNomadix BN-PPGED](https://www.biopac.com/product/bionomadix-ppg-and-eda-amplifier/) — wrist + finger/palm | [Vendor quote required](https://www.biopac.com/product/bionomadix-ppg-and-eda-amplifier/): BN-PPGED + MP200/160 + AcqKnowledge | ✕ | ✓ | △ | △ | ✕ | ✓ | ✕ | ✕ | ✕ | ✕ | ✓ Receiver |
| [Caretaker VitalStream](https://www.smartmeddevices.com/vitalstream/) — finger sensor | [Vendor quote required](https://caretakermedical.net/vitalstream/): monitor + finger sensors + API/cloud | ✕ | ✕ | ✓ | ✓ | ✓ | ✕ | ✕ | ✕ | ✕ | ✓ | △ Vendor API |
| [Finapres NOVA](https://www.finapres.com/products/hardware/finapres-nova/finapres-nova-basic) — finger cuffs | [Vendor quote required](https://www.finapres.com/request-for-quote): NOVA Basic; software modules extra | △ | ✕ | ✓ | ✓ | ✕ | ✕ | △ | ✕ | ✕ | ✓ | ✓ Analog outputs |
| [Apple Watch Series 11](https://developer.apple.com/documentation/HealthKit/running-workout-sessions) — wrist | [From $399](https://www.apple.com/shop/buy-watch/apple-watch); iPhone excluded | △ | ✕ | ✓ | ✕ | ✕ | ✕ | ✕ | ✕ | ✓ | ✕ | ✓ Public APIs |

## Quality and evidence

| Detector | Public acquisition or quality evidence | Sources |
|---|---|---|
| Samsung Galaxy Watch8 | PPG 25 Hz; HR/IBI and EDA 1 Hz; ACC 25 Hz. A Galaxy Watch6 HUT study reported AUROC 0.91 for a five-minute presyncope window. | [SDK](https://developer.samsung.com/health/sensor/guide/data-specifications.html) · [VVS study](https://academic.oup.com/ehjdh/article/7/4/ztag053/8586837) |
| Polar H10 | Live raw ECG, RR, and ACC. A 2026 healthy-adult validation reported concordance at least 0.99 and HRV error below 1%. | [Interface](https://www.polar.com/en/science/research-tools) · [Validation](https://pubmed.ncbi.nlm.nih.gov/42275859/) |
| VitalPatch | Continuous ECG, HR, respiration, temperature, activity, and posture; respiratory error increases with some movements. | [IFU](https://vitalconnect.com/docs/ifu034/IFU-034_RevA_VitalPatch4_InstructionsforUse.pdf) · [Motion study](https://pubmed.ncbi.nlm.nih.gov/34524087/) |
| Vivalink VV330 | ECG 128 Hz; published implementations use ACC at 50 Hz. | [SDK](https://www.vivalink.com/vivalink-sdk) · [GUDID](https://accessgudid.nlm.nih.gov/devices/00865064000157) · [Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC13082138/) |
| Sibel ANNE Chest + Limb | Peer-reviewed methods report ECG 512 Hz and finger PPG 128 Hz. | [Chest](https://accessgudid.nlm.nih.gov/devices/00860004541745) · [Limb](https://accessgudid.nlm.nih.gov/devices/00860004541752) · [Methods](https://www.nature.com/articles/s41746-024-01287-2) |
| Nonin 3150 BLE | Continuous SpO2 and pulse rate; public documentation does not establish raw PPG access. | [IFU](https://www.nonin.com/wp-content/uploads/3150-IFU.pdf) · [Integration study](https://pmc.ncbi.nlm.nih.gov/articles/PMC8057385/) |
| BIOPAC BN-PPGED | PPG and EDA transmit at 2,000 Hz with 10 Hz signal bandwidth. In one HUT cohort, EDA rose in 62% but did not rise in 25%. | [System](https://www.biopac.com/product/bionomadix-ppg-and-eda-amplifier/) · [EDA study](https://pubmed.ncbi.nlm.nih.gov/15316839/) |
| Caretaker VitalStream | Continuous pulse-wave and beat-to-beat BP; FDA clearance is for adults at rest. | [Product](https://www.smartmeddevices.com/vitalstream/) · [FDA K211588](https://www.accessdata.fda.gov/cdrh_docs/pdf21/K211588.pdf) |
| Finapres NOVA | Continuous finger and reconstructed brachial BP with eight analog outputs. | [Specification](https://www.finapres.com/products/hardware/finapres-nova/finapres-nova-basic) · [FDA K173916](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K173916) |
| Apple Watch | Public APIs expose processed live HR and motion, not continuous raw PPG, EDA, ECG, SpO2, temperature, or BP. | [HealthKit](https://developer.apple.com/documentation/HealthKit/running-workout-sessions) · [Core Motion](https://developer.apple.com/documentation/coremotion/) · [Transient-HR study](https://pubmed.ncbi.nlm.nih.gov/41157371/) |

## Phase-1 configuration

| Measurement role | Device | Action |
|---|---|---|
| Primary watch | Samsung Galaxy Watch8 | Bench now |
| Cardiac timing | Polar H10 | Bench now |
| EDA + fingertip PPG | BIOPAC BN-PPGED | Bench now |
| Hemodynamic ground truth | Finapres NOVA; Caretaker if Finapres is unavailable | Borrow or evaluate |
| SpO2 / peripheral pulse | Nonin 3150 BLE | Add to bench |
| Integrated clinical platform | Sibel ANNE Chest + Limb | Vendor demonstration |
| Clinical ECG patch | VitalPatch versus Vivalink | One-device bake-off |
| Consumer baseline | Apple Watch | Baseline only |

## Phase-1 pass gates

| Gate | Pass condition |
|---|---|
| Sensor-to-process latency | P95 ≤ 2 s |
| Sample continuity | ≥99%; no unexplained primary-stream gap >2 s |
| Cross-device timing | Median absolute error ≤100 ms; P95 ≤250 ms |
| Signal fidelity | Pre-registered ECG, BP, and SpO2 reference comparison; Bland–Altman limits and coverage reported |
| Workflow | Pass screen-off, reconnect, motion, cold-finger, low-perfusion, skin-pigmentation, fit, adhesive, and either-hand tests |

> Research and pre-procurement use only. Detector output must not trigger a patient alert or autonomous robot action before prospective clinical, human-factors, regulatory, and quality-system validation.
