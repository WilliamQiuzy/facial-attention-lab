# Vitestro Automated Phlebotomy: Real-Time Detector Evaluation

**Version:** 2026-07-26

**Scope:** detector quality during venipuncture, not retrospective wellness reporting

**Status:** research and pre-procurement recommendation; not a validated alerting system

## Decision

Only devices with a confirmed live measurement path are retained below. Historical
CSV export, sleep summaries, recovery scores, and values visible only in a vendor
application are not selection criteria.

The best first prototype is not a single watch. It is a synchronized stack:

1. **Samsung Galaxy Watch8** for live wrist PPG, IBI, EDA, temperature, and motion.
2. **Polar H10** for higher-fidelity cardiac timing.
3. **BIOPAC BN-PPGED** for fingertip PPG and palmar/finger EDA during signal discovery.
4. **Finapres NOVA or Caretaker VitalStream** for continuous blood-pressure reference.

**Sibel ANNE Chest + ANNE Limb** is the most complete integrated clinical candidate.
It should replace the older “ANNE One” name in procurement discussions.

## Real-time detector table

“Live access” means that timestamped measurements can reach study software during
the procedure. It does not mean that the device is already valid for presyncope
detection.

| Detector and placement | Confirmed live signals | Live path | Measurement-quality evidence | Main limitation in Vitestro | Recommended role |
|---|---|---|---|---|---|
| **Samsung Galaxy Watch8** — wrist | Raw green/red/IR PPG at 25 Hz; HR/IBI at 1 Hz; raw EDA at 1 Hz; skin/ambient temperature; 25 Hz accelerometer. ECG and SpO2 are on-demand, not continuous. | On-watch event stream through Samsung Health Sensor SDK; a companion app can forward data to the study host. Production distribution requires Samsung partner approval. | It is the only retained watch with direct vasovagal-syncope evidence: a Galaxy Watch6 study reported AUROC 0.91 for a five-minute HUT presyncope window. | Wrist PPG is motion- and perfusion-sensitive; direct evidence is from controlled tilt testing, not venipuncture. EDA is only available on Watch8 or later. | **Primary smartwatch prototype.** [SDK specification](https://developer.samsung.com/health/sensor/guide/data-specifications.html); [VVS study](https://academic.oup.com/ehjdh/article/7/4/ztag053/8586837) |
| **Polar H10** — chest strap | Raw single-lead ECG, HR, R-R intervals, and 3-axis acceleration. | Direct Bluetooth LE stream through the open Polar BLE SDK to iOS, Android, or a local bridge. | Strong agreement with reference ECG has been reported for linear HRV and autonomic-reflex measures; a 2026 study reported concordance ≥0.99 and mean absolute percentage error below 1% in healthy adults. | Not a medical monitor; chest-strap fit and contact must be checked in seated participants. It provides no PPG, EDA, SpO2, or BP. | **Low-cost cardiac-timing reference and deployable sensor candidate.** [Official research interface](https://www.polar.com/en/science/research-tools); [validation](https://pubmed.ncbi.nlm.nih.gov/42275859/) |
| **VitalConnect VitalPatch RTM / VitalPatch 4** — chest patch | Continuous ECG, HR, R-R/HRV, respiration, temperature, activity, and posture. | Patch-to-relay streaming with seconds-scale transmission and a Relay Software Library/API. | FDA-cleared secondary physiological monitor with continuous live ECG; published testing supports HR performance, although respiratory-rate error increases in some movements. | Single-use adhesive, skin reaction, placement time, and vendor integration. It is a secondary monitor and cannot replace clinical observation. | **US clinical patch finalist.** [Current IFU](https://vitalconnect.com/docs/ifu034/IFU-034_RevA_VitalPatch4_InstructionsforUse.pdf); [motion study](https://pubmed.ncbi.nlm.nih.gov/34524087/) |
| **Vivalink VV330 Continuous ECG Platform** — chest patch | 128 Hz single-lead ECG, HR, R-R intervals, and accelerometry; published deployments use 50 Hz acceleration. | Local mobile/edge SDK plus cloud services; FDA record confirms wireless transmission to a mobile host for display and storage. | FDA-cleared continuous ECG platform with direct raw-signal integration and current commercial distribution. | Adhesive workflow and a narrower signal set than Sibel; exact SDK fields, packet latency, clocks, and model number must be contract-locked. | **US open-integration ECG patch finalist.** [SDK](https://www.vivalink.com/vivalink-sdk); [FDA device record](https://accessgudid.nlm.nih.gov/devices/00865064000157); [methods example](https://pmc.ncbi.nlm.nih.gov/articles/PMC13082138/) |
| **Sibel ANNE Chest + ANNE Limb** — chest and finger/limb | Synchronized chest ECG, acceleration and temperature plus limb/finger PPG, SpO2, pulse rate, and temperature. Peer-reviewed methods report 512 Hz ECG and 128 Hz finger PPG. | Bluetooth streaming to compatible software through the Sibel SDK; current GUDID records list both sensors in commercial distribution. | FDA-cleared components provide synchronized electrical and peripheral optical measurements in one platform. | Institutional/demo procurement, adhesives, a finger sensor that may affect hand use, and SDK terms. The exact deployment configuration must be tested; “ANNE One” is the older platform name. | **Most complete integrated clinical finalist.** [GUDID Chest](https://accessgudid.nlm.nih.gov/devices/00860004541745); [GUDID Limb](https://accessgudid.nlm.nih.gov/devices/00860004541752); [methods](https://www.nature.com/articles/s41746-024-01287-2) |
| **Nonin WristOx2 3150 BLE** — wrist unit with finger probe | Continuous SpO2 and pulse rate over Bluetooth LE. | Direct BLE connection to a collector device; real-time research integrations have been demonstrated. | FDA-cleared pulse oximeter intended for well- or poorly-perfused patients; Nonin specifically markets performance under low perfusion and motion. | Public documentation does not establish access to raw PPG. SpO2 may change too slowly for the earliest warning, and finger cold/vasoconstriction can reduce signal quality. | **Finger-based perfusion, SpO2, and pulse-rate auxiliary.** [Official product](https://www.nonin.com/products/wristox2-model-3150-with-ble/); [IFU](https://www.nonin.com/wp-content/uploads/3150-IFU.pdf); [real-time integration study](https://pmc.ncbi.nlm.nih.gov/articles/PMC8057385/) |
| **BIOPAC BioNomadix BN-PPGED** — wrist transmitter with fingertip/palmar sensors | Simultaneous raw fingertip PPG/BVP and EDA/GSR; 2,000 Hz wireless transmission with both channels band-limited to 10 Hz. | Live wireless telemetry to an MP200/MP160 receiver and AcqKnowledge. | Purpose-built US research instrumentation provides direct, high-resolution access to sweating and peripheral pulse morphology. EDA rose before syncope in 62% of one HUT cohort, but was absent in 25%. | Research equipment, not a clinical monitor; finger electrodes can interfere with workflow. Pain, anxiety, temperature, and venipuncture itself can all change EDA. | **Primary EDA/hand-sweat and fingertip-PPG development reference.** [Official system](https://www.biopac.com/product/bionomadix-ppg-and-eda-amplifier/); [EDA study](https://pubmed.ncbi.nlm.nih.gov/15316839/) |
| **Caretaker VitalStream** — low-pressure finger sensor | Continuous beat-to-beat BP, pulse waveform, HR, respiration, cardiac output, stroke volume, and other hemodynamic parameters. | Live Android tablet waveforms plus wired/wireless transmission and Caretaker Remote Monitor APIs. | FDA-cleared for continuous or spot-check hemodynamic measurement in trained hands; the device analyzes pulse-wave morphology 500 times per second. | Requires calibration and is cleared for adults at rest. Finger vasoconstriction, hand motion, sensor placement, and compatibility with the robot workflow require direct testing. | **Wearable continuous-BP candidate and US reference finalist.** [Official product](https://www.smartmeddevices.com/vitalstream/); [FDA K211588](https://www.accessdata.fda.gov/cdrh_docs/pdf21/K211588.pdf) |
| **Finapres NOVA** — dual finger cuff | Continuous finger and reconstructed brachial BP waveforms; beat-to-beat systolic, mean and diastolic pressure; HR and IBI. | Real-time display plus eight configurable analog outputs for synchronized acquisition. | FDA-cleared, established continuous noninvasive BP system with upper-arm calibration; suited to reference measurement rather than unobtrusive deployment. | Approximately 5 kg, costly, and finger-cuff pressure/Physiocal recalibration can interrupt or perturb measurements. It may not fit the final product. | **Primary hemodynamic ground truth for Phase 1.** [Official specification](https://www.finapres.com/products/hardware/finapres-nova/finapres-nova-basic); [FDA K173916](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K173916) |
| **Apple Watch Series 11** — wrist | Processed HR during an active workout plus live accelerometer and gyroscope data. | HealthKit live workout session and Core Motion. | Useful ecosystem and usability baseline; consumer HR can be acceptable in stable conditions. | No public continuous raw PPG, EDA, continuous ECG, SpO2, or BP stream. Wrist HR accuracy degrades during rapid transient changes. | **Baseline only; not the primary physiological detector.** [HealthKit](https://developer.apple.com/documentation/HealthKit/running-workout-sessions); [Core Motion](https://developer.apple.com/documentation/coremotion/); [transient-HR study](https://pubmed.ncbi.nlm.nih.gov/41157371/) |

## Phase-1 quality gates

| Test | Initial pass condition | Why it matters |
|---|---|---|
| End-to-end latency | P95 ≤ 2 seconds from sensor timestamp to the Vitestro study process | A “live” vendor dashboard is insufficient if the signal reaches the detector too late. |
| Sample continuity | ≥99% expected samples and no unexplained primary-stream gap longer than 2 seconds | Short gaps can remove the rapid transition immediately before presyncope. |
| Time alignment | Median absolute clock error ≤100 ms and P95 ≤250 ms across device, robot events, ECG, and BP reference | Multimodal features are invalid when PPG, ECG, EDA, BP, and robot stages are misaligned. |
| Signal fidelity | Pre-registered agreement against ECG, Finapres/VitalStream, and reference pulse oximetry; report Bland–Altman limits and coverage | Correlation alone does not establish interchangeable measurements. |
| Workflow robustness | Test screen off/on, Bluetooth interruption, motion, cold fingers, low perfusion, skin pigmentation, wrist fit, adhesives, and use of either hand | These conditions are more important than post-study file format. |

The thresholds above are engineering targets for a bench study, not clinical
performance claims.

## Procurement order

1. Bench **Samsung Galaxy Watch8, Polar H10, Nonin 3150 BLE, and BIOPAC BN-PPGED**.
2. Use an existing **Finapres NOVA** if Mayo has access; otherwise obtain a **Caretaker VitalStream** evaluation unit.
3. Request one integrated-platform demonstration from **Sibel**, and compare it with one ECG patch from **VitalConnect or Vivalink**.
4. Keep **Apple Watch** only as the consumer baseline.

Devices removed from the primary table include ActiGraph LEAP, Biostrap Kairos,
Empatica EmbracePlus, Fitbit, Garmin, WHOOP, Pixel Watch, Huawei Watch D2,
BioButton, BioStamp, iRhythm Zio AT, and Masimo W1. They may record useful data,
but a suitable programmatic live research stream was not confirmed for the
current Vitestro workflow. Empatica explicitly states that its academic
EmbracePlus plan does not support live streaming or an SDK
([official FAQ](https://www.empatica.com/en-us/platform/research-studies/)).

No experimental detector output may trigger a patient alert or autonomous robot
action until prospective clinical, human-factors, regulatory, and quality-system
gates are passed.
