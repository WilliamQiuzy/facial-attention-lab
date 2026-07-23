# Vitestro Automated Phlebotomy: Wearable Device Evaluation

**Version:** 2026-07-23

**Evidence base:** peer-reviewed literature, regulatory material, official product pages, and developer documentation

**Baseline:** Apple Watch Series 11
**Status:** research and pre-procurement recommendation; not a validated medical device or patient-management protocol

## Executive decision

Apple Watch remains the ecosystem and user-experience baseline, but it is not the strongest device for algorithm development. It does not expose continuous raw PPG, EDA, or actual blood pressure through public interfaces.

The recommended first bench set is US-first:

1. **Apple Watch Series 11** — consumer and iOS baseline.
2. **ActiGraph LEAP** — primary US research wrist device because it exposes raw multiwavelength PPG and motion data through a clinical-research platform.
3. **Biostrap Kairos** — second US wrist device for raw PPG/IBI access and a configurable research SDK/API path.
4. **VitalConnect VitalPatch RTM** — US chest-patch reference for continuous ECG, R-R intervals, respiration, temperature, activity, and posture.

Two non-US devices remain scientifically justified controls:

- **Samsung Galaxy Watch8** is the direct-evidence smartwatch benchmark. A 2026 prospective study used 25 Hz raw PPG from Galaxy Watch6 in 132 patients and reported AUROC 0.91 for a five-minute presyncope window, with specificity 0.64 at sensitivity 0.90. This is promising but was performed during controlled head-up tilt testing, not venipuncture ([European Heart Journal – Digital Health](https://academic.oup.com/ehjdh/article/7/4/ztag053/8586837)).
- **Empatica EmbracePlus** is the autonomic reference when raw EDA is required.

No reviewed device is ready for patient alerts or autonomous robot actions. The first phase must validate measurement quality, latency, timestamps, and exportability in the actual Vitestro workflow.

## What the literature changes

The expanded literature review supports a multimodal design and argues against relying on a consumer “blood pressure” feature:

- A meta-analysis covering 71 studies and approximately 19 million blood donations found higher vasovagal-reaction risk with younger age, first donation, smaller blood volume, and lower blood pressure; fear and anxiety were also relevant in narrative synthesis ([Wu et al., 2025](https://pubmed.ncbi.nlm.nih.gov/39587929/)).
- In 1,155 tilt-test patients, a model combining continuous systolic blood-pressure and R-R trends/variability reported 95% sensitivity and 93% specificity ([Virag et al., 2007](https://pubmed.ncbi.nlm.nih.gov/17954394/)).
- A multisensor cuffless patch failed to reproduce the rapid systolic-pressure fall before reflex syncope: the reference fell by 53.5 mmHg while the patch changed by only 1 mmHg ([Groppelli et al., 2023](https://pubmed.ncbi.nlm.nih.gov/37208523/)).
- Wrist-wearable heart-rate accuracy falls during rapid transient changes, so steady-state validation is not enough for an acute presyncope use case ([Schuurmans et al., 2025](https://pubmed.ncbi.nlm.nih.gov/41157371/)).
- PPG-derived pulse-rate variability is not automatically equivalent to ECG-derived HRV; the source signal must be reported explicitly ([Kantrowitz et al., 2025](https://pubmed.ncbi.nlm.nih.gov/40809286/)).
- Optical performance must be evaluated by objective skin pigmentation, perfusion, motion, fit, and wrist circumference. A 2024 systematic review found meaningful accuracy concerns across pigmentation groups for SpO2 and wide limits of agreement for wearable pulse rate ([Singh et al., 2024](https://pubmed.ncbi.nlm.nih.gov/39388258/)).

## Device selection policy

The evaluation pool contains 15 current devices:

- **11 US or US-operationally-headquartered products:** Apple Watch, ActiGraph LEAP, Biostrap Kairos, VitalPatch RTM, Pixel Watch, Fitbit Sense 2, Garmin Venu 4, WHOOP MG, BioButton Rechargeable, Masimo W1 Medical, and iRhythm Zio AT.
- **4 non-US technical controls:** Samsung Galaxy Watch8, Empatica EmbracePlus, Corsano CardioWatch 287-2, and Huawei Watch D2.

The non-US controls were retained only when they provide unusually relevant evidence or capabilities not available in the US-first consumer set.

## Physiological capability matrix

`✓` = supported in a relevant mode; `△` = spot, sleep-only, intermittent, derived, region-limited, or contract-gated; `✗` = not confirmed in public documentation.

| Device | HR | IBI / HRV | Raw PPG / BVP | ECG | Actual BP | Continuous BP | SpO2 | Skin temperature | EDA | Respiration | Motion / posture |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Apple Watch Series 11 — baseline** | ✓ | △ | ✗ | △ | ✗ | ✗ | △ | △ | ✗ | △ | ✓ |
| **ActiGraph LEAP — US primary** | ✓ | ✓ | ✓ | ✗ | △ | ✗ | △ | ✓ | ✗ | △ | ✓ |
| **Biostrap Kairos — US primary** | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | △ | ✗ | ✗ | ✓ | ✓ |
| **VitalConnect VitalPatch RTM — US reference** | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ | ✓ |
| Google Pixel Watch 4 | ✓ | △ | ✗ | △ | ✗ | ✗ | △ | △ | ✓ | △ | ✓ |
| Fitbit Sense 2 | ✓ | △ | ✗ | △ | ✗ | ✗ | △ | △ | ✓ | △ | ✓ |
| Garmin Venu 4 | ✓ | ✓ | ✗ | △ | ✗ | ✗ | △ | ✓ | ✗ | ✓ | ✓ |
| WHOOP MG | ✓ | △ | ✗ | △ | △ | ✗ | △ | △ | ✗ | △ | ✓ |
| BioIntelliSense BioButton Rechargeable | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ | ✓ |
| Masimo W1 Medical | ✓ | △ | ✗ | △ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ |
| iRhythm Zio AT | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | △ |
| **Samsung Galaxy Watch8 — evidence control** | ✓ | ✓ | ✓ | △ | △ | ✗ | △ | ✓ | ✓ | △ | ✓ |
| **Empatica EmbracePlus — autonomic control** | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | △ | ✓ | ✓ | △ | ✓ |
| Corsano CardioWatch 287-2 | ✓ | ✓ | ✓ | △ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Huawei Watch D2 | ✓ | △ | ✗ | △ | ✓ | △ | ✓ | ✓ | ✗ | △ | ✓ |

### Blood-pressure interpretation

- Apple hypertension notifications, WHOOP Blood Pressure Insights, and Samsung cuff-calibrated spot measurements are not continuous beat-to-beat blood pressure.
- ActiGraph’s listed upper-arm BP is an external measurement pathway, not a LEAP wrist signal.
- Huawei D2 uses an inflatable wrist cuff for intermittent ambulatory measurements, typically at intervals far longer than an acute presyncope event.
- Corsano’s continuous cuffless BP claim remains a vendor claim until the exact US configuration, indication, calibration method, sample rate, latency, and fast-change performance are independently verified.
- The pilot must use a synchronized continuous reference BP system. The FDA’s cuffless-BP draft guidance specifically emphasizes dynamic-change testing against an adequately time-resolved reference ([FDA](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/cuffless-non-invasive-blood-pressure-measuring-devices-clinical-performance-testing-and-evaluation)).

## Real-time and data-extraction matrix

“Real-time” means a study application can programmatically receive timestamped samples. A value visible only in a vendor application does not qualify.

| Device | Programmatic live data | Raw waveform export | API / SDK | Manual export | Main constraint |
|---|:---:|:---:|:---:|:---:|---|
| Apple Watch Series 11 | △ | △ | ✓ | ✓ | Live processed HR/motion; saved ECG voltage; no raw PPG/EDA/BP |
| ActiGraph LEAP | △ | ✓ | ✓ | ✓ | Raw data are platform-accessible after transfer; live latency is not publicly established |
| Biostrap Kairos | △ | ✓ | ✓ | ✓ | Contract-dependent raw PPG/IBI, cloud, API, and Bluetooth SDK |
| VitalConnect VitalPatch RTM | ✓ | ✓ | ✓ | ✓ | Clinical platform; chest placement and adhesive workflow must be tested |
| Google Pixel Watch 4 | △ | ✗ | ✓ | ✓ | Processed HR; passive updates may be batched |
| Fitbit Sense 2 | ✗ | ✗ | △ | ✓ | Historical processed API/export; multi-user intraday access requires review |
| Garmin Venu 4 | △ | △ | △ | ✓ | Enterprise license; mainly processed signals |
| WHOOP MG | ✗ | ✗ | ✓ | ✓ | Aggregate/post-sync API; no public raw PPG/IBI |
| BioButton Rechargeable | △ | ✗ | △ | △ | Rest-only HR/RR outputs and platform integration; no public raw study stream |
| Masimo W1 Medical | △ | ✗ | △ | △ | Continuous UI/clinical pathway, but no open raw research interface |
| iRhythm Zio AT | △ | △ | △ | △ | Continuous ECG/event transmission through a clinical service, not an open research stream |
| Samsung Galaxy Watch8 | ✓ | ✓ | △ | ✓ | Rich Sensor SDK; partner approval required for distribution |
| Empatica EmbracePlus | △ | ✓ | ✓ | ✓ | Continuous recording; cloud latency and entitlements require contract confirmation |
| Corsano CardioWatch 287-2 | ✓ | ✓ | ✓ | ✓ | Endpoint-specific US regulatory scope and fast-BP validation remain open |
| Huawei Watch D2 | ✗ | ✗ | △ | △ | Region-limited cloud/Health Kit route; BP is intermittent |

## Device-level recommendation

| Priority | Device | Recommended Vitestro role | Decision |
|---|---|---|---|
| 1 | ActiGraph LEAP | Primary US wrist research platform | Bench now |
| 2 | Biostrap Kairos | Independent US raw-PPG/IBI wrist platform | Bench now |
| 3 | VitalConnect VitalPatch RTM | ECG/R-R/respiration/posture reference channel | Bench now |
| 4 | Apple Watch Series 11 | Consumer/iOS usability and processed-HR baseline | Bench now |
| 5 | Samsung Galaxy Watch8 | Replicate the most directly relevant smartwatch VVS paper | Buy one control unit; do not make it the US-first deployment default |
| 6 | Empatica EmbracePlus | Raw EDA/BVP/temperature autonomic control | Add if EDA is retained in the protocol |
| 7 | Corsano CardioWatch 287-2 | Cuffless-BP and all-in-one diligence candidate | No procurement decision before US regulatory and fast-change evidence review |
| 8 | Masimo W1 Medical | Medical wrist SpO2/pulse comparator | Vendor/API diligence |
| 9 | Garmin Venu 4 | Long-wear processed stream | Enterprise-contract diligence |
| 10 | BioButton Rechargeable | Ward-monitoring comparator | Not an acute primary because HR/RR are rest-gated |
| 11 | iRhythm Zio AT | Ambulatory ECG comparator | Useful for rhythm adjudication, not multimodal presyncope prediction |
| 12–15 | Pixel Watch, Fitbit Sense 2, WHOOP MG, Huawei D2 | Ecosystem, longitudinal, or intermittent-BP controls | Do not use as the first acute signal source |

## Evidence table

| Evidence question | Study | Result relevant to Vitestro | Limitation |
|---|---|---|---|
| Can smartwatch PPG predict VVS? | [Kim et al., 2026](https://academic.oup.com/ehjdh/article/7/4/ztag053/8586837) | 132 HUT patients; 25 Hz Galaxy Watch PPG; five-minute window AUROC 0.91 | Single controlled HUT setting; many events followed nitroglycerin |
| Are BP and R-R jointly useful? | [Virag et al., 2007](https://pubmed.ncbi.nlm.nih.gov/17954394/) | 1,155 HUT patients; 95% sensitivity and 93% specificity | Tilt-test algorithm, not venipuncture validation |
| Can cuffless BP follow rapid presyncope hypotension? | [Groppelli et al., 2023](https://pubmed.ncbi.nlm.nih.gov/37208523/) | Tested directly and failed to track the rapid SBP drop | One patch and controlled HUT protocol |
| What mechanisms lower BP? | [Rivasi et al., 2020](https://pubmed.ncbi.nlm.nih.gov/32460687/) | Reduced stroke volume and later cardioinhibition both contributed | Requires reference hemodynamics not available from most watches |
| Which donor factors matter? | [Wu et al., 2025](https://pubmed.ncbi.nlm.nih.gov/39587929/) | 71 studies; lower BP, young age, first donation, low blood volume, fear/anxiety | Baseline risk factors are not acute detectors |
| Are consumer HR values accurate in transient states? | [Schuurmans et al., 2025](https://pubmed.ncbi.nlm.nih.gov/41157371/) | Accuracy worsened during rapid HR changes across wrist devices | Not a venipuncture study |
| Is Apple a reasonable HR baseline? | [Hajj-Boutros et al., 2023](https://pubmed.ncbi.nlm.nih.gov/34957939/) | Apple Watch 6 had the best HR accuracy among Apple, Polar, and Fitbit in the tested activities | Older device generations and exercise tasks |
| Does skin pigmentation matter? | [Singh et al., 2024](https://pubmed.ncbi.nlm.nih.gov/39388258/) | Supports explicit pigmentation-stratified optical validation | Heterogeneous devices and study designs |
| Is PPG PRV equivalent to ECG HRV? | [Kantrowitz et al., 2025](https://pubmed.ncbi.nlm.nih.gov/40809286/) | Significant disagreement across several HRV metrics in 931 adults | Does not invalidate all PPG features; it limits interpretation |
| Are wearable cuffless BP devices established? | [Lee and Chang, 2025](https://pubmed.ncbi.nlm.nih.gov/41321467/) | Systematic review found remaining accuracy and comparability concerns | Device and calibration methods vary |
| Is VitalPatch validated under motion? | [Areia et al., 2021](https://pubmed.ncbi.nlm.nih.gov/34524087/) | HR stayed within the prespecified limit; RR errors increased for some movements | Healthy controlled protocol, not presyncope |
| Does BioButton support acute sensing? | [Weenk et al., 2024](https://pubmed.ncbi.nlm.nih.gov/39200889/) | Large hospital deployment supports trend monitoring | HR/RR are rest measurements; manufacturer conflicts disclosed |
| Is EmbracePlus optically validated? | [Gerboni et al., 2023](https://pubmed.ncbi.nlm.nih.gov/38111608/) | SpO2 validation met the reported FDA error limit under controlled no-motion conditions | Small sample, high perfusion, manufacturer-authored |
| Is Masimo W1 ECG validated? | [Clinical evaluation, 2026](https://pubmed.ncbi.nlm.nih.gov/42038677/) | Strong spot-check AF/normal-rhythm performance | Rhythm classification is not presyncope prediction |

## Phase-1 evaluation protocol

### Gate A — contract and interface

For each exact model, firmware, phone, country, and account tier:

1. Obtain data-rights, SDK, API, retention, deletion, and algorithm-development terms.
2. Record every field’s unit, sample rate, timestamp clock, quality flag, batching rule, and missing-data representation.
3. Run 60-minute sessions with screen on/off, foreground/background, lock screen, low battery, Bluetooth interruption, and phone calls.
4. Reconcile live, cloud, and manual exports from the same session.
5. Downgrade any unapproved partner-only interface to `blocked-by-vendor`.

### Gate B — synchronized signal-quality pilot

- Wear wrist devices on the arm opposite venipuncture.
- Use synchronized three-lead ECG, continuous beat-to-beat BP, and reference SpO2 when SpO2 is evaluated.
- Log Vitestro events: preparation, tourniquet, needle insertion, collection, needle removal, compression, and recovery.
- Capture symptoms separately: anxiety, pain, nausea, dizziness, visual change, sweating, and presyncope.
- Test motion, fit, wrist circumference, wrist side, ambient temperature, perfusion, and objectively measured skin pigmentation.
- Report coverage, longest missing run, clock drift, latency distribution, signal-quality flags, absolute error, and Bland–Altman agreement—not correlation alone.

### Gate C — prospective clinical feasibility

- Pre-register endpoint definitions, warning horizon, primary metrics, subgroup analyses, and model-freeze rules.
- Keep presyncope/syncope, discomfort, anxiety, and pain as separate labels.
- Split data by participant, never by window from the same participant.
- Report sensitivity, specificity, positive predictive value, false alarms per procedure and per hour, warning lead time, calibration, and abstention due to poor signal quality.
- Continue standard clinical observation and response regardless of wearable output.

## Product gates

| Gate | Pass condition | Failure action |
|---|---|---|
| G0 Data rights | Sample-level data can legally be exported and used for algorithm development | Do not procure, or retain as UI-only comparator |
| G1 Real-time path | Required samples are programmatically available with measured latency, loss, and clock behavior | Historical-data role only |
| G2 Signal validity | Predefined agreement and coverage targets are met in the Vitestro workflow and subgroups | Remove the signal or device from the model |
| G3 Event validity | A frozen model meets prospective sensitivity, false-alarm, lead-time, and calibration targets | Research only; no patient alert |
| G4 Human factors | Alerts do not delay standard care or create unsafe robot behavior | No patient-facing or robot integration |
| G5 Regulatory/QMS | Intended use, software lifecycle, cybersecurity, and supplier change control are approved | Research prototype only |

## Final recommendation

Procure the four-device US bench set first: **Apple Watch Series 11, ActiGraph LEAP, Biostrap Kairos, and VitalConnect VitalPatch RTM**. Add **one Samsung Galaxy Watch8** to reproduce the strongest direct smartwatch-VVS evidence and add **Empatica EmbracePlus** only if raw EDA remains a planned feature.

Do not select a “continuous BP watch” as the truth source. Use synchronized continuous reference blood pressure during validation, and treat Corsano as a diligence candidate until its exact US regulatory scope and fast-change performance are independently confirmed.

The machine-readable matrix is in [`../data/device_feature_matrix.csv`](../data/device_feature_matrix.csv), and all product, regulatory, and paper citations are in [`../sources/evidence_registry.csv`](../sources/evidence_registry.csv).
