# Vitestro Phlebotomy Safety: Device Evaluation

Rank balances acquisition quality, patient workflow, integration, price and availability; it is a proposed evaluation order, not a measured accuracy score.
Live streaming means continuous access in our software: Yes is documented, Conditional needs approval or setup, Unconfirmed lacks sufficient evidence, and No is unavailable through the reviewed interface.

## 1. Wrist Watches and Bands

| Rank | Product | Live streaming | Hardware price | Raw PPG | HR | Beat interval | Respiration | EDA | SpO2 | Skin temperature | Motion | ECG | BP |
|---:|---|:---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | [Samsung Galaxy Watch8, 40 mm Bluetooth](https://www.samsung.com/us/watches/galaxy-watch8/buy/galaxy-watch8-40mm-graphite-wi-fi-bluetooth-sku-sm-l320ndadxaa/) | Conditional | $299.99 | Yes | Yes | Yes | No | Yes | No | Yes | Yes | No | No |
| 2 | [Polar Loop](https://www.polar.com/us-en/loop) | Yes | $199.99 | Yes | Yes | Yes | No | No | No | Yes | Yes | No | No |
| 3 | [Garmin Venu 3, 45 mm](https://www.garmin.com/en-US/p/873008/) | Conditional | $299.99 | No | Yes | Conditional | Conditional | No | Conditional | No | Yes | No | No |
| Baseline | [Apple Watch Series 11, 42 mm GPS](https://www.apple.com/apple-watch-series-11/) | Yes | $399 | No | Yes | No | No | No | No | No | Yes | No | No |

| Product | Live interface and signal quality | Patient use and recommendation |
|---|---|---|
| Galaxy Watch8 | Watch app and phone relay. PPG: 25 Hz. HR, beat intervals and EDA: 1 Hz. Distribution requires Samsung approval. SDK data is designated for fitness/wellness, not diagnosis or treatment. [SDK](https://developer.samsung.com/health/sensor/guide/data-specifications.html). | Preferred wrist device for prototype signal coverage. Adjustable band. Wrist EDA is distinct from palmar sweat measurement. The [presyncope paper](https://doi.org/10.1093/ehjdh/ztag053) studied Watch6 during tilt testing. |
| Polar Loop | Public Android/iOS BLE SDK. PPG: 22 Hz. Acceleration: 50 Hz. Temperature: 1 Hz. High-rate mode excludes HR and pulse intervals. Acquisition software must correct PPG timing. [SDK](https://github.com/polarofficial/polar-ble-sdk/blob/master/documentation/products/Polar360.md). | Preferred direct-BLE alternative. Screenless, adjustable band. Textile requires washing and drying. [Care](https://support.polar.com/e_manuals/polar-loop/polar-loop-user-manual-english/caring-for-your-polar-loop.htm). |
| Venu 3 | Partner-gated Garmin Health SDK. HR, beat intervals, respiration, SpO2 and acceleration are SDK-level streams. Venu 3 channel coverage is not specified publicly. [SDK scope](https://developer.garmin.com/health-sdk/overview/). | U.S.-brand alternative for processed vital signs. Silicone band. Conditional on model-specific partner SDK access. [Device](https://www.garmin.com/en-US/p/873008/). |
| Apple Watch Series 11 | Workout HR through HealthKit. Motion through the watch app. Clinic-paired iPhone required for the selected setup. [Live API](https://developer.apple.com/documentation/healthkit/hkliveworkoutbuilder). | Baseline comparator. Cleanable Sport Band. [Care](https://support.apple.com/en-us/108893). The [accuracy review](https://doi.org/10.1038/s41746-025-02238-1) is not a Series 11 presyncope validation. |
| Category limits | The selected interfaces do not provide continuous BP. Contact and motion affect wrist PPG. [Signal-quality study](https://arxiv.org/abs/2307.08766). | Wrist fitting and skin contact are required. This category does not provide a universal patient fit. |

| Other screened products | Live streaming | Hardware price | Assessment |
|---|:---:|---:|---|
| [Google Pixel Watch 4, 41 mm Wi-Fi](https://store.google.com/us/product/pixel_watch_4?hl=en_US) | Yes | $349.99 | Lower-priority wrist alternative. The reviewed public interface provides HR and motion, without Samsung's broader live PPG and EDA coverage. [Health Services](https://developer.android.com/health-and-fitness/health-services). |
| [Fitbit Charge 6](https://store.google.com/us/product/fitbit_charge_6?hl=en-US) | Conditional | $159.95 | HR broadcasting to compatible exercise equipment and apps. Receiver compatibility is platform-dependent. No documented live EDA or raw PPG interface. [Broadcast requirements](https://support.google.com/googlehealth/answer/14236705?hl=en). |

## 2. Smart Rings

| Rank | Product | Live streaming | Hardware price | Raw PPG | HR | Beat interval | SpO2 | Perfusion index | EDA | Skin temperature | Motion |
|---:|---|:---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | [Wellue O2Ring S (Viatom)](https://getwellue.com/products/o2ring-s-pulse-oximeter) | Yes | $186.99 | Unconfirmed | Yes | No | Yes | Yes | No | No | Unconfirmed |
| 2 | [Wellue O2Ring (Viatom)](https://getwellue.com/products/o2ring-wearable-pulse-oximeter) | Yes | $152.90 | Unconfirmed | Yes | No | Yes | Yes | No | No | Yes |

| Product | Live interface and signal quality | Patient use and recommendation |
|---|---|---|
| O2Ring S | Android BLE SDK. Parameters: 1 Hz. Pulse waveform: 125 Hz. Separate PPG interface: 200 Hz. Raw format is not specified. [SDK](https://github.com/viatom-develop/LepuDemo#o2ring-s-bluetoothmodel_o2ring_s). | Preferred ring. Flexible fit, advertised circumference 55-80 mm. Software exposes a two-minute preparation state. [Acquisition example](https://github.com/viatom-develop/LepuDemo/blob/master/app/src/main/java/com/example/lpdemo/OxyIIActivity.kt). |
| O2Ring | Android example polls HR, SpO2, perfusion index and motion each second. Separate PPG interface: 150 Hz. Commands are serialized. [SDK](https://github.com/viatom-develop/LepuDemo#o2ring-bluetoothmodel_o2ring). | Lower-cost ring alternative. Flexible fit. Same manufacturer family as O2Ring S. [Acquisition example](https://github.com/viatom-develop/LepuDemo/blob/master/app/src/main/java/com/example/lpdemo/OxyActivity.kt). |
| Category limits | The selected ring integrations provide oxygenation and pulse parameters, not EDA or BP. | Poor perfusion and finger fit limit coverage. Rings are not a universal substitute for non-contact sensing. [Pulse-oximetry limitations](https://www.fda.gov/consumers/consumer-updates/pulse-oximeters-and-oxygen-concentrators-what-know-about-home-oxygen-therapy). |

| Other screened products | Live streaming | Hardware price | Assessment |
|---|:---:|---:|---|
| [Ultrahuman Ring AIR](https://www.ultrahuman.com/us/ring/buy/better-help/) | Unconfirmed | $349 | UltraSignal uses a selected-applicant loan kit. A retail ring purchase does not establish access to that integration. [Developer program](https://www.ultrahuman.com/us/ultrasignal/). |
| [J-Style JCRing X3](https://www.jointcorp.com/product/x3-blood-oxygen-ring/) | Unconfirmed | [$279](https://jcvital.com/products/jcring-med-x3) | The published SDK offer lacks a model-specific continuous sampling specification. Rigid ring sizes add fitting inventory. [SDK offer](https://www.jointcorp.com/sdk-api/). |
| [Open Ring](https://o-ring.tech/product/open-ring/) | Unconfirmed | EUR 790 | Store lists a preorder product. The open-hardware platform is not a standard ready-to-deploy clinical acquisition package. [Platform](https://o-ring.tech/). |
| [Oura Ring 4, Silver](https://ouraring.com/store/rings/oura-ring-4/silver) | No | $349 | Cloud API and webhooks depend on app synchronization. They are not a continuous ring-to-host stream. [API](https://cloud.ouraring.com/v2/docs). |
| [Samsung Galaxy Ring](https://www.samsung.com/us/rings/galaxy-ring/buy/galaxy-ring-titanium-silver-sku-sm-q50xnzsaxar) | No | $399.99 | Samsung Health records are not a direct ring sensor stream. The reviewed Sensor SDK targets Galaxy Watch. [Sensor SDK](https://developer.samsung.com/health/sensor/overview.html). |
| [RingConn Gen 2](https://ringconn.com/products/ringconn-gen-2) | Unconfirmed | $299 | Published product documentation covers vendor-app monitoring, not an external continuous acquisition interface. |

## 3. Finger-Attached Sensors

| Rank | Product | Live streaming | Hardware price | EDA | Raw PPG | Pulse waveform | HR | SpO2 | Perfusion index | Skin temperature |
|---:|---|:---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | [CMI Health PC-60FW (Viatom)](https://www.viatomtech.com/pulseoximeter-with-alarm) | Yes | [$19.99](https://www.cmihealth.com/products/fingertip-pulse-oximeter) | No | No | Yes | Yes | Yes | Yes | No |
| 2 | [Lepu Creative PC-66B with adult finger probe](https://www.lepucreative.com/products/lepu-medical-grade-pulse-oximeter-fingertip-pulse-oximeter-blood-oxygen-meter-handheld-overnight-oximeter-digital-portable-pc66b-measure-spo2-pulse-rate-for-neonate-kid-adult-android-iphone-with-wireless-bluetooth-connection) | Yes | $169.99 | No | No | Yes | Yes | Yes | Yes | No |
| 3 | [BerryMed BM1000-GBT](https://www.amperordirect.com/pc/berrymed-finger-oximeter-BM1000-GBT.html) | Conditional | $32.99 | No | No | Yes | Yes | Yes | Unconfirmed | No |
| EDA add-on | [Mindfield eSense Skin Response](https://mindfield-shop.com/en/product/esense-skin-response/) | Yes | EUR 149-159 | Yes | No | No | No | No | No | No |

| Product | Live interface and signal quality | Patient use and recommendation |
|---|---|---|
| PC-60FW | Android BLE SDK. HR, SpO2 and perfusion index: 1 Hz. Processed pulse waveform: 50 Hz. Probe-off and pulse-search flags included. Continuous mode required. [SDK](https://github.com/viatom-develop/LepuDemo#pc-60fw-bluetoothmodel_pc60fw). | Preferred one-piece clip for simple attachment and an explicit acquisition interface. [Device and modes](https://www.cmihealth.com/products/fingertip-pulse-oximeter). |
| PC-66B | Model-specific SDK. Parameters: 1 Hz. Processed waveform: 50 Hz. [SDK](https://github.com/viatom-develop/LepuDemo#pc-66b-bluetoothmodel_pc66b). | Interchangeable-probe alternative. Cable and handheld unit add handling. Adult-probe package is listed sold out. Neonatal probes cost extra. [Package](https://www.lepucreative.com/products/lepu-medical-grade-pulse-oximeter-fingertip-pulse-oximeter-blood-oxygen-meter-handheld-overnight-oximeter-digital-portable-pc66b-measure-spo2-pulse-rate-for-neonate-kid-adult-android-iphone-with-wireless-bluetooth-connection). |
| BM1000-GBT | Original BCI BLE packet protocol with a community host implementation. Compatibility depends on the original packet format and device firmware. [Protocol](https://github.com/zh2x/BCI_Protocol). [Host library](https://docs.circuitpython.org/projects/ble_berrymed_pulse_oximeter/en/latest/api.html). | Backup one-piece clip. The comparison covers BM1000-GBT, not other BM1000 variants. |
| eSense Skin Response | Skin conductance: 5 Hz. Phone OSC/LSL or Windows C# SDK. Windows SDK: EUR 399. Sensor plus SDK: EUR 548-558. Phones without a compatible headset input require an audio adapter. [SDK](https://mindfield-shop.com/en/product/esense-sdk-for-creating-your-own-windows-software-for-the-esense-sensors/). [Manual](https://help.mindfield.de/en/skin-response-manual). | Recommended EDA add-on. Two finger electrodes and a cable add contact and cleaning. The [device study](https://doi.org/10.1002/da.22610) concerns PTSD reactivity, not presyncope. |
| Category limits | Pulse waveforms are processed optical signals, not raw LED channels. The listed integrations exclude BP. | Poor perfusion, motion, pigmentation and nail products affect pulse oximetry. An [11-device study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10943300/) demonstrated device-dependent performance, not this shortlist's accuracy ranking. |

| Other screened products | Live streaming | Hardware price | Assessment |
|---|:---:|---:|---|
| [Nonin 3230](https://www.nonin.com/products/3230/) | Conditional | [$159](https://discountcardiology.com/products/nonin-3230-bluetooth-pulse-oximeter) | Spot-check device with proprietary 1 Hz BLE output. Shuts down after two minutes without an adequate reading. This behavior conflicts with the continuous-monitoring requirement. [Manual](https://www.nonin.com/wp-content/uploads/Operators-Manual-3230-Wireless-Oximeter-ce_ENG-FRE.pdf). [Accuracy specifications](https://www.nonin.com/support/3230/). |
| [Masimo MightySat Rx 9909](https://www.masimo.com/products/wearables/mightysatrx/) | Unconfirmed | [$299](https://www.turnermedical.com/Masimo_MightySat_Bluetooth_IPhone_Pulse_Oximeter_p/masimo_mightysat_9909.htm) | Published HR, SpO2, PI, PVi and respiratory-rate features do not establish an external continuous interface. Vendor-app streaming does not meet the integration requirement. |
| [Thought Technology eVu TPS T4500](https://thoughttechnology.com/evu-tps-package-t4500/) | Unconfirmed | $450 | One finger sensor combines skin conductance, temperature and pulse-derived HRV. Documented Bluetooth access is to the vendor app, not our software. |
| [Vernier Qubit GSR Sensor](https://www.vernier.com/product/qubit-gsr-sensor/) | Unconfirmed | $1,439 | Sensor alone exceeds the budget. Separate acquisition interface required. Sold to educational institutions. |

## 4. Consumer RGB Cameras

For Sections 4-7, nominal face pixels assume a 150 mm-wide face within a 1 m-wide image, not a 1 m camera distance; values are rounded geometric counts, not measured usable skin pixels.

| Rank | Product | Live streaming | Hardware price | Capture modes | UVC | Manual exposure | Manual white balance | Fixed-focus lens | External trigger |
|---:|---|:---:|---:|---|:---:|:---:|:---:|:---:|:---:|
| 1 | [Elgato Facecam 4K](https://www.elgato.com/us/en/p/facecam-4k) | Yes | $199.99 | Uncompressed 4K at 30 fps or 1080p at 60 fps. 4K60 uses MJPEG. | Yes | Yes | Yes | Yes | No |
| 2 | [Elgato Facecam MK.2](https://www.elgato.com/us/en/p/facecam-mk2) | Yes | $139.99 | Uncompressed 1080p at 60 fps. | Yes | Yes | Yes | Yes | No |
| 3 | [Logitech MX Brio](https://www.logitech.com/en-us/shop/p/mx-brio-4k-webcam.960-001558) | Yes | $199.99 | 4K at 30 fps or 1080p at 60 fps. | Yes | Yes | Yes | No | No |

| Product | Coverage and nominal face pixels | Live capture and timing | Camera plus one light | Recommendation |
|---|---|---|---:|---|
| Facecam 4K | 576 px at 4K. Fixed focus: 30-120 cm at 4K. Vendor FOV: 90 deg. [Specifications](https://www.elgato.com/us/en/p/facecam-4k). | UVC. Selected mode: uncompressed 4K30 NV12. Manual exposure and white balance. No external trigger. | $269.98 | Preferred consumer camera. The $60 premium over MK.2 provides more pixels for facial crops at the same framing. |
| Facecam MK.2 | 288 px at 1080p. Fixed focus: 30-120 cm. Vendor FOV: 84 deg. [Specifications](https://help.elgato.com/hc/en-us/articles/24162700661517-Elgato-Facecam-MK-2-Technical-Specifications). | UVC. Uncompressed 1080p60. Manual exposure and white balance. No external trigger. | $209.98 | Preferred budget alternative. Higher frame rate in the selected mode, but less spatial detail than 4K30. |
| MX Brio | 576 px at 4K. Diagonal FOV: 90 deg. Adjustable focus. [Controls](https://www.logitech.com/en-us/shop/p/mx-brio-4k-webcam.960-001558). | UVC. 4K30. Manual exposure, white balance and focus. No external trigger. | $269.98 | Same price as Facecam 4K. Lower priority because Elgato documents uncompressed 4K30 explicitly. Disable automatic enhancement and focus changes during capture. |
| Illumination and cost | One [Logitech Litra Glow](https://www.logitech.com/en-us/shop/p/litra-glow): $69.99. Diffuse visible-light source. | Selected setup uses fixed exposure, white balance and light output. UVC does not establish hardware synchronization across devices. | Included above | Subtotals include the camera and one light only. Robot-specific mounts and the host are separate. A shared light is counted once per station. |
| Chair coverage | Frame the face in upright and reclined positions. Preserve visible facial skin during head turns. | Record frame timing and dropped frames. Digital zoom does not increase captured skin detail. | Not a priced assembly | Start with one RGB camera and stable illumination. Add thermal sensing as a separate modality; add depth for geometry and movement analysis. |
| Evidence | The [venipuncture study](https://doi.org/10.1371/journal.pone.0314038) used Nikon AW130, not these webcams. | Compression and illumination affect rPPG. [Compression](https://doi.org/10.1016/j.bspc.2024.107445). [Low light](https://www.nature.com/articles/s41746-025-02192-y). | Not applicable | No patient attachment. Video requires model-based physiological estimation. More pixels do not establish better presyncope detection. |

| Other screened products | Live streaming | Hardware price | Assessment |
|---|:---:|---:|---|
| [Razer Kiyo V2](https://www.razer.com/streaming-cameras/razer-kiyo-v2) | Yes | $149.99 | 4K30 with Camo controls. Elgato provides a more explicit uncompressed acquisition specification for this integration. |
| [OBSBOT Tiny 2 Lite](https://www.obsbot.com/store/products/tiny-2-lite) | Yes | $159 | 4K30 and 1080p60. PTZ tracking adds moving mechanisms that are unnecessary for the selected fixed-chair installation. |
| [Elgato Facecam Pro](https://www.elgato.com/us/en/p/facecam-pro) | Yes | [$299.99 published MSRP](https://www.elgato.com/se/en/explorer/products/camera/difference-between-facecam-4k-and-facecam-pro/) | Official page states no longer available. Uncompressed output is limited to 1080p60. [Modes](https://help.elgato.com/hc/en-us/articles/10290429808653-Elgato-Facecam-Pro-Technical-Specifications). |
| [Razer Kiyo Pro Ultra](https://www.razer.com/streaming-cameras/razer-kiyo-pro-ultra) | Yes | [$299.99 published MSRP](https://www.razer.com/newsroom/company-news/razer-pushes-the-boundaries-of-gaming-innovation-with-exciting-announcements-at-ces-2023) | This review has no current U.S. purchase offer. Historical MSRP and published manual controls are insufficient for the procurement shortlist. [Controls](https://dl.razerzone.com/master-guides/RazerSynapse3/KIYOPROULTRA-00003592-en.pdf). |

## 5. Machine-Vision RGB Cameras

| Rank | Product | Live streaming | Camera price | Capture | Global shutter | Raw Bayer | External trigger | SDK |
|---:|---|:---:|---:|---|:---:|:---:|:---:|---|
| 1 | [Basler ace 2 a2A1920-160ucBAS](https://www.baslerweb.com/en-us/shop/a2a1920-160ucbas/) | Yes | $389 | 1920 x 1200 at up to 160 fps. 8-, 10- and 12-bit output. | Yes | Yes | Yes | pylon |
| 2 | [Teledyne FLIR Firefly S FFY-U3-16S2C-S](https://www.teledynevisionsolutions.com/products/firefly-s/?model=FFY-U3-16S2C-S) | Yes | $234 | 1440 x 1080 at 60 fps. 8-bit Bayer output. 10-bit ADC. | Yes | Yes | Yes | Spinnaker |
| 3 | [The Imaging Source DFK 33UX273](https://www.theimagingsource.com/en-us/product/industrial/33u/dfk33ux273/) | Yes | [$459](https://www.oemcameras.com/products/dfk33ux273-htm) | 1440 x 1080 at up to 238 fps. 12-bit packed Bayer supported. | Yes | Yes | Yes | IC Imaging Control / tiscamera |
| 4 | [Teledyne FLIR Blackfly S BFS-U3-13Y3C-C](https://www.teledynevisionsolutions.com/products/blackfly-s-usb3/?model=BFS-U3-13Y3C-C&vertical=machine%20vision&segment=iis) | Yes | $480 | 1280 x 1024. Bayer: 170 fps. BGR8: 90 fps. 10-bit ADC. | Yes | Yes | Yes | Spinnaker |

| Product | Coverage and nominal face pixels | Live capture and timing | Priced reference assembly | Recommendation |
|---|---|---|---:|---|
| Basler ace 2 | 288 px at 1920-wide output. C-mount optics set distance and FOV. | USB3 Vision, raw Bayer and hardware trigger. 8-, 10- and 12-bit output. [Documentation](https://docs.baslerweb.com/a2a1920-160ucbas). | $594.99 | Preferred industrial camera. Higher spatial resolution and output bit depth than Firefly. |
| Firefly S | 216 px at 1440-wide output. M12 optics set distance and FOV. Compact 20 g body. | USB3 Vision and hardware trigger. 8-bit Bayer despite the 10-bit ADC. [Pixel formats](https://softwareservices.flir.com/FFY-U3-16S2/latest/Model/spec.html). | $321.99-323.99 | Preferred budget industrial alternative. Second place reflects price and size, not second-best detector accuracy. |
| DFK 33UX273 | 216 px at 1440-wide output. C/CS optics set distance and FOV. | USB3 Vision, hardware trigger and 12-bit packed Bayer. Public Windows/Linux tools. [Specifications](https://www.theimagingsource.com/en-us/product/industrial/33u/dfk33ux273/). | $664.99 | High-speed alternative. Its peak frame rate does not offset Basler's lower price and higher spatial resolution for this shortlist. |
| Blackfly S | 192 px at 1280-wide output. C-mount optics set distance and FOV. | USB3 Vision and hardware trigger. The 12-bit container retains a 10-bit ADC source. [Specifications](https://softwareservices.flir.com/BFS-U3-13Y3/latest/Model/spec.html). | $685.99 | Higher-frame-rate U.S.-brand alternative to Firefly. Highest reference cost and fewest pixels in this industrial shortlist. |
| Reference optics | C-mount: [Basler C23-0824-5M, 8 mm](https://www.baslerweb.com/en-us/shop/basler-lens-c23-0824-5m-f8mm/), $136. M12: [Spinel 6 mm lens](https://www.spinelelectronics.com/product/m12-6ir5mp/), $18-20. | These are optical price references, not validated camera-lens assemblies. Working distance, FOV and mechanical clearance determine the installed lens. | Camera, reference lens and one [$69.99 light](https://www.logitech.com/en-us/shop/p/litra-glow) | Cables, mount adapters, protective housing and the host are separate. Spinel lists -19.6% TV distortion; its low price does not establish imaging quality. |
| Chair integration | Optical framing must cover upright and reclined faces. Keep hardware outside the patient's movement path. | Hardware triggers align exposures; independent device clocks still require alignment. [Capture study](https://www.nature.com/articles/s43856-024-00519-6). | Not a priced assembly | No patient attachment. Rank reflects capture control and cost, not measured rPPG superiority. |

| Other screened products | Live streaming | Camera price | Assessment |
|---|:---:|---:|---|
| [Allied Vision Alvium 1800 U-158c, S-mount with right-angle USB](https://www.alliedvision.com/en/products/area-scan-cameras/alvium/alvium-u/view/1128) | Yes | [$426](https://www.edmundoptics.com/p/allied-vision-alvium-1800-u-158c-129-16mp-s-mount-right-angle-usb-31-color-camera/42984/) | The priced retailer configuration lists 150 fps. Manufacturer documentation lists 257 fps. Conflicting configuration specifications exclude this offer from the shortlist. |

## 6. NIR and Depth Cameras

| Rank | Product | Live streaming | Hardware price | Capture | RGB | Depth | Active IR | Camera IMU | Depth hardware sync | SDK |
|---:|---|:---:|---:|---|:---:|:---:|:---:|:---:|:---:|---|
| 1 | [RealSense D435i](https://store.realsenseai.com/buy-realsense-depth-camera-d435i.html) | Yes | $334 | RGB: 1080p30. Depth: up to 1280 x 720 or 90 fps, mode-dependent. | Yes | Yes | Yes | Yes | Yes | librealsense |
| 2 | [Orbbec Gemini 2](https://store.orbbec.com/products/gemini-2) | Yes | $234 | RGB: 1080p30. Depth: 1280 x 800 at 30 fps. | Yes | Yes | Yes | Yes | Yes | Orbbec SDK |
| 3 | [RealSense D436](https://store.realsenseai.com/buy-realsense-depth-camera-d436.html) | Yes | $354 | RGB: 1280 x 800 at up to 60 fps. Global RGB and depth shutters. | Yes | Yes | Yes | Yes | Yes | librealsense |
| 4 | [Luxonis OAK-D Pro](https://shop.luxonis.com/products/oak-d-pro) | Yes | $429 | 12 MP RGB sensor with active stereo depth. Output depends on selected mode. | Yes | Yes | Yes | Yes | Conditional | DepthAI |
| 5 | [RealSense D455](https://store.realsenseai.com/buy-realsense-depth-camera-d455.html) | Yes | $419 | RGB: 1280 x 800 at 30 fps. Global RGB and depth shutters. | Yes | Yes | Yes | Yes | Yes | librealsense |

| Product | Coverage and nominal face pixels | Live capture and timing | Camera plus one light | Recommendation |
|---|---|---|---:|---|
| D435i | 288 RGB px. Ideal depth range: 0.3-3 m. RGB FOV: 69 x 42 deg. [Device](https://store.realsenseai.com/buy-realsense-depth-camera-d435i.html). | Direct SDK frames and timestamps. Rolling-shutter RGB. External depth synchronization excludes RGB. [Synchronization](https://dev.realsenseai.com/docs/multiple-depth-cameras-configuration/). | $403.99 | Preferred current acquisition candidate. U.S.-based supplier. Store lists stock, cable and tripod. |
| Gemini 2 | 288 RGB px. Ideal depth range: 0.2-5 m. [Specifications](https://store.orbbec.com/products/gemini-2). | Direct RGB and depth frames through Orbbec SDK. | $303.99 | Preferred budget depth alternative. Costs $100 less than D435i and has a shorter minimum ideal depth distance. |
| D436 | 192 RGB px. Ideal depth range: 0.3-3 m. RGB FOV: 90 x 65 deg. [Specifications](https://store.realsenseai.com/buy-realsense-depth-camera-d436.html). | SDK frames. Global-shutter RGB synchronized with depth sensors. [Architecture](https://www.realsenseai.com/products/d436/). | $423.99 | Preferred synchronized-RGB upgrade candidate. Costs $20 more than D435i, with fewer RGB pixels. Store lists out of stock and backorder; supply places it third. |
| OAK-D Pro | Face pixels depend on RGB output mode. Standard 12 MP RGB option: 66 deg horizontal FOV. [Hardware](https://docs.luxonis.com/hardware/products/OAK-D%20Pro). | DepthAI images and onboard inference. External hardware synchronization requires access to sync signals and wiring. | $498.99 | Preferred alternative when onboard inference is required. Higher price is not justified by sensor resolution alone. |
| D455 | 192 RGB px. Ideal depth range: 0.6-6 m. RGB FOV: 90 x 65 deg. [Specifications](https://store.realsenseai.com/buy-realsense-depth-camera-d455.html). | SDK access with global-shutter RGB and depth. | $488.99 | Lower priority for a close chair installation. Larger body, higher price than D436 and no stock listed. |
| Integration and cost | Keep face and torso visible throughout chair recline. Depth range is not a guarantee of RGB focus or facial detail. | Camera IMU measures camera motion, not patient motion. Depth supports geometry, not direct BP, EDA or oxygenation. [Specifications](https://www.realsenseai.com/compare-all-cameras/). | Includes one [$69.99 light](https://www.logitech.com/en-us/shop/p/litra-glow) for RGB | Count shared illumination once. Robot mounts and host are separate. RealSense listed prices exclude the store's additional [tariff surcharge](https://store.realsenseai.com/buy-realsense-depth-camera-d436.html). |

| Other screened products | Live streaming | Hardware price | Assessment |
|---|:---:|---:|---|
| [Stereolabs ZED 2i, 2.1 mm without polarizer](https://www.stereolabs.com/store/products/zed-2i) | Yes | $499 | Live SDK with passive stereo, no IR projector and a larger camera body. ZED depth processing requires a supported host. [SDK requirements](https://docs.stereolabs.com/docs/development/zed-sdk/linux). |
| [RealSense D405](https://store.realsenseai.com/buy-realsense-depth-camera-d405.html) | Yes | $272 | Ideal range: 7-50 cm. This is a close-up hand/object camera, not the selected face-and-torso camera. Store lists no stock. |

## 7. Compact Thermal Cameras and Modules

| Rank | Product | Live streaming | Hardware price | Resolution | Frame rate | Per-pixel temperature | NETD | Stated temperature accuracy |
|---:|---|:---:|---|---|---|:---:|---|---|
| 1 | [TOPDON TC001, Android/USB-C](https://www.topdon.us/products/tc001) | Conditional | $299 | 256 x 192 | 25 Hz | Conditional | Under 40 mK | Error within 2 deg C or 2%. Vendor conditions apply. |
| 2 | [FLIR Lepton 3.5](https://www.oem.flir.com/en-150/developer/lepton-family/) with [PureThermal 3](https://groupgets.com/products/purethermal-3) | Yes | $291.99 total. Core: $172. Board: $119.99. | 160 x 120 | 8.7 Hz | Yes | Under 50 mK | Typical error within 5 deg C or 5%, whichever is greater, in high-gain mode. |
| 3 | [Seek Mosaic S314SPX starter kit](https://shop.thermal.com/Mosaic-Core-Starter-Kit-320x240-57HFOV-FF) | Conditional | $599 | 320 x 240 | Up to 27 Hz | Yes | 65 mK typical. Under 100 mK maximum. | Error within 5 deg C or 5%, whichever is greater. Vendor conditions apply. |

| Product | Coverage and nominal face pixels | Live capture and timing | Additional hardware | Recommendation |
|---|---|---|---|---|
| TC001 | 38 px. FOV: 56 x 42 deg. 256 x 192 at 25 Hz. [Specifications](https://www.topdon.us/products/tc001). | Conditional on hardware revision and SDK license. [Android SDK](https://github.com/TopdonTechnology/Thermal) and [community TC001/TS001 USB implementation](https://github.com/k20202/Topdon-TC00-TS001-Thermal-Viewer). Shutter calibration interrupts frames. [Behavior](https://service.topdon.com/portal/zh/kb/articles/portable-thermal-imager). | Enclosed device. Robot mount and compatible host are separate. | First thermal evaluation candidate, Conditional on live temperature-array access. Costs $7.01 more than Lepton assembly and offers higher resolution, higher frame rate and a lower stated NETD limit. |
| Lepton 3.5 with PureThermal 3 | 24 px. Horizontal FOV: 57 deg. 160 x 120 at 8.7 Hz. [Datasheet](https://groupgets-files.s3.amazonaws.com/lepton/Lepton%20Engineering%20Datasheet%20Rev%20400%20%28500-0659-00-09%29.pdf). | USB UVC, radiometric decoding and open [board firmware](https://groupgets.com/products/purethermal-3). | Enclosure, cable, robot mount and host are separate. | Preferred open-integration fallback. U.S. core and board suppliers. Lower cost, but fewer facial pixels and slower sampling than TC001. |
| Seek S314SPX | 48 px. FOV: 56 x 42 deg. 320 x 240 at up to 27 Hz. [Specifications](https://www.thermal.com/mosaic-core.html). | Vendor SDK access is required for USB temperature frames on Windows, Linux and Android. [Core and SDK](https://www.thermal.com/uploads/1/0/1/3/101388544/c314spx_s314spx_111721.pdf). | Enclosure, robot mount and host are separate. | Higher-resolution U.S.-brand alternative. Higher stated noise and price than Lepton. Kit store lists [no stock](https://shop.thermal.com/Mosaic-Core-Starter-Kit-320x240-57HFOV-FF). |
| Measurement limits | Narrower framing provides more facial pixels but reduces head-motion coverage. Glasses and masks obscure facial regions. | Acquire radiometric arrays, not a rendered color palette. Exclude calibration-interrupted frames. NETD measures sensitivity, not absolute temperature accuracy. | No visible light is required for thermal capture. | Use stable mounting and account for ambient conditions and thermal stabilization. Thermal is a supplement to RGB, not a replacement. |
| Evidence | A [193-donor study](https://www.nature.com/articles/s41598-023-36207-z) used FLIR E95: 464 x 348, 30 Hz, at 1 m. | Pre-donation facial videos predicted later self-reported vasovagal-reaction symptom scores. Non-frontal frames were removed. | Not a shortlisted device | The study supports thermal feature research, not continuous moving-patient detection or this product ranking. |

| Other screened products | Live streaming | Hardware price | Assessment |
|---|:---:|---:|---|
| [Waveshare Thermal-45 USB Camera (C), SKU 31714](https://www.waveshare.com/thermal-45-usb-camera-c.htm?sku=31714) | Unconfirmed | $119.99 | Reviewed documentation lacks an exact SKU-level USB temperature-array specification and published NETD. [Documentation](https://www.waveshare.com/wiki/Thermal_USB_Camera_(C)). |
| [Adafruit MLX90640, 55-degree breakout](https://www.adafruit.com/product/4407) | Yes | $74.95 | Live I2C temperature arrays. At 32 x 24 pixels, it has the lowest spatial resolution in this thermal comparison. Controller or I2C host and enclosure are separate. |
| [FLIR ONE Pro](https://www.flir.com/products/flir-one-pro/) | No | [$399.99 published MSRP](https://www.flir.com/news-center/press-releases/flir-systems-announces-availability-of-third-generation-flir-one-thermal-imaging-cameras-for-smartphones-and-tablets/) | Mobile SDK terms prohibit medical/health applications. This restriction belongs to the Mobile SDK, not the separate Lepton/PureThermal interface. [SDK restriction](https://flir.custhelp.com/app/answers/detail/a_id/1717/~/flir-mobile-sdk---developer-program-frequently-asked-questions). |
