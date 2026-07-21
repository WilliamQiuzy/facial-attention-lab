import synBurnGraftReconUrl from '../../../../facial_defect_synthesis/output/synthetic/burns/burns_healed_graft-reconstructed_middle-aged-black-male_ca2269.png?url'
import synHncCheekFreeflapUrl from '../../../../facial_defect_synthesis/output/synthetic/hn_cancer/hn_cancer_healed_cheek-freeflap-healed_middle-aged-white-female_529016.png?url'
import synHncCheekTumourUrl from '../../../../facial_defect_synthesis/output/synthetic/hn_cancer/hn_cancer_preop_cheek-tumour_middle-aged-middle-eastern-female_277781.png?url'
import synMohsNasalReconUrl from '../../../../facial_defect_synthesis/output/synthetic/mohs/mohs_healed_nasal-recon-faint_elderly-hispanic-female_29a140.png?url'
import synMohsSccCheekUrl from '../../../../facial_defect_synthesis/output/synthetic/mohs/mohs_preop_scc-cheek_middle-aged-black-female_a905e7.png?url'
import synNevusCheekUrl from '../../../../facial_defect_synthesis/output/synthetic/nevus/nevus_preop_cheek-patch_young-adult-white-female_3fa4d7.png?url'
import synRhinophymaSevereUrl from '../../../../facial_defect_synthesis/output/synthetic/rhinophyma/rhinophyma_preop_severe-bulbous_elderly-middle-eastern-female_073b01.png?url'
import synTraumaCheekScarUrl from '../../../../facial_defect_synthesis/output/synthetic/trauma/trauma_preop_old-cheek-laceration-scar_middle-aged-middle-eastern-male_49cfd8.png?url'
import synTraumaFractureReconUrl from '../../../../facial_defect_synthesis/output/synthetic/trauma/trauma_healed_fracture-recon-faint_young-adult-south-asian-male_fedb1f.png?url'
import synVascularPwsUrl from '../../../../facial_defect_synthesis/output/synthetic/vascular/vascular_preop_port-wine-stain_young-adult-white-female_ad3581.png?url'
import type { WorkbenchAssetId } from './workbenchAssetDefinitions'

export const workbenchAssetUrls = {
  'SYN-MOHS-SCC-CHEEK': synMohsSccCheekUrl,
  'SYN-MOHS-NASAL-RECON': synMohsNasalReconUrl,
  'SYN-HNC-CHEEK-TUMOUR': synHncCheekTumourUrl,
  'SYN-HNC-CHEEK-FREEFLAP': synHncCheekFreeflapUrl,
  'SYN-TRAUMA-CHEEK-SCAR': synTraumaCheekScarUrl,
  'SYN-TRAUMA-FRACTURE-RECON': synTraumaFractureReconUrl,
  'SYN-RHINOPHYMA-SEVERE': synRhinophymaSevereUrl,
  'SYN-BURN-GRAFT-RECON': synBurnGraftReconUrl,
  'SYN-VASCULAR-PWS': synVascularPwsUrl,
  'SYN-NEVUS-CHEEK': synNevusCheekUrl,
} as const satisfies Readonly<Record<WorkbenchAssetId, string>>
