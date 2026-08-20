import postoperativeUrl from '../../../../facial_defect_synthesis/output/synthetic/presentation/mohs_postop_small-cheek-scar_middle-aged-black-female.png?url'
import preoperativeUrl from '../../../../facial_defect_synthesis/output/synthetic/mohs/mohs_preop_scc-cheek_middle-aged-black-female_a905e7.png?url'
import {
  presentationDemoManifest,
  type PresentationDemoAssetMetadata,
  type PresentationTimepoint,
} from './presentationDemoManifest'

export {
  PRESENTATION_BOUNDARY,
  type PresentationTimepoint,
} from './presentationDemoManifest'

export type PresentationDemoAsset = PresentationDemoAssetMetadata & Readonly<{
  url: string
}>

export const presentationDemoAssets: Readonly<
  Record<PresentationTimepoint, PresentationDemoAsset>
> = Object.freeze({
  preoperative: Object.freeze({
    ...presentationDemoManifest.preoperative,
    url: preoperativeUrl,
  }),
  postoperative: Object.freeze({
    ...presentationDemoManifest.postoperative,
    url: postoperativeUrl,
  }),
})
