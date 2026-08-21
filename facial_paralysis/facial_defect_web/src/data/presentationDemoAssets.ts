import subjectAPostoperativeUrl from '../../../../facial_defect_synthesis/output/synthetic/presentation/mohs_postop_healing-cheek-incision_middle-aged-black-female.png?url'
import subjectBPostoperativeUrl from '../../../../facial_defect_synthesis/output/synthetic/presentation/nevus_postop_healing-cheek-incision_young-adult-white-female.png?url'
import subjectAPreoperativeUrl from '../../../../facial_defect_synthesis/output/synthetic/mohs/mohs_preop_scc-cheek_middle-aged-black-female_a905e7.png?url'
import subjectBPreoperativeUrl from '../../../../facial_defect_synthesis/output/synthetic/nevus/nevus_preop_cheek-patch_young-adult-white-female_3fa4d7.png?url'
import {
  presentationDemoManifest,
  type PresentationDemoAssetMetadata,
  type PresentationSubjectId,
  type PresentationTimepoint,
} from './presentationDemoManifest'

export {
  PRESENTATION_BOUNDARY,
  presentationSubjectIds,
  presentationSubjectOptions,
  type PresentationSubjectId,
  type PresentationTimepoint,
} from './presentationDemoManifest'

export type PresentationDemoAsset = PresentationDemoAssetMetadata & Readonly<{
  url: string
}>

export const presentationDemoAssets: Readonly<
  Record<
    PresentationSubjectId,
    Readonly<Record<PresentationTimepoint, PresentationDemoAsset>>
  >
> = Object.freeze({
  'subject-a': Object.freeze({
    preoperative: Object.freeze({
      ...presentationDemoManifest['subject-a'].preoperative,
      url: subjectAPreoperativeUrl,
    }),
    postoperative: Object.freeze({
      ...presentationDemoManifest['subject-a'].postoperative,
      url: subjectAPostoperativeUrl,
    }),
  }),
  'subject-b': Object.freeze({
    preoperative: Object.freeze({
      ...presentationDemoManifest['subject-b'].preoperative,
      url: subjectBPreoperativeUrl,
    }),
    postoperative: Object.freeze({
      ...presentationDemoManifest['subject-b'].postoperative,
      url: subjectBPostoperativeUrl,
    }),
  }),
})
