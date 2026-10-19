"""Optional eval instrumentation: keep pre/post recognition scores aligned.

The upstream recognizer reorders candidates by pyramid level. Capture that
permutation before it normalizes grids in place. Original result values and
candidate order are preserved; extra keys are ignored by the official matcher.
"""
import numpy as np
from mmocr.models.textend2end.recog_heads.lra_pp_recog_head import LRANet_PP_RecogHead


def install():
    original = LRANet_PP_RecogHead.simple_test

    def traced(self, feature_maps, det_results, img_metas, rescale):
        boundaries = det_results["boundary_result"]
        before = []
        if len(boundaries):
            grids = det_results["grids_results"].view(len(boundaries), -1, 2)
            assigned = self.assign_grids_to_level(grids).detach().cpu().numpy()
            order = np.concatenate([np.flatnonzero(assigned == level) for level in range(len(feature_maps))])
            if len(order) != len(boundaries):
                raise RuntimeError("Score probe lost candidates during level assignment")
            before = np.asarray(boundaries)[order, -1].astype(float).tolist()
        result = original(self, feature_maps, det_results, img_metas, rescale)
        if len(before) != len(result["boundary_result"]):
            raise RuntimeError("Upstream recognizer changed candidate count; inspect score probe")
        result["diagnostic_detection_scores"] = before
        result["diagnostic_score_protocol"] = "pre-recognition scores; candidate order aligned by FPN level"
        return result

    LRANet_PP_RecogHead.simple_test = traced


install()
