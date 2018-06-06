import pandas as pd
import numpy as np
from cvutils.measure import compare_dice, compare_jaccard


def analyze_prediction(prediction, analysis_fns):
    return {k: fn(prediction) for k, fn in analysis_fns.items()}


def get_object_counter_fn():
    def fn(p):
        return dict(
            true=pd.Series(p.true_class_names).value_counts(),
            pred=pd.Series(p.pred_class_names).value_counts()
        )

    return fn


def mask_agg_areas(masks):
    assert masks.ndim == 3
    return pd.Series([masks[..., i].sum() for i in range(masks.shape[-1])])


def mask_agg_union_area(masks):
    assert masks.ndim == 3
    if masks.shape[-1] == 0:
        return pd.Series([])
    else:
        return pd.Series([masks.max(axis=-1).sum()])


def get_class_mask_agg_fn(class_name, mask_agg_fn=mask_agg_areas):
    def fn(p):
        return dict(
            true=mask_agg_fn(p.true_masks[..., p.true_class_names == class_name]),
            pred=mask_agg_fn(p.pred_masks[..., p.pred_class_names == class_name])
        )

    return fn


DEFAULT_PERCENTILES = [.01, .05, .1, .25, .5, .75, .9, .95, .99]


def get_class_stats_fn(class_name, mask_agg_fn=mask_agg_areas, percentiles=DEFAULT_PERCENTILES):
    agg_fn = get_class_mask_agg_fn(class_name, mask_agg_fn=mask_agg_fn)

    def fn(p):
        return {k: v.describe(percentiles=percentiles) for k, v in agg_fn(p).items()}

    return fn


def mask_select_single(masks):
    assert masks.ndim == 3
    if masks.shape[-1] == 0:
        return None, 'Empty'
    if masks.shape[-1] > 1:
        return None, 'Multiple'
    return masks[..., 0], 'Valid'


def mask_select_union(masks):
    assert masks.ndim == 3
    if masks.shape[-1] == 0:
        return None, 'Empty'
    return masks.max(axis=-1), 'Valid'


DEFAULT_SCORE_METRICS = {'jaccard': compare_jaccard, 'dice': compare_dice}


def get_scoring_fn(pred_class, true_class=None,
                   pred_mask_select_fn=mask_select_single,
                   true_mask_select_fn=mask_select_single,
                   metrics=DEFAULT_SCORE_METRICS):
    if true_class is None:
        true_class = pred_class

    def fn(p):
        pred_mask, pred_status = pred_mask_select_fn(p.pred_masks[..., p.pred_class_names == pred_class])
        true_mask, true_status = true_mask_select_fn(p.true_masks[..., p.true_class_names == true_class])
        res = dict(pred_status=pred_status, true_status=true_status)
        for k, score_fn in metrics.items():
            if pred_mask is not None and true_mask is not None:
                res[k] = score_fn(true_mask, pred_mask)
            else:
                res[k] = np.nan
        return res

    return fn


def get_default_analysis_fns():
    return dict(
        image_id=lambda p: p.image_id,
        image_info=lambda p: p.image_info,

        counts=get_object_counter_fn(),

        stats_chamber=get_class_stats_fn('Chamber'),
        stats_cellclump=get_class_stats_fn('CellClump'),
        stats_cell=get_class_stats_fn('Cell'),
        stats_cellunion=get_class_stats_fn('Cell', mask_agg_fn=mask_agg_union_area),

        scores_chamber=get_scoring_fn('Chamber'),
        scores_stnum=get_scoring_fn('StNum'),
        scores_aptnum=get_scoring_fn('AptNum'),
        scores_cellclump=get_scoring_fn('CellClump'),
        scores_cellunion=get_scoring_fn('Cell', 'CellClump', pred_mask_select_fn=mask_select_union),
        scores_cell=get_scoring_fn('Cell', 'Cell', pred_mask_select_fn=mask_select_union,
                                   true_mask_select_fn=mask_select_union)

    )
