import numpy as np 

CLASS_LABELS = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
UNKNOWN_ID = -100
N_CLASSES = len(CLASS_LABELS)

def confusion_matrix(pred_ids, gt_ids):
    
    assert pred_ids.shape == gt_ids.shape, (pred_ids.shape, gt_ids.shape)
    idxs = (gt_ids >= 0)

    return np.bincount(pred_ids[idxs] * 20 + gt_ids[idxs], minlength=400).reshape((20, 20)).astype(np.ulonglong)

def get_iou(label_id, confusion):
    
    # true positives
    tp = np.longlong(confusion[label_id, label_id])
    # false positives
    fp = np.longlong(confusion[label_id, :].sum()) - tp
    # false negatives
    fn = np.longlong(confusion[:, label_id].sum()) - tp

    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')
    acc = float(tp)/(tp + fn)
    return (float(tp) / denom, tp, denom, acc)


