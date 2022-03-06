import paddle
from paddle.nn import CrossEntropyLoss
def basic_seg_loss(preds, labels, ignore_index=255):
    criterion = CrossEntropyLoss(ignore_index=ignore_index, axis=1)
    labels = paddle.cast(labels, dtype='int64')
    loss = criterion(preds, labels)

    return loss
