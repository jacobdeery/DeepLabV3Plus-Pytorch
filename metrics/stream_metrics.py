import numpy as np
from sklearn.metrics import confusion_matrix

class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()      

class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

        self.num_bins = 10
        self.bounds = np.linspace(0, 1, self.num_bins + 1)
        self.ece_curve = np.zeros((3, self.num_bins))

    def update(self, label_trues, label_preds, uncertainties):
        for lt, lp, unc in zip(label_trues, label_preds, uncertainties):
            self.confusion_matrix += self._fast_hist( lt.flatten(), lp.flatten() )
            self.update_ece(lt, lp, unc)

    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k!="Class IoU":
                string += "%s: %f\n"%(k, v)
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def update_ece(self, label_true, label_pred, unc):
        mask = (label_true >= 0) & (label_true < self.n_classes)

        pixel_correct = label_pred[mask] == label_true[mask]
        conf = (1 - unc)[mask]

        for i in range(self.num_bins):
            lower = self.bounds[i]
            upper = self.bounds[i + 1]

            bin_idxs = (conf >= lower) & (conf < upper)
            num_in_bin = np.sum(bin_idxs)

            if num_in_bin == 0:
                continue

            mean_correct = np.mean(pixel_correct[bin_idxs])
            mean_conf = np.mean(conf[bin_idxs])

            old_px = self.ece_curve[0, i]
            old_corr = self.ece_curve[1, i]
            old_conf = self.ece_curve[2, i]

            new_px = old_px + num_in_bin
            new_corr = np.nan_to_num((old_corr * old_px + mean_correct * num_in_bin) / new_px)
            new_conf = np.nan_to_num((old_conf * old_px + mean_conf * num_in_bin) / new_px)

            self.ece_curve[:, i] = [new_px, new_corr, new_conf]


    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        ece = np.average(np.abs(self.ece_curve[1] - self.ece_curve[2]), weights=self.ece_curve[0])

        return {
                "Overall Acc": acc,
                "Mean Acc": acc_cls,
                "FreqW Acc": fwavacc,
                "Mean IoU": mean_iu,
                "Class IoU": cls_iu,
                "ECE": ece
            }
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

class AverageMeter(object):
    """Computes average values"""
    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()
    
    def reset(self, id):
        item = self.book.get(id, None)
        if item is not None:
            item[0] = 0
            item[1] = 0

    def update(self, id, val):
        record = self.book.get(id, None)
        if record is None:
            self.book[id] = [val, 1]
        else:
            record[0]+=val
            record[1]+=1

    def get_results(self, id):
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]
