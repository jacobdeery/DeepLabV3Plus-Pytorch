def calc_ece(pred, gt, unc):
    return 0

def get_uncertainty_msp(logits, temp=1):
    return 1 - (logits / temp).softmax(dim=1).max(dim=1).values
