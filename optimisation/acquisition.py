
def acquisition_score(pred, uncertainty, beta=1.0):
    # ✅ hard safety layer
    if pred is None:
        pred = 0.0
    if uncertainty is None:
        uncertainty = 1.0

    pred = float(max(0.0, min(1.0, pred)))
    uncertainty = float(max(0.0, uncertainty))

    return pred + beta * uncertainty
