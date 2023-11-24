def sensitivity(x: dict) -> float:
    return x['tp'] / (x['tp'] + x['fn'])

def specificity(x: dict) -> float:
    return x['tn'] / (x['tn'] + x['fp'])

def accuracy(x: dict) -> float:
    return (x['tp'] + x['tn'])/(x['fn']+x['tn']+x['fp']+x['tp'])

def f1score(x: dict) -> float:
    return (x['tp']*2)/(x['tp']*2 + x['fp'] + x['fn'])

def precision(x: dict) -> float:
    return x['tp'] / (x['tp'] + x['fp'])
