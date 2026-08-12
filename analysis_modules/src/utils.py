def get_missing_neurons(datasets):
    """
        Some postembryonic neurons are not in all datasets.
        HSN is only in adult
        PVW, PLN, are only in adult and dauer
        ALN, AQR, AVF, AVM, RMF, RMH, and SDQ are in L2, L3, adult and dauer 
    """
    adult_set = set(['adult-1','adult-2'])
    dauer_set = set(['dauer-1','dauer-2','dauer-3','dauer-daf2'])
    early_set = set(['L1-1','L1-2','L1-3','L1-4'])
    if all(ds in adult_set for ds in set(datasets)):
        return []
    if all(ds in adult_set.union(dauer_set) for ds in set(datasets)):
        return ['HSN']
    if set(datasets).intersection(early_set):
        return ['PVW','PLN','HSN','AVM','RMF','SDQ','RMH','AVF','AQR','ALN']
    else:
        return ['PVW','PLN','HSN']

def exclude_postemb_edgetable(edgetable, datasets):
    """
        Remove missing postemb neurons from edgetable
    """
    exclude_list = get_missing_neurons(datasets)
    filt_1 = edgetable.index.get_level_values(0).isin(exclude_list)
    filt_2 = edgetable.index.get_level_values(1).isin(exclude_list)
    return edgetable.loc[~filt_1 & ~filt_2]
