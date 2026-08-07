from pathlib import Path
import numpy as np
import pandas as pd

def load_csv_files(data_dir):
    csv_files = {}
    data_dir = Path(data_dir)
    for filepath in sorted(data_dir.glob("*.csv")):
        csv_files[filepath.stem] = filepath
    return csv_files

def csv_to_multiindex(f):
    df = pd.DataFrame(pd.read_csv(f))
    df.columns.values[0] = 'index'
    df = df.melt(id_vars="index")

    replacements = {
        "PVWL_or_R_1": "PVWL",
        "PVWL_or_R_2": "PVWL",
        "PVWL_or_R_3": "PVWR",
        "RICRa": "RICR",
        "RICRp": "RICR"
    }

    df["variable"] = df["variable"].replace(replacements)
    df["index"] = df["index"].replace(replacements)

    df.set_index(["variable","index"], inplace = True)
    df.index = df.index.set_names(("pre","post"))

    return df

def process_multiple_csvs(csv_files):
    dfs = []
    for key, value in csv_files.items():
        df = csv_to_multiindex(value)
        df = df[df.value > 0]
        df = df.rename(columns={"value": key})
        dfs.append(df)

    G = pd.concat(dfs, axis=0).fillna(0)
    G = G.astype(int)
    G = G.groupby(level=[0,1]).sum()
    return G

# exclude non-neuron, non-muscle, postemb
exclude_list = [
    'GLRV','GLRD','GLRL/R','CAN','CEPsh','CEPshV','CEPshD',
    'GLRVL','GLRVR','GLRR','GLRL','GLRDL','GLRDR', 'GLR'
    'CANL','CANR','CEPshVL','CEPshVR','CEPshDL','CEPshDR'
    ]

postembryonic = [
    "PVWL", "PVWR", "PVW", 'HSNL', 'HSNR','HSN',
    "AQR", "ALNL", "ALNR", "ALN",
    "RMHL", "RMHR", "RMH", "RMFL", "RMFR", "RMF",
    "AVFL", "AVFR", "AVF", "AVM",
    "SDQL", "SDQR", "SDQ", "PLNL", "PLNR", "PLN"
]

# Configuration
DATA_DIR = "data/raw/connectomes"
EXCLUDE_POSTEMBRYONIC = False
contactome = False # log10 for contactome

if EXCLUDE_POSTEMBRYONIC:
    exclude_list.extend(postembryonic)

csv_files = load_csv_files(DATA_DIR)
G = process_multiple_csvs(csv_files)

exclusion_filt = (
    G.index.get_level_values('pre').isin(exclude_list) |
    G.index.get_level_values('post').isin(exclude_list)
)
G = G[~exclusion_filt]

# log10 + upper left triangle for contactome
if contactome:
    G = np.log10(G).mask(np.isinf(np.log10(G)), 0)
    G['adult_sem'] = G.pop("adult_sem")
    
    index_set = set()
    for i in (G.index.tolist()):
        index_set.add(tuple(sorted(i)))
    index_list = list(index_set)

    G = G.loc[index_list]

# list all overlap
if not contactome:
    filt_nondauer = G.iloc[:,4:8].astype(bool).sum(axis=1) == 4
    filt_dauer = G.iloc[:,-3:].astype(bool).sum(axis=1) == 3
    # print(G[filt_nondauer & filt_dauer])
    
# list dauer core /exclude nondauer core indices
shared = pd.Series(0, G[filt_nondauer & filt_dauer].index)
dauer_excl = pd.Series(1, G[filt_dauer & ~filt_nondauer].index)
nondauer_excl = pd.Series(-1,G[filt_nondauer & ~filt_dauer].index)
# put them into one Pandas Series
classification = pd.concat([shared, dauer_excl, nondauer_excl])

# output for use in matrix_generation.py
classification.to_csv('dauer_change_classification.csv')