import numpy as np
import pandas as pd


def deltaE_CMC(lab_std, lab_smpl, l=2, c=1):
    L1, a1, b1 = lab_std
    L2, a2, b2 = lab_smpl
    dL, da, db = L2 - L1, a2 - a1, b2 - b1
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    dC = C2 - C1
    dH_sq = da**2 + db**2 - dC**2
    dH = np.sqrt(max(0, dH_sq))
    SL = 0.040975 * L1 / (1 + 0.01765 * L1) if L1 >= 16 else 0.511
    SC = (0.0638 * C1 / (1 + 0.0131 * C1)) + 0.638
    H1 = np.degrees(np.arctan2(b1, a1)) % 360
    F = np.sqrt(C1**4 / (C1**4 + 1900))
    if 164 <= H1 <= 345:
        T = 0.56 + abs(0.2 * np.cos(np.radians(H1 + 168)))
    else:
        T = 0.36 + abs(0.4 * np.cos(np.radians(H1 + 35)))
    SH = SC * (F * T + 1 - F)
    return np.sqrt((dL / (l * SL)) ** 2 + (dC / (c * SC)) ** 2 + (dH / SH) ** 2)


def clean_dye_id(val):
    s = str(val).strip()
    if s.endswith('.0'):
        s = s[:-2]
    if s.lower() in ['nan', 'none', '', 'null'] or pd.isna(val):
        s = '無'
    return s


def detect_dye_columns(df):
    return sorted(
        [c for c in df.columns if c.startswith('配方料號')],
        key=lambda x: int(x.replace('配方料號', '')),
    )


def transform_bag_of_dyes(df_clean, dye_cols, known_dyes=None):
    df_out = df_clean.copy()
    if known_dyes is None:
        raw_values = df_out[dye_cols].values.flatten()
        known_dyes = sorted(list(set([clean_dye_id(d) for d in raw_values if clean_dye_id(d) != '無'])))
    dye_f_cols = [f'Dye_{d}' for d in known_dyes]
    for dfc in dye_f_cols:
        df_out[dfc] = 0.0

    for d_col in dye_cols:
        n_col = d_col.replace('料號', '濃度')
        if n_col in df_out.columns:
            for d in known_dyes:
                mask = df_out[d_col] == d
                df_out.loc[mask, f'Dye_{d}'] += df_out[n_col][mask]

    df_out['Total_Conc'] = df_out[dye_f_cols].sum(axis=1)
    df_out['Log_Total_Conc'] = np.log1p(df_out['Total_Conc'])

    base_cols = ['標準樣L', '標準樣a', '標準樣b', '色系名稱', '色系編號', 'DPF', 'OP否', 'Total_Conc', 'Log_Total_Conc']
    return df_out[base_cols + dye_f_cols], known_dyes
