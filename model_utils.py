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


def build_data_reference(df_raw, dye_cols, known_dyes):
    df_ref = df_raw.copy()
    for dc in dye_cols:
        df_ref[dc] = df_ref[dc].apply(clean_dye_id)

    X_ref, _ = transform_bag_of_dyes(df_ref, dye_cols, known_dyes=known_dyes)

    num_cols = ['標準樣L', '標準樣a', '標準樣b', 'DPF', 'Total_Conc', 'Log_Total_Conc']
    numeric_stats = {}
    for col in num_cols:
        series = pd.to_numeric(X_ref[col], errors='coerce').dropna()
        numeric_stats[col] = {
            'min': float(series.min()),
            'max': float(series.max()),
            'mean': float(series.mean()),
            'std': float(series.std(ddof=0) if series.std(ddof=0) > 1e-8 else 1.0),
        }

    combo_set = set()
    for _, row in df_ref[dye_cols].iterrows():
        combo = sorted([clean_dye_id(row[c]) for c in dye_cols if clean_dye_id(row[c]) != '無'])
        combo_set.add('+'.join(combo) if combo else '僅基礎藥劑(無染料)')

    return {
        'numeric_stats': numeric_stats,
        'seen_OP否': sorted(df_ref['OP否'].astype(str).unique().tolist()),
        'seen_色系名稱': sorted(df_ref['色系名稱'].astype(str).unique().tolist()),
        'seen_色系編號': sorted(df_ref['色系編號'].astype(str).unique().tolist()),
        'seen_combos': sorted(combo_set),
    }


def assess_input_data_confidence(df_input, dye_cols, known_dyes, data_reference):
    X_input, _ = transform_bag_of_dyes(df_input.copy(), dye_cols, known_dyes=known_dyes)
    row_raw = df_input.iloc[0]
    row_feat = X_input.iloc[0]

    op_seen = str(row_raw['OP否']) in set(data_reference.get('seen_OP否', []))
    name_seen = str(row_raw['色系名稱']) in set(data_reference.get('seen_色系名稱', []))
    id_seen = str(row_raw['色系編號']) in set(data_reference.get('seen_色系編號', []))
    cat_score = np.mean([op_seen, name_seen, id_seen])

    num_scores = []
    num_details = []
    for col, stats in data_reference.get('numeric_stats', {}).items():
        val = float(row_feat[col])
        v_min, v_max, v_mean, v_std = stats['min'], stats['max'], stats['mean'], stats['std']
        if v_min <= val <= v_max:
            score = 1.0
        else:
            distance = min(abs(val - v_min), abs(val - v_max))
            score = float(np.exp(-distance / max(v_std, 1e-8)))
        num_scores.append(score)
        num_details.append({'欄位': col, '輸入值': val, '訓練範圍': f"[{v_min:.3f}, {v_max:.3f}]", '子分數': score})
    num_score = float(np.mean(num_scores)) if num_scores else 0.0

    input_combo = sorted([clean_dye_id(row_raw[c]) for c in dye_cols if clean_dye_id(row_raw[c]) != '無'])
    input_combo_key = '+'.join(input_combo) if input_combo else '僅基礎藥劑(無染料)'
    combo_seen = input_combo_key in set(data_reference.get('seen_combos', []))
    combo_score = 1.0 if combo_seen else 0.35

    total_score = 100 * (0.4 * cat_score + 0.35 * num_score + 0.25 * combo_score)
    if total_score >= 80:
        level = '高'
    elif total_score >= 60:
        level = '中'
    else:
        level = '低'

    return {
        'score': float(total_score),
        'level': level,
        'category': {
            'score': float(cat_score),
            'OP否是否出現在訓練資料': op_seen,
            '色系名稱是否出現在訓練資料': name_seen,
            '色系編號是否出現在訓練資料': id_seen,
        },
        'numeric': {
            'score': num_score,
            'details': num_details,
        },
        'combo': {
            'score': combo_score,
            'input_combo': input_combo_key,
            'seen_in_training': combo_seen,
        },
    }
