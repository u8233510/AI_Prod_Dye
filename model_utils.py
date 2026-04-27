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

    combo_keys = []
    for _, row in df_ref[dye_cols].iterrows():
        combo = sorted([clean_dye_id(row[c]) for c in dye_cols if clean_dye_id(row[c]) != '無'])
        combo_keys.append('+'.join(combo) if combo else '僅基礎藥劑(無染料)')

    combo_freq = pd.Series(combo_keys).value_counts().to_dict()
    cat_triplet_keys = (
        df_ref['OP否'].astype(str).str.strip()
        + '|'
        + df_ref['色系名稱'].astype(str).str.strip()
        + '|'
        + df_ref['色系編號'].astype(str).str.strip()
    )
    cat_triplet_freq = cat_triplet_keys.value_counts().to_dict()

    lab_array = df_ref[['標準樣L', '標準樣a', '標準樣b']].apply(pd.to_numeric, errors='coerce').dropna().values
    lab_bin_size = {'L': 2.0, 'a': 2.0, 'b': 2.0}
    lab_bins = (
        (
            (df_ref['標準樣L'] / lab_bin_size['L']).round().astype(int).astype(str)
            + '|'
            + (df_ref['標準樣a'] / lab_bin_size['a']).round().astype(int).astype(str)
            + '|'
            + (df_ref['標準樣b'] / lab_bin_size['b']).round().astype(int).astype(str)
        )
        .value_counts()
        .to_dict()
    )

    return {
        'total_rows': int(len(df_ref)),
        'numeric_stats': numeric_stats,
        'seen_OP否': sorted(df_ref['OP否'].astype(str).unique().tolist()),
        'seen_色系名稱': sorted(df_ref['色系名稱'].astype(str).unique().tolist()),
        'seen_色系編號': sorted(df_ref['色系編號'].astype(str).unique().tolist()),
        'seen_combos': sorted(combo_freq.keys()),
        'combo_freq': combo_freq,
        'cat_triplet_freq': cat_triplet_freq,
        'lab_points': lab_array.tolist(),
        'lab_bin_size': lab_bin_size,
        'lab_bins': lab_bins,
    }


def assess_input_data_confidence(df_input, dye_cols, known_dyes, data_reference):
    X_input, _ = transform_bag_of_dyes(df_input.copy(), dye_cols, known_dyes=known_dyes)
    row_raw = df_input.iloc[0]
    row_feat = X_input.iloc[0]

    total_rows = max(int(data_reference.get('total_rows', 0)), 1)

    op_val = str(row_raw['OP否']).strip()
    name_val = str(row_raw['色系名稱']).strip()
    id_val = str(row_raw['色系編號']).strip()
    op_seen = op_val in set(data_reference.get('seen_OP否', []))
    name_seen = name_val in set(data_reference.get('seen_色系名稱', []))
    id_seen = id_val in set(data_reference.get('seen_色系編號', []))

    cat_triplet_key = f'{op_val}|{name_val}|{id_val}'
    cat_triplet_count = int(data_reference.get('cat_triplet_freq', {}).get(cat_triplet_key, 0))
    cat_triplet_ratio = cat_triplet_count / total_rows
    cat_score = float(min(1.0, np.log1p(cat_triplet_count) / np.log1p(20)))

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
    combo_count = int(data_reference.get('combo_freq', {}).get(input_combo_key, 0))
    combo_seen = combo_count > 0
    combo_ratio = combo_count / total_rows
    combo_score = float(min(1.0, np.log1p(combo_count) / np.log1p(20)))

    lab_input = np.array([float(row_feat['標準樣L']), float(row_feat['標準樣a']), float(row_feat['標準樣b'])], dtype=float)
    lab_points = np.array(data_reference.get('lab_points', []), dtype=float)
    if lab_points.size > 0:
        distances = np.linalg.norm(lab_points - lab_input, axis=1)
        nearest_lab_distance = float(distances.min())
        cnt_r15 = int((distances <= 1.5).sum())
        cnt_r30 = int((distances <= 3.0).sum())
        cnt_r50 = int((distances <= 5.0).sum())
    else:
        nearest_lab_distance = float('inf')
        cnt_r15 = cnt_r30 = cnt_r50 = 0

    lab_bin_size = data_reference.get('lab_bin_size', {'L': 2.0, 'a': 2.0, 'b': 2.0})
    lab_bin_key = (
        f"{int(round(lab_input[0] / max(float(lab_bin_size.get('L', 2.0)), 1e-8)))}|"
        f"{int(round(lab_input[1] / max(float(lab_bin_size.get('a', 2.0)), 1e-8)))}|"
        f"{int(round(lab_input[2] / max(float(lab_bin_size.get('b', 2.0)), 1e-8)))}"
    )
    lab_bin_count = int(data_reference.get('lab_bins', {}).get(lab_bin_key, 0))
    lab_score = float(min(1.0, np.log1p(max(cnt_r30, lab_bin_count)) / np.log1p(25)))

    support_index = 100 * (0.25 * cat_score + 0.25 * num_score + 0.30 * combo_score + 0.20 * lab_score)
    estimated_correctness = float(min(95.0, max(15.0, 10 + 0.85 * support_index)))
    total_score = support_index
    if total_score >= 80:
        level = '高'
    elif total_score >= 60:
        level = '中'
    else:
        level = '低'

    return {
        'score': float(total_score),
        'level': level,
        'estimated_correctness': estimated_correctness,
        'category': {
            'score': float(cat_score),
            'OP否是否出現在訓練資料': op_seen,
            '色系名稱是否出現在訓練資料': name_seen,
            '色系編號是否出現在訓練資料': id_seen,
            '三欄位組合出現次數': cat_triplet_count,
            '三欄位組合出現比例': float(cat_triplet_ratio),
        },
        'numeric': {
            'score': num_score,
            'details': num_details,
        },
        'combo': {
            'score': combo_score,
            'input_combo': input_combo_key,
            'seen_in_training': combo_seen,
            '出現次數': combo_count,
            '出現比例': float(combo_ratio),
        },
        'lab_region': {
            'score': lab_score,
            'nearest_distance': nearest_lab_distance,
            'count_within_1_5': cnt_r15,
            'count_within_3_0': cnt_r30,
            'count_within_5_0': cnt_r50,
            'bin_count': lab_bin_count,
            'bin_key': lab_bin_key,
        },
    }
