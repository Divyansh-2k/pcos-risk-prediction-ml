import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def validate(df, features, name):
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"{name} missing: {missing}")
    return features


def scale(df, features):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(df[features]), columns=features)


def load_data(path):
    df = pd.read_csv(path)

    df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True)

    if 'Unnamed: 44' in df.columns:
        df = df.drop(columns=['Unnamed: 44'])

    df = df.dropna().reset_index(drop=True)

    print(f"Data Cleaned. Shape: {df.shape}")

    y = df['PCOS (Y/N)'].astype(int)

    cols = ['BMI', 'Cycle length(days)', 'Hb(g/dl)', 'Pulse rate(bpm)']
    for c in cols:
        df[c] = df[c].clip(df[c].quantile(0.01), df[c].quantile(0.99))

    df = pd.get_dummies(df, columns=['Blood Group', 'Cycle(R/I)'], prefix=['BG', 'Cycle'])

    # fix inplace warning
    for col in ['AMH(ng/mL)', 'II beta-HCG(mIU/mL)']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

    skewed = ['AMH(ng/mL)', 'I beta-HCG(mIU/mL)',
              'II beta-HCG(mIU/mL)', 'PRL(ng/mL)', 'TSH (mIU/L)']

    for col in skewed:
        df[col] = np.log1p(df[col])

    ovarian = [
        'Follicle No. (L)', 'Follicle No. (R)',
        'Avg. F size (L) (mm)', 'Avg. F size (R) (mm)',
        'Endometrium (mm)'
    ]

    hormonal = [
        'FSH(mIU/mL)', 'LH(mIU/mL)', 'FSH/LH',
        'AMH(ng/mL)', 'TSH (mIU/L)', 'PRL(ng/mL)',
        'PRG(ng/mL)', 'I beta-HCG(mIU/mL)',
        'II beta-HCG(mIU/mL)', 'Vit D3 (ng/mL)'
    ]

    clinical = [
        'Age (yrs)', 'Weight (Kg)', 'Height(Cm)', 'BMI',
        'Pulse rate(bpm)', 'RR (breaths/min)', 'Hb(g/dl)',
        'Cycle length(days)', 'Pregnant(Y/N)', 'No. of abortions',
        'Hip(inch)', 'Waist(inch)', 'Waist:Hip Ratio',
        'RBS(mg/dl)', 'BP _Systolic (mmHg)', 'BP _Diastolic (mmHg)'
    ]

    lifestyle = [
        'Weight gain(Y/N)', 'hair growth(Y/N)', 'Skin darkening (Y/N)',
        'Hair loss(Y/N)', 'Pimples(Y/N)', 'Fast food (Y/N)',
        'Reg.Exercise(Y/N)', 'Marraige Status (Yrs)'
    ]

    ovarian = validate(df, ovarian, "Ovarian")
    hormonal = validate(df, hormonal, "Hormonal")
    clinical = validate(df, clinical, "Clinical")
    lifestyle = validate(df, lifestyle, "Lifestyle")

    X = {
        "Ovarian": scale(df, ovarian),
        "Hormonal": scale(df, hormonal),
        "Clinical": scale(df, clinical),
        "Lifestyle": scale(df, lifestyle)
    }

    for k, v in X.items():
        print(f"\n{k}: {v.shape}")

    return df, y, X