import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def stability_analysis(
    statcast_df,
    features,
    sample_sizes=None,
    n_replicates=50,
    n_pitchers=50,
    group_by_pitch_type=False,
    random_state=42,
):
    """
    For each feature, compute stability across sample sizes by bootstrapping
    pitch-level data and measuring the sampling SD of estimates across replicates.

    Parameters
    ----------
    statcast_df          : pitch-level DataFrame (statcast_25_clean)
    features             : list of feature names to test
    sample_sizes         : list of ints; defaults to [10, 25, 50, 75, 100, 150, 200]
    n_replicates         : bootstrap replicates per sample size
    n_pitchers           : how many groups to sample for the analysis;
                           only groups with >= max(sample_sizes) pitches are eligible
    group_by_pitch_type  : if True, group by (player_name, pitch_type) instead of
                           player_name alone; use for pitch-characteristic features
    random_state         : for reproducibility

    Returns
    -------
    stability_df : aggregated long-format DataFrame with columns
                   [sample_size, feature, mean_se, p25_se, p50_se, p75_se]
    raw_df       : one row per (group, sample_size, feature, replicate-summary)
    """
    if sample_sizes is None:
        sample_sizes = [10, 25, 50, 75, 100, 150, 200]

    min_pitches = max(sample_sizes)
    rng         = np.random.default_rng(random_state)

    group_cols = ['player_name', 'pitch_type'] if group_by_pitch_type else ['player_name']

    pitch_counts = statcast_df.groupby(group_cols).size()
    eligible     = pitch_counts[pitch_counts >= min_pitches].index.tolist()
    n_sample     = min(n_pitchers, len(eligible))
    sampled      = [eligible[i] for i in rng.choice(len(eligible), size=n_sample, replace=False)]

    rows = []
    for key in sampled:
        if group_by_pitch_type:
            name, pitch_type = key
            group_data = statcast_df[
                (statcast_df['player_name'] == name) &
                (statcast_df['pitch_type'] == pitch_type)
            ].reset_index(drop=True)
            label = f"{name} / {pitch_type}"
        else:
            name       = key
            group_data = statcast_df[statcast_df['player_name'] == name].reset_index(drop=True)
            label      = name

        for n in sample_sizes:
            if n > len(group_data):
                continue

            replicate_vals = {f: [] for f in features}
            for _ in range(n_replicates):
                subset = group_data.sample(n=n, replace=False, random_state=None)
                for feature in features:
                    vals = subset[feature].dropna()
                    if len(vals) > 0:
                        replicate_vals[feature].append(vals.mean())

            for feature in features:
                vals = replicate_vals[feature]
                if len(vals) < 2:
                    continue
                vals = np.array(vals)
                rows.append({
                    'group':       label,
                    'sample_size': n,
                    'feature':     feature,
                    'sampling_sd': vals.std(),
                    'mean_est':    round(vals.mean(), 2),
                })

    raw_df = pd.DataFrame(rows)

    stability_df = (
        raw_df.groupby(['sample_size', 'feature'])['sampling_sd']
        .agg(
            mean_se='mean',
            p25_se=lambda x: np.percentile(x, 25),
            p50_se=lambda x: np.percentile(x, 50),
            p75_se=lambda x: np.percentile(x, 75),
        )
        .round(4)
        .reset_index()
    )

    return stability_df, raw_df

