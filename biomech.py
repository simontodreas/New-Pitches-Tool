from distances import compute_euclidean_distances, compute_mahalanobis_distances
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

def evaluate_biomech_features(pitcher_summ, arsenal_comp, feature_sets, min_pitches=100, distance_fn=compute_euclidean_distances):
    """
    For each candidate feature set, compute biomechanical distances and correlate
    with arsenal distances. Returns a summary DataFrame ranked by Spearman correlation.
    
    Parameters:
        pitcher_summ  : pitcher-level summary DataFrame
        arsenal_comp  : arsenal distance DataFrame from compare_all_arsenals()
        feature_sets  : dict of {label: [feature columns]}
        min_pitches   : minimum pitches filter
    
    Returns:
        DataFrame with feature set label, Spearman rho, p-value, and n pairs
    """
    arsenal_both = pd.concat([
        arsenal_comp.rename(columns={'player_name1': 'p1', 'player_name2': 'p2'}),
        arsenal_comp.rename(columns={'player_name2': 'p1', 'player_name1': 'p2'})
    ])
    arsenal_lookup = arsenal_both.set_index(['p1', 'p2'])['arsenal_distance']

    results = []
    for label, features in feature_sets.items():
        biomech = distance_fn(
            pitcher_summ,
            features=features,
            label_cols=['player_name'],
            min_pitches=min_pitches
        )

        biomech['arsenal_distance'] = biomech.apply(
            lambda r: arsenal_lookup.get((r['player_name1'], r['player_name2']), np.nan), axis=1
        )
        biomech = biomech.dropna(subset=['arsenal_distance'])

        rho, pval = spearmanr(biomech['distance'], biomech['arsenal_distance'])
        results.append({
            'features': label,
            'spearman_rho': round(rho, 4),
            'p_value': pval,
            'n_pairs': len(biomech)
        })

    return pd.DataFrame(results).sort_values('spearman_rho').reset_index(drop=True)