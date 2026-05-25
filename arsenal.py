import numpy as np
import itertools
from scipy.optimize import linear_sum_assignment
import pandas as pd

# Constants
PITCH_CHAR_FEATURES = ['release_speed', 'pfx_x', 'pfx_z']

# Arsenal distance functions

def compare_all_arsenals(pitch_distances, penalty_pctile):
    """
    Compute pairwise arsenal distances for all pitcher pairs.

    Parameters:
        pitch_distances : long-form DataFrame output from compute_mahalanobis_distances()
        penalty_pctile  : default penalty when the pitchers have different size arsenals

    Returns:
        Long-form DataFrame with columns pitcher1, pitcher2, arsenal_distance
    """
    unmatched_penalty = np.percentile(pitch_distances["distance"], penalty_pctile)
    print(unmatched_penalty)

    # Build a lookup dict: (pitcher1, pitch_type1, pitcher2, pitch_type2) -> distance
    # Store both directions so we don't need to flip later
    dist_lookup = {}
    for row in pitch_distances.itertuples(index=False):
        dist_lookup[(row.player_name1, row.game_year1, row.pitch_type1,
                     row.player_name2, row.game_year2, row.pitch_type2)] = row.distance
        dist_lookup[(row.player_name2, row.game_year2, row.pitch_type2,
                     row.player_name1, row.game_year1, row.pitch_type1)] = row.distance

    # Build a dict: pitcher -> list of pitch types
    pitcher_pitches = (
        pitch_distances[["player_name1", "game_year1", "pitch_type1"]]
        .rename(columns={"player_name1": "player_name", "game_year1":"game_year", "pitch_type1": "pitch_type"})
        .drop_duplicates()
        .groupby(["player_name", "game_year"])["pitch_type"]
        .apply(list)
        .to_dict()
    )

    pitcher_years = list(pitcher_pitches.keys())  # list of (player_name, game_year) tuples
    results = []

    for (p1, y1), (p2, y2) in itertools.combinations(pitcher_years, 2):
        pitches1 = pitcher_pitches[(p1, y1)]
        pitches2 = pitcher_pitches[(p2, y2)]
        n1, n2 = len(pitches1), len(pitches2)
        n = max(n1, n2)

        # Build cost matrix with unmatched penalty as default
        cost_matrix = np.full((n, n), unmatched_penalty)
        for i, pt1 in enumerate(pitches1):
            for j, pt2 in enumerate(pitches2):
                key = (p1, y1, pt1, p2, y2, pt2)
                if key in dist_lookup:
                    cost_matrix[i, j] = dist_lookup[key]

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        avg_distance = cost_matrix[row_ind, col_ind].sum() / n
        results.append((p1, y1, p2, y2, avg_distance))

    return (
        pd.DataFrame(results, columns=["player_name1", "game_year1", "player_name2", "game_year2", "arsenal_distance"])
        .sort_values("arsenal_distance")
        .reset_index(drop=True)
    )


def arsenal_internal_distances(pitch_type_summ, pitch_features=PITCH_CHAR_FEATURES):
    """
    For each pitcher, compute the average (and percentiles) of the closest
    distance from each pitch to any other pitch in their arsenal.
    Gives a natural scale for novelty_distance_threshold.
    """
    rows = []
    pitchers = pitch_type_summ['player_name'].unique()

    for name in pitchers:
        arsenal = (
            pitch_type_summ[pitch_type_summ['player_name'] == name]
            .dropna(subset=pitch_features)
            .reset_index(drop=True)
        )
        if len(arsenal) < 2:
            continue

        scaler = StandardScaler().fit(arsenal[pitch_features])
        X = scaler.transform(arsenal[pitch_features].values)

        # Distance matrix; set diagonal to inf so a pitch isn't its own closest
        dist_matrix = cdist(X, X, metric='euclidean')
        np.fill_diagonal(dist_matrix, np.inf)
        min_dists = dist_matrix.min(axis=1)

        rows.append({
            'player_name': name,
            'n_pitches':   len(arsenal),
            'mean_min_dist':   round(min_dists.mean(), 3),
            'min_min_dist':    round(min_dists.min(), 3),
            'p25_min_dist':    round(np.percentile(min_dists, 25), 3),
            'p50_min_dist':    round(np.percentile(min_dists, 50), 3),
            'p75_min_dist':    round(np.percentile(min_dists, 75), 3),
            'p90_min_dist':    round(np.percentile(min_dists, 90), 3),
        })

    df = pd.DataFrame(rows).sort_values('mean_min_dist').reset_index(drop=True)
    
    print("── Arsenal internal distances (across all pitchers) ──")
    print(df[['mean_min_dist', 'min_min_dist', 'p25_min_dist', 'p50_min_dist', 
              'p75_min_dist', 'p90_min_dist']].describe().round(3).to_string())
    
    return df