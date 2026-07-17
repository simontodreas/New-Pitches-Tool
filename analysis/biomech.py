from src.distances import compute_euclidean_distances
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# The canonical feature list lives in src.pitch_suggestions; import rather than
# redefine so these analyses can't drift from the app.
from src.pitch_suggestions import BIOMECH_FEATURES


def evaluate_biomech_features(pitcher_summ, arsenal_comp, feature_sets, min_pitches=20, 
                              distance_fn=compute_euclidean_distances):
    """
    For each candidate feature set, compute biomechanical distances and correlate
    with arsenal distances. Returns a summary DataFrame ranked by Spearman correlation.

    Expects single-season inputs: distances and the arsenal join use player_name
    alone, so a multi-year pitcher_summ would pair pitchers with themselves
    across years.

    Parameters:
        pitcher_summ  : single-season pitcher-level summary DataFrame
        arsenal_comp  : arsenal distance DataFrame from compare_all_arsenals()
        feature_sets  : dict of {label: [feature columns]}
        min_pitches   : minimum pitches filter
        distance_fn   : compute_euclidean_distances or compute_mahalanobis_distances

    Returns:
        DataFrame with columns Features, Spearman_rho, p_value, sorted ascending
        by Spearman_rho (strongest feature set last)
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
            'Features': label,
            'Spearman_rho': round(rho, 4),
            'p_value': pval,
        })

    return pd.DataFrame(results).sort_values('Spearman_rho').reset_index(drop=True)

def biomech_threshold_coverage(
    pitcher_summ,
    thresholds=(1.0, 1.5, 2.0),
    min_pitches=100,
    biomech_features=BIOMECH_FEATURES,
):
    """
    For each pitcher (anchored on their most recent qualifying year), count the
    comps within each candidate biomech distance threshold and summarize coverage:
    how many comps a threshold leaves the typical pitcher, and what share of
    pitchers it strands with zero (pct_zero) or fewer than 5 (pct_lt5) comps.
    Comp pitcher-years are deduplicated to the closest year per comp, mirroring
    suggest_pitches.

    Prints the summary table and returns it: one row per threshold with columns
    threshold, mean_comps, p10/p25/p50/p75_comps, pct_zero, pct_lt5.
    """
    biomech_dist = compute_euclidean_distances(
        pitcher_summ,
        features=biomech_features,
        label_cols=['player_name', 'game_year'],
        min_pitches=min_pitches,
    )

    targets = (
        pitcher_summ[pitcher_summ['n'] >= min_pitches]
        .sort_values('game_year', ascending=False)
        .drop_duplicates(subset='player_name')
        [['player_name', 'game_year']]
    )

    left = biomech_dist.merge(
        targets, left_on=['player_name1', 'game_year1'], right_on=['player_name', 'game_year']
    )[['player_name', 'game_year', 'player_name2', 'game_year2', 'distance']].rename(
        columns={'player_name2': 'comp_pitcher', 'game_year2': 'comp_year'}
    )

    right = biomech_dist.merge(
        targets, left_on=['player_name2', 'game_year2'], right_on=['player_name', 'game_year']
    )[['player_name', 'game_year', 'player_name1', 'game_year1', 'distance']].rename(
        columns={'player_name1': 'comp_pitcher', 'game_year1': 'comp_year'}
    )

    target_pairs = pd.concat([left, right], ignore_index=True)

    # Remove self-comparisons
    target_pairs = target_pairs[target_pairs['player_name'] != target_pairs['comp_pitcher']]

    # Deduplicate comp pitcher-years: keep only the closest year per comp,
    # mirroring the drop_duplicates logic in suggest_pitches
    target_pairs = (
        target_pairs
        .sort_values('distance')
        .drop_duplicates(subset=['player_name', 'game_year', 'comp_pitcher'])
        .reset_index(drop=True)
    )

    rows = []
    for threshold in thresholds:
        comp_counts = (
            target_pairs[target_pairs['distance'] <= threshold]
            .groupby(['player_name', 'game_year'])
            .size()
            .reindex(pd.MultiIndex.from_frame(targets), fill_value=0)
            .values
        )

        rows.append({
            'threshold':  threshold,
            'mean_comps': round(comp_counts.mean(), 1),
            'p10_comps':  int(np.percentile(comp_counts, 10)),
            'p25_comps':  int(np.percentile(comp_counts, 25)),
            'p50_comps':  int(np.percentile(comp_counts, 50)),
            'p75_comps':  int(np.percentile(comp_counts, 75)),
            'pct_zero':   round((comp_counts == 0).mean() * 100, 1),
            'pct_lt5':    round((comp_counts < 5).mean() * 100, 1),
        })

    df = pd.DataFrame(rows)
    print("── Biomech threshold coverage ──")
    print(df.to_string(index=False))
    return df


def biomech_threshold_calibration(
    pitcher_summ,
    arsenal_comp,
    biomech_features=BIOMECH_FEATURES,
    min_pitches=100,
    n_bins=20,
    max_biomech_dist=None,
):
    """
    Bin pitcher pairs by biomechanical distance and compute mean/median arsenal
    distance within each bin. Helps calibrate a biomech threshold by showing
    where the biomech→arsenal signal holds vs. degrades.

    Parameters:
        pitcher_summ      : pitcher-level summary DataFrame
        arsenal_comp      : arsenal distance DataFrame from compare_all_arsenals()
        biomech_features  : list of biomechanical feature columns
        min_pitches       : minimum pitches filter passed to distance function
        n_bins            : number of equal-width bins for biomech distance
        max_biomech_dist  : if set, drop pairs with biomech distance above this value
                            before binning (trims the long right tail)

    Returns:
        bin_df : DataFrame with columns:
                   biomech_bin_mid  – bin midpoint
                   mean_arsenal     – mean arsenal distance in that bin
                   median_arsenal   – median arsenal distance in that bin
                   n_pairs          – number of pairs in the bin
                 bin_df.attrs['n_total_pairs'] holds the pair count before the
                 max_biomech_dist trim; plot_threshold_calibration reads it to
                 normalize its CDF over all pairs.
    """
    biomech_dist = compute_euclidean_distances(
        pitcher_summ,
        features=biomech_features,
        label_cols=['player_name', 'game_year'],
        min_pitches=min_pitches,
    )

    arsenal_both = pd.concat([
        arsenal_comp[['player_name1', 'game_year1', 'player_name2', 'game_year2', 'arsenal_distance']],
        arsenal_comp.rename(columns={
            'player_name1': 'player_name2', 'game_year1': 'game_year2',
            'player_name2': 'player_name1', 'game_year2': 'game_year1',
        })[['player_name1', 'game_year1', 'player_name2', 'game_year2', 'arsenal_distance']],
    ])
    arsenal_lookup = arsenal_both.set_index(
        ['player_name1', 'game_year1', 'player_name2', 'game_year2']
    )['arsenal_distance']

    merged = biomech_dist.copy()
    merged['arsenal_distance'] = merged.apply(
        lambda r: arsenal_lookup.get(
            (r['player_name1'], r['game_year1'], r['player_name2'], r['game_year2']), np.nan
        ),
        axis=1,
    )
    merged = merged.dropna(subset=['arsenal_distance'])
    n_total_pairs = len(merged)  # all pairs with an arsenal distance, before trimming outliers

    if max_biomech_dist is not None:
        merged = merged[merged['distance'] <= max_biomech_dist]

    merged['biomech_bin'] = pd.cut(merged['distance'], bins=n_bins)

    rows = []
    for bin_interval, group in merged.groupby('biomech_bin', observed=True):
        rows.append({
            'biomech_bin_mid': round(bin_interval.mid, 3),
            'mean_arsenal':    round(group['arsenal_distance'].mean(), 4),
            'median_arsenal':  round(group['arsenal_distance'].median(), 4),
            'n_pairs':         len(group),
        })

    bin_df = pd.DataFrame(rows)
    bin_df.attrs['n_total_pairs'] = n_total_pairs
    return bin_df


def plot_threshold_calibration(bin_df, threshold=1.5, total_pairs=None):
    """
    Two stacked panels sharing a biomech-distance x-axis:
      top    - median arsenal distance per biomech-distance bin
      bottom - CDF of pitcher pairs over biomech distance (cumulative share of
               pairs within each distance, built from the per-bin counts)
    A dashed line marks the chosen `threshold`.

    The CDF is normalized over *all* pairs, including any dropped by
    `max_biomech_dist`: the total is read from `bin_df.attrs['n_total_pairs']`
    (set by `biomech_threshold_calibration`), or supplied via `total_pairs`,
    falling back to the in-frame count. When the frame was trimmed the curve
    ends below 1.0 - that gap is the share of far-outlier pairs left off the axis.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Top: median arsenal distance per biomech-distance bin
    ax1.plot(bin_df['biomech_bin_mid'], bin_df['median_arsenal'],
             color='steelblue', lw=2, marker='o', ms=4)
    ax1.set_ylabel('Median arsenal distance')
    ax1.set_title('Arsenal distance vs. biomechanical distance bin')

    # Bottom: CDF of pairs over biomech distance, normalized over ALL pairs
    total = (total_pairs if total_pairs is not None
             else bin_df.attrs.get('n_total_pairs', bin_df['n_pairs'].sum()))
    cdf = bin_df['n_pairs'].cumsum() / total
    ax2.plot(bin_df['biomech_bin_mid'], cdf,
             color='steelblue', lw=2, marker='o', ms=4)
    ax2.set_ylabel('Cumulative share of pairs')
    ax2.set_xlabel('Biomechanical distance (bin midpoint)')
    ax2.set_ylim(0, 1.02)

    # Threshold marker on both panels
    for ax in (ax1, ax2):
        ax.axvline(threshold, color='firebrick', lw=1.5, ls='--', zorder=1)
    ax1.text(threshold, ax1.get_ylim()[1], f' threshold = {threshold}',
             color='firebrick', fontsize=9, va='top', ha='left')

    plt.tight_layout()
    plt.show()