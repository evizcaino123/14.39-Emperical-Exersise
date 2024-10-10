import argparse
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests  # Added for multiple testing correction
from src.data_loading import get_n_f_y_matrix_separate


def get_configs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_data', type=str, default='P_Data_Extract_From_Doing_Business.xlsx')
    parser.add_argument('--calculate_p_values', type=bool, default=True)
    parser.add_argument('--generate_plots', type=bool, default=True)
    parser.add_argument('--save_dir', type=str, default='result1/test7')
    parser.add_argument('--alpha_init', type=float, default=0.05)
    parser.add_argument('--feature_list', type=list, default=[
        # Original features
        'Enforcing contracts (DB04-15 methodology) - Score',
        'Resolving insolvency - Score',
        'Starting a business: Procedures required - Men (number) - Score',
        'Registering property: Procedures (number) - Score',
        'Dealing with construction permits: Time (days)  - Score',
        'Resolving insolvency: Strength of insolvency framework index (0-16) - Score',
        'Registering property: Cost (% of property value)',
    ])
    parser.add_argument('--target_features', type=list, default=[
        'Starting a business: Cost - Men (% of income per capita)',
        'Starting a business - Score',
        'Starting a business: Paid-in Minimum capital (% of income per capita) - Score'
    ])
    parser.add_argument('--start_year', type=int, default=2004)
    parser.add_argument('--end_year', type=int, default=2014)
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    return args


def calculate_normalized_changes(matrix):
    """
    Calculate normalized changes across years for each country and feature.

    Args:
        matrix: numpy array of shape (n_countries, n_features, n_years)

    Returns:
        normalized_changes: numpy array of shape (n_countries, n_features, n_years - 1)
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized_changes = (matrix[:, :, 1:] - matrix[:, :, :-1]) / matrix[:, :, :-1]
        normalized_changes = np.where(np.isfinite(normalized_changes), normalized_changes, np.nan)
    return normalized_changes


def calculate_p_values(results_matrix, data_matrix, alpha_init, country_names, data_feature_names, result_feature_names):
    """
    For each country, calculates regression, correlation, and t-test p-values by analyzing changes in features.

    Args:
        results_matrix: numpy array of shape (n_countries, n_result_features, n_years)
        data_matrix: numpy array of shape (n_countries, n_data_features, n_years)
        alpha_init: initial alpha value for significance level
        country_names: list of country names
        data_feature_names: list of data feature names
        result_feature_names: list of result feature names

    Returns:
        p_values_df: pandas DataFrame with regression, correlation, and t-test p-values
    """
    n_countries, n_data_features, n_years = data_matrix.shape
    _, n_result_features, _ = results_matrix.shape

    # Calculate normalized changes
    data_changes = calculate_normalized_changes(data_matrix)  # shape (n_countries, n_data_features, n_years - 1)
    results_changes = calculate_normalized_changes(results_matrix)  # shape (n_countries, n_result_features, n_years - 1)

    # Initialize list to store P-values
    p_values_list = []

    for country_idx in range(n_countries):
        country_name = country_names[country_idx]

        X = data_changes[country_idx, :, :].T  # shape (n_years - 1, n_data_features)
        y = results_changes[country_idx, :, :].T  # shape (n_years - 1, n_result_features)

        # For each result feature, run regression against data features
        for result_feature_idx in range(n_result_features):
            y_feature = y[:, result_feature_idx]

            # Handle missing data
            mask_y = ~np.isnan(y_feature)
            X_masked = X[mask_y]
            y_masked = y_feature[mask_y]

            # Also ensure X_masked does not have NaNs
            mask_X = ~np.isnan(X_masked).any(axis=1)
            X_masked = X_masked[mask_X]
            y_masked = y_masked[mask_X]

            # For the t-test and correlation, we need to handle each data feature separately
            for feature_idx in range(n_data_features):
                data_feature_name = data_feature_names[feature_idx]
                result_feature_name = result_feature_names[result_feature_idx]
                x_feature = X_masked[:, feature_idx]

                # Mask to remove NaNs
                mask = ~np.isnan(x_feature) & ~np.isnan(y_masked)
                x_feature_masked = x_feature[mask]
                y_feature_masked = y_masked[mask]

                # Proceed only if we have enough data points
                if len(x_feature_masked) > 2:
                    # Pearson correlation
                    corr_coef, corr_p_value = stats.pearsonr(x_feature_masked, y_feature_masked)

                    # Paired t-test
                    t_stat, t_p_value = stats.ttest_rel(x_feature_masked, y_feature_masked)

                    # Store the results
                    p_values_list.append({
                        'Country': country_name,
                        'Result Feature': result_feature_name,
                        'Data Feature': data_feature_name,
                        'Regression P-Value': np.nan,  # Will fill in later
                        'Correlation P-Value': corr_p_value,
                        'Correlation Coefficient': corr_coef,
                        'T-Test P-Value': t_p_value,
                        'T-Test Statistic': t_stat
                    })
                else:
                    # Not enough data
                    p_values_list.append({
                        'Country': country_name,
                        'Result Feature': result_feature_name,
                        'Data Feature': data_feature_name,
                        'Regression P-Value': np.nan,  # Will fill in later
                        'Correlation P-Value': np.nan,
                        'Correlation Coefficient': np.nan,
                        'T-Test P-Value': np.nan,
                        'T-Test Statistic': np.nan
                    })

            # Now perform regression using all data features
            if X_masked.shape[0] > n_data_features:
                # Add constant term for intercept
                X_design = sm.add_constant(X_masked)  # shape (n_samples, n_features + 1)

                # Perform regression using statsmodels
                model = sm.OLS(y_masked, X_design)
                results = model.fit()

                # Get p-values (exclude intercept)
                p_values = results.pvalues[1:]

                # Update the regression p-values in p_values_list
                idx_start = len(p_values_list) - n_data_features
                for feature_idx in range(n_data_features):
                    p_values_list[idx_start + feature_idx]['Regression P-Value'] = p_values[feature_idx]
            else:
                # Not enough data for regression, regression p-values remain NaN
                pass

    p_values_df = pd.DataFrame(p_values_list)

    # Multiple testing corrections and hypothesis testing results
    for test in ['Regression', 'Correlation', 'T-Test']:
        p_val_col = f'{test} P-Value'
        p_values = p_values_df[p_val_col].values

        # Apply Bonferroni correction
        bonferroni_corrected = multipletests(p_values, alpha=alpha_init, method='bonferroni')
        p_values_df[f'{test} P-Value Bonferroni'] = bonferroni_corrected[1]
        p_values_df[f'{test} Reject Bonferroni'] = bonferroni_corrected[0]

        # Apply Holm correction
        holm_corrected = multipletests(p_values, alpha=alpha_init, method='holm')
        p_values_df[f'{test} P-Value Holm'] = holm_corrected[1]
        p_values_df[f'{test} Reject Holm'] = holm_corrected[0]

        # Unadjusted hypothesis test
        p_values_df[f'{test} Reject Unadjusted'] = p_values_df[p_val_col] < alpha_init

    return p_values_df


def plot_p_values(p_values_df, save_dir, alpha_init):
    """
    Generates various plots based on the P-values DataFrame.

    Args:
        p_values_df: pandas DataFrame containing P-values and correlation coefficients.
        save_dir: directory to save the plots.
        alpha_init: significance level (alpha)
    """
    # Ensure the save directory exists
    plots_dir = os.path.join(save_dir, 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Heatmaps and Histograms for each Result Feature separately
    result_features = p_values_df['Result Feature'].unique()

    for result_feature in result_features:
        rf_p_values_df = p_values_df[p_values_df['Result Feature'] == result_feature]

        # Heatmap of Regression P-Values
        pivot_table = rf_p_values_df.pivot_table(
            index='Data Feature',
            columns='Country',
            values='Regression P-Value',
            aggfunc=np.mean
        )

        # Check if pivot_table is empty or contains only NaN values
        if pivot_table.empty or pivot_table.isnull().all().all():
            print(f"No valid Regression P-Value data for {result_feature}, skipping heatmap.")
        else:
            plt.figure(figsize=(12, 10))
            sns.heatmap(pivot_table, annot=True, cmap='viridis', cbar_kws={'label': 'P-Value'})
            plt.title(f'Heatmap of Mean Regression P-Values for {result_feature}')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'heatmap_regression_pvalues_{result_feature}.png'))
            plt.close()

        # Histograms of T-Test P-Values with alpha lines
        ttest_pvalues = rf_p_values_df['T-Test P-Value'].dropna()
        if not ttest_pvalues.empty:
            plt.figure(figsize=(10, 6))
            sns.histplot(ttest_pvalues, bins=20, kde=True, color='blue', label='T-Test P-Values')
            plt.axvline(x=alpha_init, color='red', linestyle='--', label=f'Alpha = {alpha_init}')
            plt.title(f'Histogram of T-Test P-Values for {result_feature}')
            plt.xlabel('P-Value')
            plt.ylabel('Frequency')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'histogram_ttest_pvalues_{result_feature}.png'))
            plt.close()
        else:
            print(f"No valid T-Test P-Value data for {result_feature}, skipping histogram.")

    for result_feature in result_features:
        rf_p_values_df = p_values_df[p_values_df['Result Feature'] == result_feature]

        # Heatmap of Regression P-Values
        pivot_table = rf_p_values_df.pivot_table(
            index='Data Feature',
            columns='Country',
            values='Regression P-Value',
            aggfunc=np.mean
        )

        # Histograms of P-Values
        plt.figure(figsize=(10, 6))
        sns.histplot(rf_p_values_df['Regression P-Value'].dropna(), bins=20, kde=True)
        plt.title(f'Histogram of Regression P-Values for {result_feature}')
        plt.xlabel('P-Value')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'histogram_regression_pvalues_{result_feature}.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        sns.histplot(rf_p_values_df['Correlation P-Value'].dropna(), bins=20, kde=True)
        plt.title(f'Histogram of Correlation P-Values for {result_feature}')
        plt.xlabel('P-Value')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'histogram_correlation_pvalues_{result_feature}.png'))
        plt.close()

    # Additional plots can be added as required.


def main():
    configs = get_configs()
    df = pd.read_excel(configs.path_to_data)

    # Assuming get_n_f_y_matrix_separate returns the data matrix and additional info
    data_matrix, country_names, data_feature_names = get_n_f_y_matrix_separate(
        df, configs.feature_list, configs.start_year, configs.end_year
    )
    print("Data matrix loaded.")
    results_matrix, _, result_feature_names = get_n_f_y_matrix_separate(
        df, configs.target_features, configs.start_year, configs.end_year
    )
    print("Results matrix loaded.")
    if configs.calculate_p_values:
        p_values_df = calculate_p_values(
            results_matrix, data_matrix, configs.alpha_init,
            country_names, data_feature_names, result_feature_names
        )

        # Save P-values to CSV
        p_values_df.to_csv(os.path.join(configs.save_dir, 'p_values.csv'), index=False)

        # Save hypothesis test results for each test
        for test in ['Regression', 'Correlation', 'T-Test']:
            test_columns = [
                'Country', 'Result Feature', 'Data Feature',
                f'{test} P-Value', f'{test} Reject Unadjusted',
                f'{test} P-Value Bonferroni', f'{test} Reject Bonferroni',
                f'{test} P-Value Holm', f'{test} Reject Holm'
            ]
            test_df = p_values_df[test_columns]
            test_df.to_csv(os.path.join(configs.save_dir, f'{test}_hypothesis_test_results.csv'), index=False)

        if configs.generate_plots:
            # Generate plots
            plot_p_values(p_values_df, configs.save_dir, configs.alpha_init)


if __name__ == '__main__':
    main()
