import numpy as np
import pandas as pd

def get_n_f_y_matrix_separate(data_df, feature_list, start_year, end_year):
    cleaned_data = clean_data(data_df)
    series_names = cleaned_data['Series Name'].unique()
    
    present_features = [feature for feature in feature_list if feature in series_names]
    missing_features = [feature for feature in feature_list if feature not in series_names]
    print
    print("Features present in data:")
    print(present_features)
    print("Features missing from data:")
    print(missing_features)
    
    # Proceed only with present features
    filtered_data = isolate_features(cleaned_data, present_features)
    
    # Define the year columns based on the specified range
    year_columns = [f"{year} [YR{year}]" for year in range(start_year, end_year + 1)]
    relevant_columns = ['Country Name', 'Series Name'] + year_columns
    filtered_data = filtered_data[relevant_columns]

    # Get unique countries and features
    countries = filtered_data['Country Name'].unique()
    features = feature_list  # Use the provided feature list
    years = range(start_year, end_year + 1)
    num_countries = len(countries)
    num_features = len(features)
    num_years = len(years)

    # Initialize the N x F x Y matrix with NaN values
    n_f_y_matrix = np.full((num_countries, num_features, num_years), np.nan)

    # Create mappings from country names and feature names to indices
    country_index = {country: idx for idx, country in enumerate(countries)}
    feature_index = {feature: idx for idx, feature in enumerate(features)}
    # Mapping from year column names to year indices
    year_column_index = {year_col: idx for idx, year_col in enumerate(year_columns)}

    # Populate the N x F x Y matrix
    for idx, row in filtered_data.iterrows():
        country = row['Country Name']
        feature = row['Series Name']
        if country in country_index and feature in feature_index:
            c_idx = country_index[country]
            f_idx = feature_index[feature]
            for y_col in year_columns:
                y_idx = year_column_index[y_col]
                value = row[y_col]
                n_f_y_matrix[c_idx, f_idx, y_idx] = value
    print("Data loaded successfully!")
    return n_f_y_matrix, countries, features

def clean_data(data_df):
    # Replace '..' with NaN and convert columns to numeric
    data_df = data_df.copy()  # Avoid modifying the original DataFrame
    data_df.replace("..", np.nan, inplace=True)
    yearly_columns = [col for col in data_df.columns if '[YR' in col]
    data_df.dropna(subset=yearly_columns, how='all', inplace=True)
    for col in yearly_columns:
        data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
    return data_df

def isolate_features(data_df, feature_list):
    # Filter the DataFrame to include only the specified features
    filtered_data = data_df[data_df['Series Name'].isin(feature_list)]
    return filtered_data
