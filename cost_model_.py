import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

###################################################################################################
### uncertainty factors meaning:
# A1 Weather 
# B2 Crew absenteeism 
# C3 Regulatory requirements (interpretation and implementation of government policy)
# D4 Design changes 
# E5 Scarcity of resources due to geographical location (economic activity level)
# F6 Social or political discontent of work men (labour unrest)
# G7 Crew interfacing 
# H8 Project complexity 
# I9 Ground/soil conditions (foundation conditions) 
# G10 Space congestion (overcrowding of workmen due to interface of activities under progress)

# K11 Managerial ability of consultant team involved 
# L12 Legal problems 
# M13 Rework due to poor material quality 
# N14 Rework due to poor work poor workmanship 
# O15 Many parties are involved directly 
# P16 Inconvenient site access 
# Q17 Limited construction area 
# R18 Delays in decision-making by project owner 
# S19 Postponement of project

# T20 Delays in payment 
# U21 Late site handover 
# V22 Late submission of nominated materials 
# W23 Late design works
# X24 Mistake in design
# Y25 Inappropriate design
# Z26 Low qualification and professional training of employees
# AA27 Late inspection by consultants
# AB28 Late issuing of approval documents
# AC29 Financial problems (limitations on provision of credit)

# AD30 Force majeure
# AE31 Corruption (political issues)
# AF32 Inflation rate
# AG33 Interest rate
# AH34 Exchange rate (availability and fluctuation in foreign exchange)
# AI35 Social and cultural conditions in the region
# AJ36 Unestimated work amounts in project’s estimate
# AK37 Unclear responsibility limits and no strict contractual obligations
# AL38 Customs and import restrictions and procedures
# AM39 Unnecessary interference by client

# AN40 Oil price
# AO41 Transportation prices
# AP42 Personal interest among consultants
# AQ43 Global economic recession
# AR44 End-users interferences
# AS45 Insecurity
###################################################################################################

def load_and_preprocess_data(filepath):
    """Load the dataset and preprocess."""
    data = pd.read_excel(filepath)
    data.drop('Unnamed: 6', axis=1, inplace=True)
    return data

def predict_initial_contract_sum(model_path, data, scope_columns):
    """Predict initial contract sums using XGBoost model."""
    xgb_model = XGBRegressor(objective='reg:squarederror')
    xgb_model.load_model(model_path)
    scope_data = data[scope_columns]
    return xgb_model.predict(scope_data.values)

def calculate_deviations(data, predictions):
    """Calculate deviations as the difference between actual and predicted sums."""
    data['Deviation'] = data['Actual Contract Sum'].values - predictions

def train_random_forest(X, y):
    """Train RandomForestRegressor and return the model and its feature importances."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=150, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    return rf_model, rf_model.feature_importances_

def calculate_composite_scores_and_ranks(data, feature_importances):
    """Calculate composite scores and adjust ranks."""
    X = data.loc[:, 'A1':'AS45']
    data['Composite Score'] = X.dot(feature_importances)
    data['Composite Rank'] = data['Composite Score'].rank(ascending=False)

def train_nearestN_model(positive_deviations_df, negative_deviations_df):
    # Combine the dataframes and label the deviation direction
    combined_df = pd.concat([
        positive_deviations_df.assign(deviation_direction=1),
        negative_deviations_df.assign(deviation_direction=0)
    ])

    # Features for the model
    X = combined_df.loc[:, 'A1':'AS45']
    # Fit the Nearest Neighbors model
    neighbors_model = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(X)

    return neighbors_model, combined_df

def get_majority_cluster(new_entry, neighbors_model, combined_df):
    # New entry features A1 to AS45
    # Find the 3 nearest neighbors of the new entry
    distances, indices = neighbors_model.kneighbors([new_entry])

    # Determine the majority cluster among the nearest neighbors
    nearest_labels = combined_df.iloc[indices[0]]['deviation_direction']
    majority_cluster = nearest_labels.mode()[0]  # 1 for positive, 0 for negative deviation

    return majority_cluster


def rank_and_adjust(new_entry_features, feature_importances, rank_table, direction, positive_deviations_df, negative_deviations_df):
    """
    Rank a new entry and adjust existing ranks in the uncertainty factors rank table.

    Parameters:
    - new_entry_features: A list or array of feature values A1 to AS45 for the new entry.
    - feature_importances: An array of feature importances.
    - rank_table: The existing ranking table including 'Composite Score' and 'Composite Rank' columns.

    Returns:
    - Updated ranking table with the new entry included and ranks adjusted.
    """
    # Calculate the composite score for the new entry
    new_entry_score = np.dot(new_entry_features, feature_importances)
    
    if not isinstance(new_entry_features, list):
        new_entry_features = new_entry_features.tolist()
    # Append the new entry to the existing rank table, ensuring we match the expected column count
    new_entry_data = new_entry_features + [new_entry_score] + [None]  # Adding None for the 'Composite Rank' which will be recalculated
    new_entry_df = pd.DataFrame([new_entry_data], columns=rank_table.columns)
    
    # Append the new entry DataFrame row to the existing table
    updated_table = pd.concat([rank_table, new_entry_df], ignore_index=True)
    new_entry_score = updated_table.iloc[updated_table.index[-1]]['Composite Score']
    
    if direction == 0:
        updated_table['Deviation'] = negative_deviations_df['Deviation'].to_list() + [None]
    else:
        updated_table['Deviation'] = positive_deviations_df['Deviation'].to_list() + [None]
        
    # Recalculate the composite scores and ranks
    updated_table['Composite Rank'] = updated_table['Composite Score'].rank(ascending=False)
    
    return updated_table.sort_values(by='Composite Rank'), new_entry_score


def get_new_entry_adjustment(new_entry, feature_importances, nearestN_model, combined_df, positive_deviations_df, negative_deviations_df, verbose=False):
    
    majority_cluster = get_majority_cluster(new_entry, nearestN_model, combined_df)
    if verbose: print('Majority cluster:', majority_cluster)
    
    # Based on the majority cluster, proceed with calculating the composite score and ranking
    # within the appropriate deviation DataFrame (positive_deviation_df or negative_deviation_df)

    positive_deviations_rankTable = positive_deviations_df.loc[:,list(positive_deviations_df.columns.values[6:51])+\
        list(positive_deviations_df.columns.values[52:])]

    negative_deviations_rankTable = negative_deviations_df.loc[:,list(negative_deviations_df.columns.values[6:51])+\
        list(negative_deviations_df.columns.values[52:])]

    if majority_cluster == 0:
        table, score = rank_and_adjust(new_entry, feature_importances, negative_deviations_rankTable, majority_cluster, positive_deviations_df, negative_deviations_df)
    else:
        table, score = rank_and_adjust(new_entry, feature_importances, positive_deviations_rankTable, majority_cluster, positive_deviations_df, negative_deviations_df)
    
    if verbose:
        print('################################################################################################################')
        print(table)
    
    new_entry_rank = table[table['Composite Score'] == score]['Composite Rank'].values[0]
    # Determine the indices for surrounding entries
    total_entries = len(table)
    lower_bound = int(max(1, new_entry_rank - 2))  # Ensure bounds are within table limits
    upper_bound = int(min(total_entries, new_entry_rank + 2))

    # Calculate the mean adjustment of the surrounding entries
    surrounding_adjustments = table.iloc[lower_bound-1:upper_bound]['Deviation']
    mean_adjustment = surrounding_adjustments.mean()

    if verbose:
        print('######################## Adjustment to initial contract sum prediction based on uncertainty variables ##########################')
        print('Adjustment: ', mean_adjustment)
    
    return mean_adjustment
    

if __name__ == '__main__':

    filepath = 'Dataset for Cost Prediction.xlsx'
    model_path = 'project_scope_model.json'
    data = load_and_preprocess_data(filepath)
    
    project_scope_columns = ['Initial Estimated Duration (Months)', 'Gross Floor Area (M2)', 'Building Height (Metres)']
    initial_predictions = predict_initial_contract_sum(model_path, data, project_scope_columns)
    calculate_deviations(data, initial_predictions)
    
    X = data.loc[:, 'A1':'AS45']
    y = data['Deviation']
    rf_model, feature_importances = train_random_forest(X, y)
    calculate_composite_scores_and_ranks(data, feature_importances)
    
    # Sort the dataframe based on composite rank
    df_composite_sorted = data.sort_values(by='Composite Rank')

    positive_deviations_df = df_composite_sorted[df_composite_sorted['Deviation'] > 0]
    negative_deviations_df = df_composite_sorted[df_composite_sorted['Deviation'] < 0]
    
    nearestN_model, combined_df = train_nearestN_model(positive_deviations_df, negative_deviations_df)
    
    # evaluate Hybrid model approach
    data_ = load_and_preprocess_data(filepath).sample(10)
    
    actual_cost = data_.loc[:, 'Actual Contract Sum']
    
    initial_cost_preds = predict_initial_contract_sum(model_path, data_, project_scope_columns)
    
    # print(actual_cost)
    print(initial_cost_preds)
    
    uncertainty_entries = data_.loc[:, 'A1':'AS45']
    
    adjustments = []
    for entry in uncertainty_entries.values:
        adjustment = get_new_entry_adjustment(entry, feature_importances, nearestN_model, combined_df, verbose=False)
        adjustments.append(adjustment)
    
    # print(adjustments)
    print('Actual cost: ', actual_cost.values)
    print('Models prediction: ', initial_cost_preds + np.array(adjustments))
    
    # print('MSE: ', mean_squared_error(actual_cost.values, initial_cost_preds + np.array(adjustments)))