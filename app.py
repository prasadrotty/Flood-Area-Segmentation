import streamlit as st
import pandas as pd
import numpy as np
from cost_model_ import predict_initial_contract_sum, get_new_entry_adjustment, load_and_preprocess_data, calculate_deviations
from cost_model_ import train_random_forest, calculate_composite_scores_and_ranks, train_nearestN_model


filepath = 'Dataset for Cost Prediction.xlsx'
model_path = 'project_scope_model.json'

@st.cache_data
def load_data_and_models():
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
    
    return nearestN_model, combined_df, feature_importances, positive_deviations_df, negative_deviations_df

def evaluate_uncertainty_adjustments(new_entry, feature_importances, nearestN_model, combined_df, positive_deviations_df, negative_deviations_df):
    
    adjustment = get_new_entry_adjustment(new_entry, feature_importances, nearestN_model, combined_df, positive_deviations_df, negative_deviations_df, verbose=False)
    return adjustment
    
# Title for the app
st.title('Construction Cost Prediction Tool')

# User input fields for project scope
duration = st.number_input('Initial Estimated Duration (Months)', min_value=1.0, step=1.0, value=57.0)
building_height = st.number_input('Building Height (Metres)', min_value=1.0, step=1.0, value=7.0)
floor_area = st.number_input('Gross Floor Area (M2)', min_value=1.0, step=1.0, value=730.0)

uncertainty_factors = [5, 5, 4, 4, 4, 3, 3, 4, 4, 3, 4, 5, 4, 5, 5, 5, 5,
       5, 3, 4, 4, 4, 4, 3, 3, 3, 4, 4, 4, 3, 3, 4, 4, 3,
       4, 3, 3, 3, 4, 5, 3, 4, 3, 5, 5]

    # Sliders for uncertainty factors A1 to AS45
with st.expander("Uncertainty Factors"):
    # Sliders for uncertainty factors A1 to AS45 with default values from uncertainty_factors list
    a1 = st.slider('Weather', 0, 5, uncertainty_factors[0])
    b2 = st.slider('Crew Absenteeism', 0, 5, uncertainty_factors[1])
    c3 = st.slider('Regulatory Requirements', 0, 5, uncertainty_factors[2])
    d4 = st.slider('Design Changes', 0, 5, uncertainty_factors[3])
    e5 = st.slider('Scarcity of Resources', 0, 5, uncertainty_factors[4])
    f6 = st.slider('Labour Unrest', 0, 5, uncertainty_factors[5])
    g7 = st.slider('Crew Interfacing', 0, 5, uncertainty_factors[6])
    h8 = st.slider('Project Complexity', 0, 5, uncertainty_factors[7])
    i9 = st.slider('Ground/Soil Conditions', 0, 5, uncertainty_factors[8])
    j10 = st.slider('Space Congestion', 0, 5, uncertainty_factors[9])
    k11 = st.slider('Managerial Ability', 0, 5, uncertainty_factors[10])
    l12 = st.slider('Legal Problems', 0, 5, uncertainty_factors[11])
    m13 = st.slider('Rework due to Poor Material', 0, 5, uncertainty_factors[12])
    n14 = st.slider('Rework due to Poor Workmanship', 0, 5, uncertainty_factors[13])
    o15 = st.slider('Many Parties Involved', 0, 5, uncertainty_factors[14])
    p16 = st.slider('Inconvenient Site Access', 0, 5, uncertainty_factors[15])
    q17 = st.slider('Limited Construction Area', 0, 5, uncertainty_factors[16])
    r18 = st.slider('Delays in Decision-Making', 0, 5, uncertainty_factors[17])
    s19 = st.slider('Postponement of Project', 0, 5, uncertainty_factors[18])
    t20 = st.slider('Delays in Payment', 0, 5, uncertainty_factors[19])
    u21 = st.slider('Late Site Handover', 0, 5, uncertainty_factors[20])
    v22 = st.slider('Late Submission of Materials', 0, 5, uncertainty_factors[21])
    w23 = st.slider('Late Design Works', 0, 5, uncertainty_factors[22])
    x24 = st.slider('Mistake in Design', 0, 5, uncertainty_factors[23])
    y25 = st.slider('Inappropriate Design', 0, 5, uncertainty_factors[24])
    z26 = st.slider('Low Qualification of Employees', 0, 5, uncertainty_factors[25])
    aa27 = st.slider('Late Inspection', 0, 5, uncertainty_factors[26])
    ab28 = st.slider('Late Issuing of Documents', 0, 5, uncertainty_factors[27])
    ac29 = st.slider('Financial Problems', 0, 5, uncertainty_factors[28])
    ad30 = st.slider('Force Majeure', 0, 5, uncertainty_factors[29])
    ae31 = st.slider('Corruption', 0, 5, uncertainty_factors[30])
    af32 = st.slider('Inflation Rate', 0, 5, uncertainty_factors[31])
    ag33 = st.slider('Interest Rate', 0, 5, uncertainty_factors[32])
    ah34 = st.slider('Exchange Rate', 0, 5, uncertainty_factors[33])
    ai35 = st.slider('Social and Cultural Conditions', 0, 5, uncertainty_factors[34])
    aj36 = st.slider('Unestimated Work Amounts', 0, 5, uncertainty_factors[35])
    ak37 = st.slider('Unclear Responsibility Limits', 0, 5, uncertainty_factors[36])
    al38 = st.slider('Customs and Import Restrictions', 0, 5, uncertainty_factors[37])
    am39 = st.slider('Client Interference', 0, 5, uncertainty_factors[38])
    an40 = st.slider('Oil Price', 0, 5, uncertainty_factors[39])
    ao41 = st.slider('Transportation Prices', 0, 5, uncertainty_factors[40])
    ap42 = st.slider('Personal Interest Among Consultants', 0, 5, uncertainty_factors[41])
    aq43 = st.slider('Global Economic Recession', 0, 5, uncertainty_factors[42])
    ar44 = st.slider('End-Users Interferences', 0, 5, uncertainty_factors[43])
    as45 = st.slider('Insecurity', 0, 5, uncertainty_factors[44])

new_factors = [
    a1, b2, c3, d4, e5, f6, g7, h8, i9, j10,
    k11, l12, m13, n14, o15, p16, q17, r18, s19,
    t20, u21, v22, w23, x24, y25, z26, aa27, ab28,
    ac29, ad30, ae31, af32, ag33, ah34, ai35, aj36,
    ak37, al38, am39, an40, ao41, ap42, aq43, ar44, as45
]

if st.button('Predict Construction Cost'):
    
    nearestN_model, combined_df, feature_importances, positive_deviations_df, negative_deviations_df = load_data_and_models()
    
    # Preprocess inputs
    input_features = pd.DataFrame([[duration, floor_area, building_height]], 
                                   columns=['Initial Estimated Duration (Months)', 'Gross Floor Area (M2)', 'Building Height (Metres)'])
    
    # Call the model prediction function
    initial_cost_pred = predict_initial_contract_sum(model_path, input_features, input_features.columns)
    
    # adjustment calculation based on user inputs for uncertainty factors
    adjustment = evaluate_uncertainty_adjustments(new_factors, feature_importances, nearestN_model, combined_df, positive_deviations_df, negative_deviations_df)
    final_cost = initial_cost_pred+adjustment
    
    st.write(f"Predicted Construction Cost: {final_cost[0]}")
    # st.write(f"Predicted Construction Cost: ${final_cost:,.2f}")

