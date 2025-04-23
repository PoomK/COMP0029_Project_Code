import pandas as pd
import numpy as np
import pickle
import os

def predict_svm(home_team: str, away_team: str, gw: int):
    """
    Predict the match result using an SVM model with one-hot encoding 
    for the Home Team and Away Team features.

    Args:
        home_team (str): The home team's name (must match CSV index).
        away_team (str): The away team's name (must match CSV index).
        gw (int): The gameweek number.

    Returns:
        (str, float, float, float):
            - result_message: e.g. "Match: Arsenal vs Fulham\nPrediction: Arsenal Win"
            - home_prob: Probability that the home side wins
            - draw_prob: Probability of a draw
            - away_prob: Probability that the away side wins
    """

    # 1. Load the one-hot encoder (the same one used in training the SVM)
    with open("../Pre-Processing/Encoders_final/onehot_encoder.pkl", "rb") as f:
        ohe = pickle.load(f)

    # 2. Load your trained SVM model (make sure it was trained with probability=True)
    with open("../Models/saved_models_result/svm_model.pkl", "rb") as f:
        svm_model = pickle.load(f)

    # 3. Define your home/away mappings (same as in logistic_regression code)
    home_mapping = {
        "Points": "Home_Points",
        "Goals_Scored": "Home_Goals_Scored",
        "Goals_Conceded": "Home_Goals_Conceded",
        "Wins": "Home_Wins",
        "Draws": "Home_Draws",
        "Losses": "Home_Losses",
        "Form_Score": "Home_Form_Score",
        "Shots on Goal": "Home Shots on Goal",
        "Corner Kicks": "Home Corner Kicks",
        "Ball Possession": "Home Ball Possession",
        "Yellow Cards": "Home Yellow Cards",
        "Red Cards": "Home Red Cards",
        "Offsides": "Home Offsides",
        "expected_goals": "Home expected_goals",
    }

    away_mapping = {
        "Points": "Away_Points",
        "Goals_Scored": "Away_Goals_Scored",
        "Goals_Conceded": "Away_Goals_Conceded",
        "Wins": "Away_Wins",
        "Draws": "Away_Draws",
        "Losses": "Away_Losses",
        "Form_Score": "Away_Form_Score",
        "Shots on Goal": "Away Shots on Goal",
        "Corner Kicks": "Away Corner Kicks",
        "Ball Possession": "Away Ball Possession",
        "Yellow Cards": "Away Yellow Cards",
        "Red Cards": "Away Red Cards",
        "Offsides": "Away Offsides",
        "expected_goals": "Away expected_goals",
    }

    # 4. Load the per-team stats for the specified gameweek
    team_df = pd.read_csv(
        f"Prediction_Features/Team_Form_And_Averages_2024_25_gw_{gw}.csv",
        index_col="Team"
    )

    # 5. Construct one row of data
    row = {
        "Home Team": home_team,
        "Away Team": away_team,
    }

    # Extract numeric stats for home and away
    home_row = team_df.loc[home_team]
    away_row = team_df.loc[away_team]

    for src_col, dst_col in home_mapping.items():
        row[dst_col] = home_row[src_col]

    for src_col, dst_col in away_mapping.items():
        row[dst_col] = away_row[src_col]

    df_input = pd.DataFrame([row])

    # 6. Split out the categorical columns and one-hot encode them
    categorical_cols = ["Home Team", "Away Team"]
    df_cat = df_input[categorical_cols]
    arr_encoded = ohe.transform(df_cat)  # transform, not fit_transform

    # 7. Convert the one-hot array back to DataFrame columns
    encoded_cols = ohe.get_feature_names_out(categorical_cols)
    df_cat_encoded = pd.DataFrame(arr_encoded, columns=encoded_cols, index=df_input.index)

    # 8. Combine numeric + encoded columns
    df_num = df_input.drop(columns=categorical_cols)
    df_encoded = pd.concat([df_num, df_cat_encoded], axis=1)

    # 9. Reindex columns so they line up with what the SVM expects
    #    If scikit-learn >= 1.0, an SVC will have feature_names_in_.
    #    If you get AttributeError, see below for solutions.
    model_cols = svm_model.column_list
    df_encoded = df_encoded.reindex(columns=model_cols, fill_value=0)

    # 10. Predict the class and probabilities
    y_pred = svm_model.predict(df_encoded)
    y_probs = svm_model.predict_proba(df_encoded)

    classes_order = svm_model.classes_  # e.g. [0, 2, 1]
    probs = y_probs[0]                  # e.g. [0.526, 0.091, 0.383]

    # Find the index of the highest probability
    idx_max = np.argmax(probs)          # e.g. 0 if 0.526 is largest
    predicted_class = classes_order[idx_max]  
    # e.g. if classes_order=[0,2,1], idx_max=0 => predicted_class=0

    # For clarity, map 0->"Home Win", 1->"Draw", 2->"Away Win"
    class_mapping = {
        0: "Home Win",
        1: "Draw",
        2: "Away Win"
    }

    prediction_str = class_mapping[predicted_class]

    # Unpack probabilities into home/draw/away 
    # by reading them from class_proba_dict or a direct assignment:
    #   - if classes_order[0] = 0 (Home), classes_order[1] = 1 (Draw), ...
    # But let's build a dictionary so we don't mismatch:

    class_proba_dict = dict(zip(classes_order, probs))
    home_prob = class_proba_dict[0]
    draw_prob = class_proba_dict[1]
    away_prob = class_proba_dict[2]

    # 12. Convert that prediction_str to "Arsenal Win" or "Fulham Win"
    if prediction_str == "Home Win":
        nice_prediction = f"{home_team} Win"
    elif prediction_str == "Away Win":
        nice_prediction = f"{away_team} Win"
    else:
        nice_prediction = "Draw"

    # 13. Build a readable result message
    result_message = (
        f"Match: {home_team} vs {away_team}\n"
        f"Prediction: {nice_prediction}"
    )

    return nice_prediction, result_message, home_prob, draw_prob, away_prob