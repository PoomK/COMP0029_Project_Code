import pandas as pd
import numpy as np
import pickle
import os
import tensorflow as tf

def predict_mlp(home_team: str, away_team: str, gw: int):
    # 1. Load the one-hot encoder (the same one used in training the MLP)
    with open("../Pre-Processing/Encoders_final/onehot_encoder.pkl", "rb") as f:
        ohe = pickle.load(f)

    # 2. Load your trained MLP model (Keras format, with probability outputs from final Softmax)
    mlp_model = tf.keras.models.load_model("../Models/saved_models_result/mlp_result_model.h5")

    # 3. Load the scaler used for numeric columns
    #    (e.g., StandardScaler, MinMaxScaler, etc. saved via pickle)
    with open("../Models/saved_models_result/mlp_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # 4. Define your home/away mappings (same as your logistic regression/SVM code)
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
        "expected_goals": "Home_expected_goals",
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
        "expected_goals": "Away_expected_goals",
    }

    # 5. Load the per-team stats for the specified gameweek
    team_df = pd.read_csv(
        f"Prediction_Features/Team_Form_And_Averages_2024_25_gw_{gw}.csv",
        index_col="Team"
    )

    # 6. Construct one row of data
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

    # 7. One-hot encode the categorical columns (Home Team, Away Team)
    categorical_cols = ["Home Team", "Away Team"]
    df_cat = df_input[categorical_cols]
    arr_encoded = ohe.transform(df_cat)  # transform, not fit_transform
    encoded_cols = ohe.get_feature_names_out(categorical_cols)
    df_cat_encoded = pd.DataFrame(arr_encoded, columns=encoded_cols, index=df_input.index)

    # 8. Combine numeric + encoded columns
    df_num = df_input.drop(columns=categorical_cols)
    df_encoded = pd.concat([df_num, df_cat_encoded], axis=1)

    # 9. Make sure columns are in the same order as training
    #    If you have a stored column list (like 'mlp_columns.pkl'), load it and reindex:
    #    For example:
    #    with open("mlp_columns.pkl","rb") as f:
    #        model_cols = pickle.load(f)
    #    OR if your MLP approach was exactly the same as logistic regression & SVM, you can reuse that column list.
    #    We'll assume you have "mlp_model.column_list" or a known list:
    with open("../Models/saved_models_result/logistic_regression_model_l2.pkl", "rb") as f:
        logreg_model = pickle.load(f)
    model_cols = logreg_model.feature_names_in_
    df_encoded = df_encoded.reindex(columns=model_cols, fill_value=0)

    # 10. Scale the numeric data with the same scaler used in training
    X_scaled = scaler.transform(df_encoded)

    # 11. Get MLP predictions
    #     For a 3-class problem with softmax output, shape -> (1,3)
    y_probs = mlp_model.predict(X_scaled)[0]  # e.g. [0.65, 0.20, 0.15]

    # 12. Identify the predicted class by highest probability
    idx_max = np.argmax(y_probs)  # 0,1,2
    # For clarity, assume we used: 0 -> Home Win, 1 -> Draw, 2 -> Away Win
    class_mapping = {
        0: "Home Win",
        1: "Draw",
        2: "Away Win"
    }
    prediction_str = class_mapping[idx_max]

    # 13. Map that to "Arsenal Win" or "Fulham Win" or "Draw"
    if prediction_str == "Home Win":
        nice_prediction = f"{home_team} Win"
    elif prediction_str == "Away Win":
        nice_prediction = f"{away_team} Win"
    else:
        nice_prediction = "Draw"

    # 14. Build a readable result message
    result_message = (
        f"Match: {home_team} vs {away_team}\n"
        f"Prediction: {nice_prediction}"
    )

    # 15. Extract probabilities in a stable order: 0->home, 1->draw, 2->away
    #     y_probs is in [home_prob, draw_prob, away_prob] if your model training 
    #     used that same order consistently. Confirm or rearrange as needed.
    home_prob = y_probs[0]
    draw_prob = y_probs[1]
    away_prob = y_probs[2]

    # 16. Return the same tuple you return in your other predict functions
    return nice_prediction, result_message, home_prob, draw_prob, away_prob
