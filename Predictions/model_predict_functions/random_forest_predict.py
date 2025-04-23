import pandas as pd
import numpy as np
import pickle
import os

def predict_rf(home_team: str, away_team: str, gw: int):
  with open("../Pre-Processing/Encoders_final/Home_Team_label_encoder.pkl", "rb") as f:
      le_home = pickle.load(f)

  with open("../Pre-Processing/Encoders_final/Away_Team_label_encoder.pkl", "rb") as f:
      le_away = pickle.load(f)

  with open("../Models/saved_models_result/random_forest_model.pkl", "rb") as f:
      rf_model = pickle.load(f)
  
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
  
  team_df = pd.read_csv(f"Prediction_Features/Team_Form_And_Averages_2024_25_gw_{gw}.csv", index_col="Team")
  
  row = {
    "Home Team": home_team,
    "Away Team": away_team,
  }
  
  home_row = team_df.loc[home_team]
  away_row = team_df.loc[away_team]

  for src_col, dst_col in home_mapping.items():
      row[dst_col] = home_row[src_col]

  for src_col, dst_col in away_mapping.items():
      row[dst_col] = away_row[src_col]
  
  df_input = pd.DataFrame([row])
  
  model_cols = rf_model.feature_names_in_
  df_input = df_input.reindex(columns=model_cols, fill_value=0)
  
  y_pred = rf_model.predict(df_input)
  y_probs = rf_model.predict_proba(df_input)  # If you want the probability distribution

  # 10. Map your integer predictions to a readable label
  class_mapping = {
      0: "Home Win",
      1: "Draw",
      2: "Away Win"
  }
  home_prob, draw_prob, away_prob = y_probs[0]
  prediction_label = y_pred[0]
  prediction_str = class_mapping[prediction_label]

  # 11. Make it "nice"
  if prediction_str == "Home Win":
      nice_prediction = f"{home_team} Win"
  elif prediction_str == "Away Win":
      nice_prediction = f"{away_team} Win"
  else:
      nice_prediction = "Draw"

  # 12. Build a result message
  result_message = (
      f"Match: {home_team} vs {away_team}\n"
      f"Prediction: {nice_prediction}"
  )

  # 13. Return desired outputs
  return nice_prediction, result_message, home_prob, draw_prob, away_prob