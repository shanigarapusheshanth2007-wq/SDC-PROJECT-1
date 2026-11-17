import pandas as pd
import numpy as np

def explain_prediction(input_data, model, X_train):
    original_prob = model.predict_proba(input_data)[0]
    class_index = list(model.classes_).index(1)
    original_approval = original_prob[class_index]
    influences = []
    
    for col in input_data.columns:
        modified_input = input_data.copy()
        modified_input[col] = X_train[col].median()
        modified_prob = model.predict_proba(
            modified_input
        )[0][class_index]
        impact = original_approval - modified_prob
        influences.append({
            "Feature": col,
            "Input_Value": input_data[col].iloc[0],
            "Impact": float(impact)
        })
    result = pd.DataFrame(influences)
    result["Absolute_Impact"] = result["Impact"].abs()
    total = result["Absolute_Impact"].sum()
    if total != 0:
        result["Influence_Percentage"] = (
            result["Absolute_Impact"] / total * 100
        )
    else:
        result["Influence_Percentage"] = 0
    result = result.sort_values(
        by="Absolute_Impact",
        ascending=False
    )
    return result