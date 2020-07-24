from prediction import get_columns_ML_prediction, get_columns_types

#
# def routine_ml(csv_detective_results, file_path, model_ml, num_rows=500):
#     try:
#         assert model_ml is not None
#         dict_result = {}
#         y_pred, y_pred_proba, csv_info = get_columns_ML_prediction(file_path, model_ml, csv_detective_results,
#                                                      num_rows=num_rows)
#
#         dict_result["columns_ml"] = get_columns_types(y_pred, csv_info)
#
#     except:
#         pass