import logging

from prediction import get_columns_classes, classes2types, probabilities2scored_types

logger = logging.getLogger()


def routine_ml(csv_detective_results, file_path, model_ml, num_rows=500, return_probabilities=False):
    assert model_ml is not None
    y_true = model_ml.classes_
    y_pred, y_pred_proba, csv_info = get_columns_classes(csv_path=file_path, model=model_ml,
                                                         csv_metadata=csv_detective_results,
                                                         num_rows=num_rows,
                                                         return_probas=return_probabilities)

    if return_probabilities and y_pred_proba.size > 0:
        csv_detective_results["columns_ml_probas"] = probabilities2scored_types(y_true, y_pred_proba, csv_info)
    if y_pred.size > 0:
        csv_detective_results["columns_ml"] = classes2types(y_pred, csv_info)
    return csv_detective_results


def join_reports(dict_rb: dict, dict_ml: dict):
    """
    Aligns both results from the rule based system with the machine learning system into a
    single dict. This new dict is a dict of lists, where the outer keys are the types. Each type has
    as value a list of dicts with the info about each column in the csv. Like so:

    "code_departement": [
      {
        "colonne": "dep",
        "score_rb": 0.6,
        "score_ml": 0.6
      },
      {
        "colonne": "reg",
        "score_rb": 0.5,
        "score_ml": 0.35
      },
      {
        "colonne": "assoc",
        "score_rb": 0.2,
        "score_ml": 0.05
      }
    ],

    Parameters
    ----------
    dict_rb Dict with the rb results
    dict_ml Dict with the ml results

    Returns
    -------
    A single dict with both results combined.
    """
    all_types = set(dict_rb.keys()).union(dict_ml.keys())
    full_report = {}
    for t in all_types:
        rb_list, ml_list = [], []
        if t in dict_rb:
            rb_list = dict_rb[t]
        if t in dict_ml:
            ml_list = dict_ml[t]
        if rb_list and ml_list:
            both_lists = []
            # join lists
            for col_dict_rb in rb_list:
                col_name_rb = col_dict_rb["colonne"]
                for col_dict_ml in ml_list:
                    if col_dict_ml["colonne"].lower() == col_name_rb.lower():
                        col_dict_both = {**col_dict_ml, **col_dict_rb}
                        both_lists.append(dict(col_dict_both))
                        break
            full_report[t] = both_lists
        else:
            full_report[t] = rb_list or ml_list
            # TODO: fix this, should be dealt with at csv_detective!  dropping unnamed columns :/
            filtered_list = []
            for d in full_report[t]:
                if "Unnamed" not in d["colonne"]:
                    filtered_list.append(d)

            full_report[t] = filtered_list
            # add empty values to score_ml/score_rb
            completed_list = []
            for d in full_report[t]:
                if "score_ml" in d:
                    d["score_rb"] = 0.0
                else:
                    d["score_ml"] = 0.0
                completed_list.append(dict(d))
            full_report[t] = completed_list
    return full_report
