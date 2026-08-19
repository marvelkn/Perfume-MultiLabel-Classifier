import pickle

labels = ['floral', 'woody', 'citrus', 'fruity', 'sweet']
print(f"{'Label':<18} | {'n_estimators':<12} | {'max_depth':<9} | {'learning_rate':<13} | {'reg_alpha':<9} | reg_lambda")
print('-' * 90)
for label in labels:
    path = f'models/xgb_models/xgb_{label}.pkl'
    with open(path, 'rb') as f:
        model = pickle.load(f)
    p = model.get_params()
    print(f"{label:<18} | {p['n_estimators']:<12} | {p['max_depth']:<9} | {p['learning_rate']:<13.4f} | {str(p.get('reg_alpha','N/A')):<9} | {p.get('reg_lambda','N/A')}")
