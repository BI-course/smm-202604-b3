import joblib, warnings
warnings.filterwarnings("ignore")
for f in ["lasso_regressor_for_sme_revenue", "svc_classifier_for_sme_credit_risk"]:
    print("==", f)
    p = joblib.load(f"model/{f}.joblib")
    ct = [s for _, s in p.steps if hasattr(s, "transformers_")][0]
    for _, trans, cols in ct.transformers_:
        for _, s in (trans.steps if hasattr(trans, "steps") else [(None, trans)]):
            if hasattr(s, "categories_"):
                for c, cats in zip(cols, s.categories_):
                    print(f"  {c}: {list(cats)}")
