"""
api.py

A RESTish API, built with Flask, that serves FOUR machine learning artifacts
produced in earlier labs:

    1. Regression       -- SME monthly revenue prediction
    2. Classification   -- SME credit risk category
    3. Clustering       -- Mall customer segment assignment
    4. Association Rule -- Product recommendation from a cart
       Mining (ARM)

The point of this file is to show you that a fitted scikit-learn /
imbalanced-learn Pipeline is a SINGLE object that already knows how to impute
missing values, encode categorical columns, scale numeric columns, and (for
the classifier) apply SMOTE only during training -- never during inference.
Because all of that preprocessing lives inside the Pipeline itself, this file
never manually label-encodes anything, never manually engineers date features,
and never manually reorders columns before calling .predict(). It builds a
DataFrame straight from the incoming JSON and hands it to the pipeline as-is.
This is the main lesson of the lab: "the pipeline IS the deployment artifact."

Run this file directly to start a local development server:

    python api.py
"""

import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)

# -----------------------------------------------------------------------
# CORS (Cross-Origin Resource Sharing)
# -----------------------------------------------------------------------
# By default, a browser blocks JavaScript on one origin (e.g., a page served
# from http://127.0.0.1:5500) from calling an API on a different origin
# (e.g., http://127.0.0.1:5000). `flask_cors` relaxes that restriction for the
# specific origins we trust during local development.
#
# A few alternative configurations are shown below (commented out) so you can
# see how the restriction can be loosened or tightened depending on your
# deployment target. Only ONE of these should be active at a time.

# Option A (the most permissive. This is fine for a quick classroom demo or
# local development, *NEVER* for production):
# CORS(app)

# Option B (allow only a specific frontend origin):
# CORS(app, resources={r"/api/*": {"origins": "http://127.0.0.1:5500"}})

# Option C (the one actually in use here -- allow localhost/127.0.0.1 on the
# ports our lab's frontend files are typically served from):
CORS(app, resources={r"/api/*": {"origins": [
    "http://localhost:5000", "http://127.0.0.1:5000",
    "http://localhost:5500", "http://127.0.0.1:5500",
    "https://localhost:443", "https://127.0.0.1:443",
    "http://localhost:63342",  # VS Code Live Server extension
    "null",  # allows opening a frontend .html file directly (file://) with no web server
]}})

# -----------------------------------------------------------------------
# The best practice is to load the four persisted artifacts ONCE, at startup,
# not per-request.
# Loading inside a route function would re-read the file from disk on every
# single API call, which is slow and pointless -- the fitted model never
# changes while the server is running.
# -----------------------------------------------------------------------
MODEL_DIR = "./model"

regression_pipeline = joblib.load(
    os.path.join(MODEL_DIR, "lasso_regressor_for_sme_revenue.joblib")
)

classification_pipeline = joblib.load(
    os.path.join(MODEL_DIR, "svc_classifier_for_sme_credit_risk.joblib")
)

clustering_artifact = joblib.load(
    os.path.join(MODEL_DIR, "kmeans_mall_customer_segmentation.joblib")
)
clustering_scaler = clustering_artifact["scaler"]
clustering_kmeans = clustering_artifact["kmeans"]

arm_artifact = joblib.load(
    os.path.join(MODEL_DIR, "apriori_recommendation_rules.joblib")
)
recommendation_lookup = arm_artifact["recommendation_lookup"]
sku_lookup = arm_artifact["sku_lookup"]

print("All the four artifacts have been loaded. The API is ready.")


# --- MEMBER 1: REGRESSION ROUTE ---
@app.route("/api/v1/models/regression/predict", methods=["POST"])
def predict_regression():
    try:
        # 1. Get JSON data from the request
        data = request.get_json()
        
        # 2. Convert to DataFrame (make sure it's a list of dicts for pandas)
        input_df = pd.DataFrame([data])
        
        # 3. Make the prediction using the CORRECT variable name (regression_pipeline)
        prediction = regression_pipeline.predict(input_df)
        
        # 4. Extract the result (if it returns an array, get the first item)
        pred_value = prediction[0]
        
        # 5. Return as JSON
        return jsonify({
            "status": "success",
            "model": "SME Revenue Regressor",
            "prediction": float(pred_value) 
        }), 200

    except Exception as e:
        # Return an error if the user sends bad data
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


# --- MEMBER 2: CLASSIFICATION ROUTE ---
@app.route("/api/v1/models/classification/predict", methods=["POST"])
def predict_classification():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])

        # Make prediction
        prediction = classification_pipeline.predict(input_df)

        # Extract the result (it is already a string like 'High' or 'Low')
        pred_class = prediction[0]

        # Return JSON (REMOVE the int() wrapper here!)
        return jsonify({
            "status": "success",
            "model": "SME Credit Risk Classifier",
            "prediction": str(pred_class)  # Convert to string just to be safe
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

# 3. # --- MEMBER 3: CLUSTERING ROUTE ---
@app.route("/api/v1/models/clustering/predict", methods=["POST"])
def predict_clustering():
    try:
        # 1. Get JSON data
        data = request.get_json()
        
        # 2. Convert to DataFrame
        input_df = pd.DataFrame([data])
        
        # 3. CRITICAL STEP: Scale the data using the loaded scaler
        # K-Means uses Euclidean distance, so features must be on the same scale!
        scaled_data = clustering_scaler.transform(input_df)
        
        # 4. Make the prediction using the scaled data
        cluster_label = clustering_kmeans.predict(scaled_data)
        
        # 5. Extract the result (e.g., 0, 1, 2, 3, or 4)
        cluster_id = int(cluster_label[0])
        
        # 6. Return JSON
        return jsonify({
            "status": "success",
            "model": "Mall Customer Segmentation",
            "cluster_id": cluster_id
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


# --- MEMBER 4: RECOMMENDER ROUTE ---

@app.route("/api/v1/models/recommender/predict", methods=["POST"])
def predict_recommender():
    try:
        # 1. Get the input basket from the JSON payload
        data = request.get_json()
        basket = data.get("basket", [])

        if not basket:
            return jsonify({
                "status": "error",
                "message": "Basket is empty"
            }), 400

        # 2. Collect recommendations from the pre-loaded lookup table
        recommendations = {}

        basket_set = set(basket)
        for antecedent, recs in recommendation_lookup.items():
            # Keys are frozensets of items; match when all are in the basket
            if antecedent.issubset(basket_set):

                # each rec is (recommended_item, confidence, lift)
                for rec_item, *_, lift_score in recs:

                    # Don't recommend something they already have in the basket
                    if rec_item not in basket:

                        # Keep the highest lift score if multiple items
                        # recommend the same thing
                        if (
                            rec_item not in recommendations
                            or lift_score > recommendations[rec_item]
                        ):
                            recommendations[rec_item] = lift_score

        # 3. Sort the recommendations by lift score (highest first)
        sorted_recs = sorted(
            recommendations.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # 4. Get the top 3 recommendations
        top_3 = [rec[0] for rec in sorted_recs[:3]]

        # 5. Return JSON
        return jsonify({
            "status": "success",
            "model": "Pharmacy Recommender",
            "input_basket": basket,
            "recommendations": top_3
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/ecommerce-recommender/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"cart": ["H06", "H07"]}'
#
#   curl -X POST http://127.0.0.1:5000/api/v1/models/ecommerce-recommender/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"cart": ["P04", "P01"]}'
#
# Note: there is no scikit-learn ".predict()" here -- the mined association
# rules were precomputed into a lookup dictionary
# (antecedent SKU-set -> ranked candidate consequents)
# during the notebook's persistence stage. This
# endpoint performs the exact same lookup+rank step a production recommender
# microservice would perform on every incoming request; it does not mine
# for the rules again.
@app.route("/api/v1/models/ecommerce-recommender/predictions", methods=["POST"])
def recommend_products():
    payload = request.get_json()
    cart = set(payload.get("cart", []))
    top_n = int(payload.get("top_n", 3))

    candidates = {}
    for antecedent, consequent_list in recommendation_lookup.items():
        if antecedent.issubset(cart):
            for sku, confidence, lift in consequent_list:
                if sku not in cart:
                    if sku not in candidates or confidence > candidates[sku][0]:
                        candidates[sku] = (confidence, lift)

    ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)[:top_n]
    recommendations = [
        {
            "sku": str(sku),
            "product_name": sku_lookup.loc[sku, "product_name"],
            "confidence": round(float(confidence), 3),
            "lift": round(float(lift), 2),
        }
        for sku, (confidence, lift) in ranked
    ]

    return jsonify({
        "cart": sorted(cart),
        "recommendations": recommendations,
        "model": "Apriori association rules, cleaned + precomputed lookup",
    })


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------
if __name__ == "__main__":
    # Development server. debug=True gives auto-reload and an interactive
    # debugger on errors -- convenient in class, but NEVER use debug=True in
    # a deployed/public environment (it can expose a remote code execution
    # console to anyone who can reach the server).
    app.run(debug=True)

    # Production-style alternative (no debugger, no auto-reload):
    # app.run(debug=False, host="0.0.0.0", port=5000)

    # HTTPS/TLS alternative for local development (requires a self-signed
    # certificate; see app_server_reverse_proxy_server_setup.md):
    # app.run(debug=False, ssl_context=("cert.pem", "key.pem"))
