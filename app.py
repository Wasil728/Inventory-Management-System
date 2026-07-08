import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from paths import DATA_PATH, DATA_SET_DIR, MODEL_PATH, STATIC_DIR
from utils import generate_inventory_report, get_low_stock_products, get_near_expiry_products

# Short header label and full product name (override with APP_NAME / APP_FULL_NAME).
APP_NAME = os.environ.get("APP_NAME", "IMS")
APP_FULL_NAME = os.environ.get("APP_FULL_NAME", "Inventory Management System")

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")


def page(template: str, *, nav_active: str = "", **kwargs):
    """Render HTML with shared layout context."""
    ctx = {
        "app_name": APP_NAME,
        "app_full_name": APP_FULL_NAME,
        "nav_active": nav_active,
        **kwargs,
    }
    return render_template(template, **ctx)

# Configuration (paths.py uses /tmp on Vercel for uploads and model I/O)
app.config["UPLOAD_FOLDER"] = DATA_SET_DIR
app.config["MODEL_PATH"] = MODEL_PATH
app.config["DATA_PATH"] = DATA_PATH
# Limit uploads to 5MB
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

# Load the pickled model
# The pickle is expected to be a dict: {"model": ..., "scaler": ..., "meta": {...}}
model_data = None


def load_trained_model():
    """Load the trained model with proper error handling. Returns dict or None."""
    global model_data
    try:
        if not os.path.exists(app.config['MODEL_PATH']):
            print(f"Model file not found at {app.config['MODEL_PATH']}")
            return None

        with open(app.config['MODEL_PATH'], 'rb') as model_file:
            md = pickle.load(model_file)

        # md should be a dict containing keys 'model', optionally 'scaler' and 'meta'
        if isinstance(md, dict) and 'model' in md:
            model_data = md
            print("Model data loaded successfully!")
            return model_data

        # Backwards compatibility: a raw sklearn estimator
        if hasattr(md, 'predict'):
            model_data = {'model': md, 'scaler': None, 'meta': {}}
            print("Loaded raw model (no meta)")
            return model_data

        print("Loaded data is not a valid model format")
        return None
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None


# Try to load model on startup
model_data = load_trained_model()


def simple_prediction(quantity1, quantity2, quantity3):
    """Simple prediction using weighted average"""
    try:
        weights = [0.2, 0.3, 0.5]
        prediction = (quantity1 * weights[0] + quantity2 * weights[1] + quantity3 * weights[2])
        return prediction
    except Exception as e:
        print(f"Simple prediction error: {str(e)}")
        return None


@app.route('/')
def home():
    return page("index.html", nav_active="home")


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload with validation before saving"""
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "No selected file"}), 400

        if not file.filename.endswith('.csv'):
            return jsonify({"success": False, "error": "Please upload a CSV file"}), 400

        # Read into pandas to validate columns before saving
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return jsonify({"success": False, "error": f"Invalid CSV: {str(e)}"}), 400

        from utils import validate_csv_data
        valid, msg = validate_csv_data(df)
        if not valid:
            return jsonify({"success": False, "error": msg}), 400

        # Save the file to the data directory
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'data.csv')
        df.to_csv(file_path, index=False)

        return jsonify({"success": True, "message": "File uploaded and validated successfully!"}), 200

    except Exception as e:
        print(f"Upload error: {str(e)}")
        return jsonify({"success": False, "error": f"Error saving file: {str(e)}"}), 500


@app.route('/inventory')
def inventory():
    """Display inventory with restocking and expiry recommendations"""
    try:
        if not os.path.exists(app.config['DATA_PATH']):
            return page("error.html", nav_active="", error="No data file. Upload a CSV from Home first.")

        df = pd.read_csv(app.config['DATA_PATH'])
        low_stock_recommendations = get_low_stock_products(df)
        near_expiry_recommendations = get_near_expiry_products(df)
        from utils import calculate_inventory_metrics
        metrics = calculate_inventory_metrics(df)

        return page(
            "inventory.html",
            nav_active="inventory",
            restock_recommendations=low_stock_recommendations,
            near_expiry_recommendations=near_expiry_recommendations,
            metrics=metrics,
        )
    except Exception as e:
        return page("error.html", nav_active="", error=f"Inventory error: {str(e)}")


@app.route('/predict', methods=["GET", "POST"])
def predict():
    """Handle prediction requests using trained model + scaler (if present)"""
    global model_data
    if request.method == "POST":
        try:
            data = request.get_json()
            if not data:
                return jsonify({"success": False, "error": "No data provided"}), 400

            # Determine expected time_steps
            time_steps = 3
            model_meta = model_data.get('meta') if model_data else None
            if model_meta and isinstance(model_meta, dict):
                time_steps = int(model_meta.get('time_steps', 3))

            # Collect quantities quantity1..quantityN
            quantities = []
            for i in range(time_steps):
                key = f'quantity{i+1}'
                if key not in data:
                    return jsonify({"success": False, "error": f"Missing field: {key}. Expected {time_steps} quantities."}), 400
                q = float(data.get(key, 0))
                if q < 0:
                    return jsonify({"success": False, "error": "Quantities must be non-negative"}), 400
                quantities.append(q)

            # If we have a trained model, use it
            if model_data and 'model' in model_data:
                try:
                    model = model_data['model']
                    scaler = model_data.get('scaler')

                    X = np.array([quantities])  # shape (1, time_steps)
                    if scaler is not None:
                        try:
                            X_proc = scaler.transform(X)
                        except Exception:
                            # scaler might expect a different shape — attempt reshape
                            X_proc = scaler.transform(np.asarray(X).reshape(1, -1))
                    else:
                        X_proc = X

                    pred = model.predict(X_proc)
                    # model.predict may return scalar-like arrays
                    if hasattr(pred, '__iter__'):
                        prediction_value = float(np.ravel(pred)[0])
                    else:
                        prediction_value = float(pred)

                    resp = {
                        "success": True,
                        "prediction": prediction_value,
                        "method": "model",
                        "model_type": model_data.get('meta', {}).get('model_type') if model_data.get('meta') else None,
                        "train_mae": model_data.get('meta', {}).get('train_mae') if model_data.get('meta') else None,
                        "test_mae": model_data.get('meta', {}).get('test_mae') if model_data.get('meta') else None,
                        "time_steps": model_data.get('meta', {}).get('time_steps') if model_data.get('meta') else time_steps,
                    }
                    return jsonify(resp)
                except Exception as model_error:
                    print(f"Model prediction failed: {str(model_error)}")
                    # Fall back to simple prediction
                    pass

            # Fallback
            if len(quantities) >= 3:
                prediction_value = simple_prediction(quantities[-3], quantities[-2], quantities[-1])
            else:
                # If fewer than 3, pad with zeros
                padded = [0, 0, 0]
                for i in range(len(quantities)):
                    padded[-len(quantities) + i] = quantities[i]
                prediction_value = simple_prediction(padded[-3], padded[-2], padded[-1])

            if prediction_value is not None:
                return jsonify({
                    "success": True,
                    "prediction": float(prediction_value),
                    "method": "simple_weighted_average"
                })
            else:
                return jsonify({"success": False, "error": "Failed to make prediction"}), 500

        except ValueError as e:
            return jsonify({"success": False, "error": f"Invalid input data: {str(e)}"}), 400
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return jsonify({"success": False, "error": f"Failed to make prediction: {str(e)}"}), 500

    elif request.method == "GET":
        return page("prediction.html", nav_active="predict", model_loaded=bool(model_data))


@app.route('/analytics')
def sales_analytics():
    """Display sales analytics with improved error handling"""
    try:
        if not os.path.exists(app.config['DATA_PATH']):
            return page("error.html", nav_active="", error="No data file. Upload a CSV from Home first.")

        data = pd.read_csv(app.config['DATA_PATH'])
        total_sales = float(data["total_revenue"].sum())
        average_order_value = float(data["total_revenue"].mean())
        top_selling_products = data.nlargest(5, "quantity_stock")
        bottom_selling_products = data.nsmallest(5, "quantity_stock")

        top_selling_dict = []
        for _, row in top_selling_products.iterrows():
            top_selling_dict.append({
                'product_id': int(row['product_id']),
                'product_name': str(row['product_name']),
                'quantity_stock': int(row['quantity_stock']),
                'total_revenue': float(row['total_revenue'])
            })

        bottom_selling_dict = []
        for _, row in bottom_selling_products.iterrows():
            bottom_selling_dict.append({
                'product_id': int(row['product_id']),
                'product_name': str(row['product_name']),
                'quantity_stock': int(row['quantity_stock']),
                'total_revenue': float(row['total_revenue'])
            })

        row_count = int(len(data))
        chart_labels = [str(x) for x in data["product_id"].tolist()]
        chart_revenue = [float(x) for x in data["total_revenue"].tolist()]

        return page(
            "analytics.html",
            nav_active="analytics",
            total_sales=total_sales,
            average_order_value=average_order_value,
            row_count=row_count,
            top_selling_products=top_selling_dict,
            bottom_selling_products=bottom_selling_dict,
            chart_labels=chart_labels,
            chart_revenue=chart_revenue,
        )
    except Exception as e:
        print(f"Analytics error: {str(e)}")
        return page("error.html", nav_active="", error=f"Analytics error: {str(e)}")


@app.route('/train', methods=['POST'])
def train_model_route():
    """Train the prediction model"""
    try:
        from Prediction import main as train_prediction_model
        print("Starting model training process...")
        success = train_prediction_model()

        if success:
            global model_data
            model_data = load_trained_model()
            if model_data is not None:
                return jsonify({"success": True, "message": "Model trained successfully!"}), 200
            else:
                return jsonify({"success": False, "error": "Model training completed but failed to load the model."}), 500
        else:
            return jsonify({"success": False, "error": "Model training failed. Check the logs for details."}), 500
    except Exception as e:
        print(f"Training error: {str(e)}")
        return jsonify({"success": False, "error": f"Error training model: {str(e)}"}), 500


@app.route('/api/inventory-summary')
def inventory_summary():
    """API endpoint for inventory summary"""
    try:
        print(f"Checking for data file at: {app.config['DATA_PATH']}")

        if not os.path.exists(app.config['DATA_PATH']):
            print("Data file not found, returning default metrics")
            return jsonify({
                "metrics": {
                    "total_products": 0,
                    "low_stock_count": 0,
                    "average_stock_level": 0,
                    "total_stock_value": 0,
                    "near_expiry_count": 0,
                    "total_revenue": 0,
                    "average_order_value": 0
                },
                "message": "No data file found. Please upload a CSV file first."
            }), 200

        print("Data file found, generating report...")
        report = generate_inventory_report(app.config['DATA_PATH'])

        if report:
            print(f"Report generated successfully: {report}")
            return jsonify(report)
        else:
            print("Failed to generate report")
            return jsonify({
                "metrics": {
                    "total_products": 0,
                    "low_stock_count": 0,
                    "average_stock_level": 0,
                    "total_stock_value": 0,
                    "near_expiry_count": 0,
                    "total_revenue": 0,
                    "average_order_value": 0
                },
                "error": "Failed to generate report"
            }), 200
    except Exception as e:
        print(f"Error in inventory summary: {str(e)}")
        return jsonify({
            "metrics": {
                "total_products": 0,
                "low_stock_count": 0,
                "average_stock_level": 0,
                "total_stock_value": 0,
                "near_expiry_count": 0,
                "total_revenue": 0,
                "average_order_value": 0
            },
            "error": str(e)
        }), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
    
