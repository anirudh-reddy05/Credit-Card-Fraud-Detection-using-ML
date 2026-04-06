import os
import sys
from datetime import datetime
sys.path.append("XBNet")

from . import db
from flask import Blueprint, current_app, render_template, request, redirect, url_for, flash, abort
import traceback
from flask_login import login_required, current_user
from sqlalchemy import func
from .models import User, PredictionLog
from functools import wraps
from werkzeug.security import generate_password_hash
import joblib
import pandas as pd
import torch
import torch.nn.functional as F
from XBNet.training_utils import predict, predict_proba
import numpy as np


views = Blueprint('views', __name__)


# @views.route('/selection', methods=['GET', 'POST'])
# def selection():
#     return render_template("selection.html")


# @views.route('/index.html', methods=['GET', 'POST'])
# @login_required
# def index():
#     X = pd.DataFrame()
#     if request.method == "POST":
#         get_model = joblib.load("app/xbnet_models/model.pkl")
#         time = request.form.get("time")
#         v1 = request.form.get("v1")
#         v2 = request.form.get("v2")
#         v3 = request.form.get("v3")
#         v4 = request.form.get("v4")
#         v5 = request.form.get("v5")
#         amount = request.form.get("amount")
#         X = pd.DataFrame([[time, v1, v2, v3, v4, v5, amount]], columns = ["Time", "V1", "V2", "V3", "V4", "V5", "Amount"]).astype(float)
#         X.to_csv('user_data.csv', index=None)
#         prediction = predict(get_model,X.to_numpy()[0,:])
#         return redirect(url_for('prediction_output', pred=prediction))
#     else:
#         prediction = ""
#     return render_template("index.html")

# Flask server-side code

@views.route('/upload_page.html', methods=['GET', 'POST'])
@login_required
def upload_page():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('upload_page.html', error="No file selected")
            
        file = request.files['file']
        
        # Check if a file was selected
        if file.filename == '':
            return render_template('upload_page.html', error="No file selected")
            
        # Check if it's a CSV file
        if not file.filename.lower().endswith('.csv'):
            return render_template('upload_page.html', error="Please upload a CSV file")
            
        try:
            # Try to read the CSV file with pandas
            df = pd.read_csv(file)
            
            # Verify required core columns are present
            required_core_columns = [
                'amt', 'zip', 'lat', 'long', 'genders', 
                'Essentials', 'Leisure', 'Wellness', 'Other'
            ]
            
            missing_cols = [col for col in required_core_columns if col not in df.columns]
            if missing_cols:
                return render_template('upload_page.html', 
                    error=f"Missing required columns: {', '.join(missing_cols)}")
                
            # Check data types
            try:
                df['amt'] = pd.to_numeric(df['amt'])
                df['zip'] = df['zip'].astype(str)  # ZIP should be string to preserve leading zeros
                df['lat'] = pd.to_numeric(df['lat'])
                df['long'] = pd.to_numeric(df['long'])
                df[['genders', 'Essentials', 'Leisure', 'Wellness', 'Other']] = \
                    df[['genders', 'Essentials', 'Leisure', 'Wellness', 'Other']].astype(int)
                    
                # Add any missing state columns with 0s
                state_columns = ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL',
                               'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA',
                               'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE',
                               'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI',
                               'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']
                               
                for state in state_columns:
                    if state not in df.columns:
                        df[state] = 0
                    else:
                        df[state] = df[state].astype(int)
            except ValueError as e:
                return render_template('upload_page.html', 
                    error="Invalid data types in columns. Please ensure numeric values are provided.")
            
            # Save the validated file
            save_path = os.path.join(current_app.instance_path, "user_data.csv")
            df.to_csv(save_path, index=False)
            
            return redirect(url_for('views.prediction_output'))
            
        except Exception as e:
            return render_template('upload_page.html', 
                error=f"Error processing file: {str(e)}. Please ensure the CSV file matches the required format.")

    return render_template('upload_page.html')
@views.route('/prediction_output/table')
@login_required
def prediction_output():
    try:
        # Load the file using pandas
        X = pd.read_csv(os.path.join(current_app.instance_path, "user_data.csv"))

        # --- existing prediction logic follows ---

        # Select only the required features in the correct order
        required_features = ['amt', 'zip', 'lat', 'long', 'genders',
                            'Essentials', 'Leisure', 'Wellness', 'Other',
                            'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL',
                            'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA',
                            'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE',
                            'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI',
                            'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']

        # Ensure all state columns exist, if not create them with 0s
        for state in required_features[9:]:  # Skip non-state columns
            if state not in X.columns:
                X[state] = 0

        # Select and order columns properly
        X = X[required_features]

        # Unpickle the classifier
        # Ensure the original module used when the model was pickled is importable
        try:
            import importlib
            # Preferred modern package import
            models_xb_mod = importlib.import_module('XBNet.models_xb')
            # Some models were pickled with module name 'models_xb' (no package)
            # so make sure that name is available in sys.modules for unpickling
            sys.modules['models_xb'] = models_xb_mod
        except Exception:
            try:
                # Fallback: try to import module as top-level if present
                import models_xb as models_xb_mod
                sys.modules['models_xb'] = models_xb_mod
            except Exception:
                # If import fails, continue and let joblib.load raise a clear error
                pass

        get_model = joblib.load(os.path.join(current_app.instance_path, "v2.pkl"))

        # Try to auto-load a saved preprocessor (scaler/pipeline) if present in instance/
        preprocessor = None
        for name in ('preprocessor.pkl', 'scaler.pkl', 'pipeline.pkl', 'transformer.pkl', 'preproc.pkl'):
            path = os.path.join(current_app.instance_path, name)
            if os.path.exists(path):
                try:
                    preprocessor = joblib.load(path)
                    print(f"Loaded preprocessor from {path}")
                    break
                except Exception as e:
                    print(f"Failed to load preprocessor {path}: {e}")
                    preprocessor = None
        used_preprocessor = False

        # Prepare input array for prediction (coerce types and handle NaNs)
        # Ensure the DataFrame columns match the model expected number of features.
        model_expected_dim = None
        try:
            if hasattr(get_model, 'X') and getattr(get_model, 'X') is not None:
                model_expected_dim = int(np.array(get_model.X).shape[1])
        except Exception:
            model_expected_dim = None

        # If model expects a different number of features, trim or pad the DataFrame accordingly
        if model_expected_dim is not None:
            if X.shape[1] > model_expected_dim:
                # trim extra columns (keep left-most columns)
                print(f"Trimming input features from {X.shape[1]} to {model_expected_dim}")
                X = X.iloc[:, :model_expected_dim]
            elif X.shape[1] < model_expected_dim:
                # pad missing columns with zeros on the right
                diff = model_expected_dim - X.shape[1]
                print(f"Padding input features from {X.shape[1]} to {model_expected_dim} by adding {diff} zero-columns")
                for i in range(diff):
                    X[f'_pad_{i}'] = 0

        input_array = X.to_numpy()[0, :].astype(object)
        # Try to convert entries to float using column-aware parsing
        cleaned = []
        for idx, v in enumerate(input_array):
            colname = X.columns[idx]
            # Handle ZIP codes: try to extract numeric digits first
            if colname.lower() == 'zip' or colname.lower() == 'zipcode':
                parsed = None
                try:
                    # If it's numeric-like, convert directly
                    parsed = float(v)
                except Exception:
                    try:
                        s = str(v)
                        digits = ''.join([c for c in s if c.isdigit()])
                        if digits:
                            parsed = float(digits)
                    except Exception:
                        parsed = None
                if parsed is None:
                    # fallback to small hash but we'll rescale later if model stats exist
                    try:
                        h = abs(hash(str(v))) % 1000
                        parsed = float(h) / 1000.0
                    except Exception:
                        parsed = 0.0
                cleaned.append(parsed)
                continue

            try:
                cleaned.append(float(v))
            except Exception:
                # fallback: small normalized hash
                try:
                    h = abs(hash(str(v))) % 1000
                    cleaned.append(float(h) / 1000.0)
                except Exception:
                    cleaned.append(0.0)
        input_np = np.array(cleaned, dtype=float)

        # If a preprocessor was found, try to apply it to the whole DataFrame first (preferred)
        input_scaled = input_np
        try:
            if preprocessor is not None:
                try:
                    # Some preprocessors expect a DataFrame; try feeding X (full DataFrame)
                    transformed = preprocessor.transform(X)
                    # If transform returned an array with shape match, use it
                    if hasattr(transformed, 'shape') and transformed.shape[1] >= 1:
                        # take first row
                        input_scaled = np.array(transformed)[0].astype(float)
                        used_preprocessor = True
                        print('Applied preprocessor to full DataFrame')
                except Exception:
                    # try transforming single-row input array
                    try:
                        transformed = preprocessor.transform(input_np.reshape(1, -1))
                        input_scaled = np.array(transformed)[0].astype(float)
                        used_preprocessor = True
                        print('Applied preprocessor to single row')
                    except Exception:
                        used_preprocessor = False
        except Exception as e:
            print(f'Preprocessor application failed: {e}')

        # If no preprocessor was applied, fall back to winsorization + standardization
        if not used_preprocessor:
            try:
                if model_expected_dim is not None and hasattr(get_model, 'X') and getattr(get_model, 'X') is not None:
                    model_X = np.array(get_model.X).astype(float)
                    if model_X.ndim == 2 and model_X.shape[1] == input_np.shape[0]:
                        # compute robust bounds from training data (1st and 99th percentiles)
                        q_low = np.nanpercentile(model_X, 1.0, axis=0)
                        q_high = np.nanpercentile(model_X, 99.0, axis=0)
                        # avoid degenerate bounds
                        equal_mask = (q_high - q_low) == 0
                        if np.any(equal_mask):
                            # replace degenerate bounds with mean +/- std
                            m_mean_tmp = np.nanmean(model_X, axis=0)
                            m_std_tmp = np.nanstd(model_X, axis=0)
                            q_low[equal_mask] = m_mean_tmp[equal_mask] - 3.0 * np.maximum(m_std_tmp[equal_mask], 1.0)
                            q_high[equal_mask] = m_mean_tmp[equal_mask] + 3.0 * np.maximum(m_std_tmp[equal_mask], 1.0)

                        # winsorize input to [q_low, q_high]
                        input_clip = np.clip(input_np, q_low, q_high)

                        # now standardize using mean/std
                        m_mean = np.nanmean(model_X, axis=0)
                        m_std = np.nanstd(model_X, axis=0)
                        m_std[m_std == 0] = 1.0
                        input_scaled = (input_clip - m_mean) / m_std
                        # Clip extreme z-scores to avoid very large logits
                        input_scaled = np.clip(input_scaled, -5.0, 5.0)
                    else:
                        # shapes don't match; leave unscaled
                        input_scaled = input_np
            except Exception as e:
                print(f"Warning: could not scale/winsorize input using model stats: {e}")
                input_scaled = input_np
        try:
            if model_expected_dim is not None and hasattr(get_model, 'X') and getattr(get_model, 'X') is not None:
                model_X = np.array(get_model.X).astype(float)
                if model_X.ndim == 2 and model_X.shape[1] == input_np.shape[0]:
                    # compute robust bounds from training data (1st and 99th percentiles)
                    q_low = np.nanpercentile(model_X, 1.0, axis=0)
                    q_high = np.nanpercentile(model_X, 99.0, axis=0)
                    # avoid degenerate bounds
                    equal_mask = (q_high - q_low) == 0
                    if np.any(equal_mask):
                        # replace degenerate bounds with mean +/- std
                        m_mean_tmp = np.nanmean(model_X, axis=0)
                        m_std_tmp = np.nanstd(model_X, axis=0)
                        q_low[equal_mask] = m_mean_tmp[equal_mask] - 3.0 * np.maximum(m_std_tmp[equal_mask], 1.0)
                        q_high[equal_mask] = m_mean_tmp[equal_mask] + 3.0 * np.maximum(m_std_tmp[equal_mask], 1.0)

                    # winsorize input to [q_low, q_high]
                    input_clip = np.clip(input_np, q_low, q_high)

                    # now standardize using mean/std
                    m_mean = np.nanmean(model_X, axis=0)
                    m_std = np.nanstd(model_X, axis=0)
                    m_std[m_std == 0] = 1.0
                    input_scaled = (input_clip - m_mean) / m_std
                    # Clip extreme z-scores to avoid very large logits
                    input_scaled = np.clip(input_scaled, -5.0, 5.0)
                else:
                    # shapes don't match; leave unscaled
                    input_scaled = input_np
        except Exception as e:
            print(f"Warning: could not scale/winsorize input using model stats: {e}")
            input_scaled = input_np

        # Try to get raw model logits and compute probability with a temperature fallback
        fraud_prob = None
        try:
            # run raw forward
            get_model.eval()
            t = torch.from_numpy(input_scaled.astype('float32')).unsqueeze(0)
            with torch.no_grad():
                raw = get_model(t, train=False)

            # raw may be shape [1,1] or [1] etc. extract scalar
            try:
                raw_val = float(raw.cpu().numpy().flatten()[0])
            except Exception:
                raw_val = None

            # if we got a raw logit, compute prob via sigmoid; if extreme, apply temperature scaling
            if raw_val is not None:
                # dynamic temperature: scale by raw magnitude so extreme logits are softened
                temperature = max(1.0, abs(raw_val))
                prob_temp = float(torch.sigmoid(torch.tensor(raw_val / temperature)).item())
                prob_direct = float(torch.sigmoid(torch.tensor(raw_val)).item())
                # blend the direct and softened probabilities to avoid extremes
                fraud_prob = float((prob_temp + prob_direct) / 2.0)

        except Exception:
            fraud_prob = None

        # If we couldn't compute from raw logits, fallback to predict_proba
        if fraud_prob is None:
            try:
                raw_proba = predict_proba(get_model, input_scaled)
                try:
                    fraud_prob = float(raw_proba)
                except Exception:
                    fraud_prob = float(torch.tensor(raw_proba).cpu().detach().numpy().flatten()[0])
            except Exception:
                fraud_prob = 0.0

        # Convert to percentage and clamp to [0,100]
        fraud_probability = min(max(fraud_prob * 100.0, 0.0), 100.0)
        non_fraud_probability = 100.0 - fraud_probability

        print(f"Fraud Probability: {fraud_probability:.2f}%")
        print(f"Non-Fraud Probability: {non_fraud_probability:.2f}%")

    # (logging moved below after output is determined)

        user_data = pd.read_csv(os.path.join(current_app.instance_path, "user_data.csv"))

        # Determine output based on probability threshold
        if fraud_probability < 50:
            output = 'Non-Fraudulent Activity'
            confidence = min(non_fraud_probability, 100)  # Cap at 100%
        else:
            output = 'Fraudulent Activity'
            confidence = min(fraud_probability, 100)  # Cap at 100%

        message = f"The Model is {confidence:.1f}% sure that it is {output}!"

        # Log the prediction (safe commit) but don't let logging failures break response
        if current_user.is_authenticated:
            try:
                log = PredictionLog(
                    user_id=current_user.id,
                    prediction=output,
                    confidence=(fraud_probability if output == 'Fraudulent Activity' else non_fraud_probability)
                )
                db.session.add(log)
                db.session.commit()
            except Exception as e:
                # don't break the response if logging fails; just print for debugging
                print(f"Failed to log prediction: {e}")

        # Prepare table(s) for rendering
        if user_data.empty:
            tables = []
        else:
            tables = [user_data.to_html()]

        return render_template('prediction_output.html', 
                                 tables=tables, 
                                 titles=[''], 
                                 output=output, 
                                 model_says=message,
                                 fraud_probability=f"{fraud_probability:.1f}%")

    except Exception as e:
        tb = traceback.format_exc()
        # Log full traceback to server console
        print('Prediction error traceback:\n', tb)
        short = str(e)
        return render_template('upload_page.html', error=f"Prediction error: {short}")

@views.route('/access_models.html')
@login_required
def access_models():

    return render_template('access_models.html')


@views.route('/debug_model', methods=['GET', 'POST'])
def debug_model():
    """Upload a single-row CSV or submit the current user_data.csv to inspect
    how the model cleans, scales and scores that row. Returns a simple debug page.
    """
    if request.method == 'POST':
        # find uploaded file or fallback to instance/user_data.csv
        file = request.files.get('file')
        try:
            if file and file.filename:
                df = pd.read_csv(file)
            else:
                df = pd.read_csv(os.path.join(current_app.instance_path, 'user_data.csv'))
        except Exception as e:
            return render_template('debug_model.html', error=f'Could not read CSV: {e}')

        if df.shape[0] < 1:
            return render_template('debug_model.html', error='CSV must contain at least one row')

        # only use the first row for diagnostics
        row = df.iloc[0:1].copy()

        # prepare features list consistent with prediction route
        required_features = ['amt', 'zip', 'lat', 'long', 'genders',
                            'Essentials', 'Leisure', 'Wellness', 'Other',
                            'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL',
                            'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA',
                            'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE',
                            'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI',
                            'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']

        for state in required_features[9:]:
            if state not in row.columns:
                row[state] = 0

        row = row.reindex(columns=required_features, fill_value=0)

        # load model similarly to prediction route
        try:
            import importlib
            models_xb_mod = importlib.import_module('XBNet.models_xb')
            sys.modules['models_xb'] = models_xb_mod
        except Exception:
            try:
                import models_xb as models_xb_mod
                sys.modules['models_xb'] = models_xb_mod
            except Exception:
                pass

        try:
            get_model = joblib.load(os.path.join(current_app.instance_path, 'v2.pkl'))
        except Exception as e:
            return render_template('debug_model.html', error=f'Could not load model: {e}')

        # convert row to cleaned numeric input (reuse earlier logic)
        input_array = row.to_numpy()[0, :].astype(object)
        cleaned = []
        for v in input_array:
            try:
                cleaned.append(float(v))
            except Exception:
                try:
                    h = abs(hash(str(v))) % 1000
                    cleaned.append(float(h) / 1000.0)
                except Exception:
                    cleaned.append(0.0)
        input_np = np.array(cleaned, dtype=float)

        # Attempt to apply preprocessor for debug route as well
        input_scaled = input_np.copy()
        model_stats = None
        used_preprocessor_debug = False
        try:
            # look for saved preprocessor in instance
            preprocessor = None
            for name in ('preprocessor.pkl', 'scaler.pkl', 'pipeline.pkl', 'transformer.pkl', 'preproc.pkl'):
                path = os.path.join(current_app.instance_path, name)
                if os.path.exists(path):
                    try:
                        preprocessor = joblib.load(path)
                        break
                    except Exception:
                        preprocessor = None
            if preprocessor is not None:
                try:
                    transformed = preprocessor.transform(row)
                    input_scaled = np.array(transformed)[0].astype(float)
                    used_preprocessor_debug = True
                    model_stats = {'preprocessor': True}
                except Exception:
                    try:
                        transformed = preprocessor.transform(input_np.reshape(1, -1))
                        input_scaled = np.array(transformed)[0].astype(float)
                        used_preprocessor_debug = True
                        model_stats = {'preprocessor': True}
                    except Exception:
                        used_preprocessor_debug = False
        except Exception:
            used_preprocessor_debug = False

        if not used_preprocessor_debug:
            try:
                if hasattr(get_model, 'X') and getattr(get_model, 'X') is not None:
                    model_X = np.array(get_model.X).astype(float)
                    if model_X.ndim == 2 and model_X.shape[1] == input_np.shape[0]:
                        q_low = np.nanpercentile(model_X, 1.0, axis=0)
                        q_high = np.nanpercentile(model_X, 99.0, axis=0)
                        equal_mask = (q_high - q_low) == 0
                        if np.any(equal_mask):
                            m_mean_tmp = np.nanmean(model_X, axis=0)
                            m_std_tmp = np.nanstd(model_X, axis=0)
                            q_low[equal_mask] = m_mean_tmp[equal_mask] - 3.0 * np.maximum(m_std_tmp[equal_mask], 1.0)
                            q_high[equal_mask] = m_mean_tmp[equal_mask] + 3.0 * np.maximum(m_std_tmp[equal_mask], 1.0)
                        input_clip = np.clip(input_np, q_low, q_high)
                        m_mean = np.nanmean(model_X, axis=0)
                        m_std = np.nanstd(model_X, axis=0)
                        m_std[m_std == 0] = 1.0
                        input_scaled = (input_clip - m_mean) / m_std
                        input_scaled = np.clip(input_scaled, -5.0, 5.0)
                        model_stats = {'mean': m_mean.tolist(), 'std': m_std.tolist(), 'q_low': q_low.tolist(), 'q_high': q_high.tolist()}
            except Exception:
                model_stats = None

        # run model forward to get logits and an adjusted probability
        try:
            get_model.eval()
            import torch as _torch
            with _torch.no_grad():
                t = _torch.from_numpy(input_scaled.astype('float32')).unsqueeze(0)
                raw = get_model(t, train=False)
                raw_out = raw.cpu().numpy().tolist()
                try:
                    sig = _torch.sigmoid(raw).cpu().numpy().tolist()
                except Exception:
                    sig = None
                try:
                    sm = _torch.nn.functional.softmax(raw, dim=1).cpu().numpy().tolist()
                except Exception:
                    sm = None

                # compute adjusted prob via temperature scaling fallback
                try:
                    raw_val = float(raw.cpu().numpy().flatten()[0])
                    temperature = max(1.0, abs(raw_val))
                    prob_temp = float(_torch.sigmoid(_torch.tensor(raw_val / temperature)).item())
                    prob_direct = float(_torch.sigmoid(_torch.tensor(raw_val)).item())
                    adjusted_prob = float((prob_temp + prob_direct) / 2.0)
                except Exception:
                    adjusted_prob = None
        except Exception as e:
            return render_template('debug_model.html', error=f'Error running model: {e}')

        return render_template('debug_model.html',
                               cleaned=input_np.tolist(),
                               scaled=input_scaled.tolist(),
                               model_stats=model_stats,
                               raw=raw_out,
                               sigmoid=sig,
                               softmax=sm,
                               adjusted_prob=adjusted_prob)

        return render_template('debug_model.html',
                               cleaned=input_np.tolist(),
                               scaled=input_scaled.tolist(),
                               model_stats=model_stats,
                               raw=raw_out,
                               sigmoid=sig,
                               softmax=sm)

    return render_template('debug_model.html')


@views.route('/data_analysis.html')
@login_required
def data_analysis():
    try:
        # Load the file using pandas
        user_data_path = os.path.join(current_app.instance_path, "user_data.csv")
        big_data_path = os.path.join(current_app.instance_path, "bigdata.csv")
        
        if not os.path.exists(user_data_path):
            return render_template('/data_analysis.html', error="No user data found. Please upload a CSV file first.")
            
        if not os.path.exists(big_data_path):
            return render_template('/data_analysis.html', error="Reference data not found. Please contact support.")
        
        # Load the data files
        X = pd.read_csv(user_data_path)
        big_data = pd.read_csv(big_data_path)

        # Calculate category averages separately
        avg_amounts = []
        for category in ['Essentials', 'Leisure', 'Wellness', 'Other']:
            category_data = big_data[big_data[category] == 1]
            if not category_data.empty:
                avg_amount = category_data['amt'].mean()
            else:
                avg_amount = 0
            avg_amounts.append({'Category': category, 'Average Amount': float(avg_amount)})  # Convert to float for JSON serialization
        avg_amt_by_category_1 = avg_amounts  # Pass the list directly instead of DataFrame
        
        # Calculate user total spending
        user_spending_1 = X['amt'].sum() if 'amt' in X.columns else 0

        # Get the model prediction
        try:
            # Prepare features in correct order
            required_features = ['amt', 'zip', 'lat', 'long', 'genders', 
                            'Essentials', 'Leisure', 'Wellness', 'Other',
                            'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL',
                            'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA',
                            'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE',
                            'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI',
                            'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']
            
            # Ensure all required columns exist
            for col in required_features:
                if col not in X.columns:
                    if col in ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL',
                            'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA',
                            'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE',
                            'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI',
                            'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']:
                        X[col] = 0
                    else:
                        return render_template('/data_analysis.html', 
                            error=f"Missing required column: {col}. Please ensure your data includes all required fields.")

            # Order columns as required by model
            X = X[required_features]
            
            # Load model and make prediction
            get_model = joblib.load(os.path.join(current_app.instance_path, "v2.pkl"))
            prediction = predict(get_model, X.to_numpy()[0,:])
            
            # Add prediction to dataframe
            X['is_fraud'] = prediction
            
            # Get model file info
            all_file_names = os.listdir(current_app.instance_path)
            pkl_file_names = [f for f in all_file_names if f.endswith('.pkl')]
            most_recent_pkl_data = pkl_file_names[0] if pkl_file_names else "v2.pkl"
            
            res = 'Non-Fraudulent Activity' if prediction == 1 else 'Fraudulent Activity'
            
            return render_template('/data_analysis.html', 
                                model_name=most_recent_pkl_data,
                                res=res,
                                bigd=big_data,
                                usr_d=X,
                                user_spending=user_spending_1,
                                avg_amt_by_category=avg_amt_by_category_1)
                                
        except Exception as e:
            return render_template('/data_analysis.html', 
                error=f"Error making prediction: {str(e)}. Please ensure your data is in the correct format.")
            
    except Exception as e:
        return render_template('/data_analysis.html', 
            error=f"Error analyzing data: {str(e)}. Please try again or contact support.")


@views.route('/my_documents.html')
@login_required
def my_documents():
    all_file_names = os.listdir(current_app.instance_path)
    csv_file_names = [f for f in all_file_names if f.endswith('.csv')]
    most_recent_user_data = csv_file_names[1]

    if request.method == 'POST':

        # Get a new file from the form request
        file = request.files['file']

        if file:

            # Save the file to the server
            file.save(os.path.join(current_app.instance_path, "user_data_new.csv"))

            return redirect(url_for('views.prediction_output'))

    return render_template('my_documents.html', recent_file=most_recent_user_data)

@views.route('/marketplace.html')
@login_required
def marketplace():

    return render_template('/marketplace.html')


# Admin access decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash('You need admin privileges to access this page.', category='error')
            return redirect(url_for('views.upload_page'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash('You must have admin privileges to access this page.', 'error')
            return redirect(url_for('auth.login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

# Admin routes
@views.route('/admin')
@login_required
def admin():
    if not current_user.is_admin:
        flash('Admin access required', 'error')
        return redirect(url_for('views.upload_page'))
        
    # Get basic stats
    total_users = User.query.count()
    active_models = len([f for f in os.listdir(current_app.instance_path) if f.endswith('.pkl')])

    # Count predictions made today
    try:
        today = datetime.now().date()
        predictions_today = PredictionLog.query.filter(func.date(PredictionLog.created_at) == today).count()
        # Recent activity (most recent 10 predictions)
        activities = PredictionLog.query.order_by(PredictionLog.created_at.desc()).limit(10).all()
    except Exception as e:
        # If the PredictionLog table doesn't exist or query fails, fall back to 0 and empty list
        print(f"Warning: could not query PredictionLog: {e}")
        predictions_today = 0
        activities = []

    return render_template('admin.html', 
                         total_users=total_users,
                         active_models=active_models,
                         predictions_today=predictions_today,
                         activities=activities)

@views.route('/admin/users')
@login_required
@admin_required
def admin_users():
    users = User.query.all()
    return render_template('admin.html', section='users', users=users)

@views.route('/admin/models')
@login_required
@admin_required
def admin_models():
    models = [f for f in os.listdir(current_app.instance_path) if f.endswith('.pkl')]
    return render_template('admin.html', section='models', models=models)

@views.route('/admin/logs')
@login_required
@admin_required
def admin_logs():
    logs = PredictionLog.query.order_by(PredictionLog.created_at.desc()).limit(100).all()
    return render_template('admin.html', section='logs', logs=logs)

# User settings route
@views.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    if not current_user.is_authenticated:
        flash('Please log in to access settings.', category='error')
        return redirect(url_for('auth.login'))

    if request.method == 'POST':
        # Handle password change
        new_password = request.form.get('newPassword')
        confirm_password = request.form.get('confirmPassword')
        
        if new_password and confirm_password:
            if new_password != confirm_password:
                flash('Passwords do not match.', category='error')
            elif len(new_password) < 7:
                flash('Password must be at least 7 characters.', category='error')
            else:
                try:
                    current_user.password = generate_password_hash(new_password, method='sha256')
                    db.session.commit()
                    flash('Password updated successfully!', category='success')
                    return redirect(url_for('views.settings'))
                except Exception as e:
                    flash(f'Error updating password: {str(e)}', category='error')
                    db.session.rollback()
    
    return render_template('settings.html')

# Public support and about pages
@views.route('/support')
def support():
    return render_template('support.html')


@views.route('/about')
def about():
    return render_template('about.html')