import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import RobustScaler
import joblib
import json
from tensorflow.keras.saving import register_keras_serializable
from typing import Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Constants
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
# Load feature names dynamically
with open(os.path.join(MODELS_DIR, 'feature_names.json')) as f:
    INPUT_FEATURES = json.load(f)
TARGETS = ['Tg', 'FFV', 'Tc', 'Density']

# Custom scheduler
@register_keras_serializable(package="Custom")
class WarmupExponentialDecay(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps, decay_steps, decay_rate):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.initial_lr * (step / self.warmup_steps)
        decay_lr = self.initial_lr * tf.math.pow(self.decay_rate, step / self.decay_steps)
        return tf.where(step <= self.warmup_steps, warmup_lr, decay_lr)

    def get_config(self):
        return {
            'initial_lr': self.initial_lr,
            'warmup_steps': self.warmup_steps,
            'decay_steps': self.decay_steps,
            'decay_rate': self.decay_rate
        }

class ModelLoader:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance._load_models()
        return cls._instance
    
    def _load_models(self):
        try:
            os.makedirs(MODELS_DIR, exist_ok=True)
            
            self.model = keras.models.load_model(
                os.path.join(MODELS_DIR, 'enhanced_model.keras'),
                custom_objects={'WarmupExponentialDecay': WarmupExponentialDecay}
            )
            
            self.x_scaler = joblib.load(os.path.join(MODELS_DIR, 'x_scaler.joblib'))
            logger.info(f"Scaler expects {self.x_scaler.n_features_in_} features")
            
            self.y_scalers = {
                target: joblib.load(os.path.join(MODELS_DIR, f'y_scaler_{target}.joblib'))
                for target in TARGETS
            }
            
            # Load feature names
            feature_names_path = os.path.join(MODELS_DIR, 'feature_names.json')
            if os.path.exists(feature_names_path):
                with open(feature_names_path) as f:
                    self.feature_names = json.load(f)
                logger.info(f"Model expects features: {self.feature_names}")
            else:
                raise ValueError("feature_names.json not found in models directory")
            
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise

try:
    model_loader = ModelLoader()
except Exception as e:
    logger.critical(f"Startup failed: {str(e)}")
    raise

def validate_input(input_data: Dict[str, float]) -> Optional[str]:
    for feature in model_loader.feature_names:
        if feature not in input_data and not feature.startswith('feat_') and not feature.endswith('_missing'):
            return f"Missing required field: {feature}"
        try:
            if feature in input_data:
                float(input_data[feature])
        except (ValueError, TypeError):
            return f"Invalid value for {feature}"
    return None

def preprocess_input(input_data: Dict[str, float]) -> np.ndarray:
    """Preprocess input to exactly match training data format"""
    try:
        df = pd.DataFrame([input_data])
        
        # 1. Handle missing values and create missing indicators
        numeric_cols = [col for col in model_loader.feature_names if not col.startswith('feat_') and not col.endswith('_missing')]
        for col in numeric_cols:
            if col not in df.columns:
                df[col] = 0.0
            df[f'{col}_missing'] = df[col].isna().astype(int)
            df[col].fillna(df[col].median(), inplace=True)
        
        # 2. Create engineered features
        df['feat_mean'] = df[numeric_cols].mean(axis=1)
        df['feat_std'] = df[numeric_cols].std(axis=1)
        df['feat_range'] = df[numeric_cols].max(axis=1) - df[numeric_cols].min(axis=1)
        df['feat_skew'] = df[numeric_cols].skew(axis=1)
        df['feat_q90'] = df[numeric_cols].quantile(0.9, axis=1)
        
        # 3. Select only the expected features in correct order
        for col in model_loader.feature_names:
            if col not in df.columns:
                df[col] = 0.0
        X = df[model_loader.feature_names].values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        return model_loader.x_scaler.transform(X)
    
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise ValueError(f"Input processing error: {str(e)}")

def make_predictions(scaled_input: np.ndarray) -> Dict[str, float]:
    try:
        predictions_scaled = model_loader.model.predict(scaled_input, verbose=0)
        return {
            target: float(model_loader.y_scalers[target].inverse_transform(
                predictions_scaled[i].reshape(-1, 1))[0][0])
            for i, target in enumerate(TARGETS)
        }
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise RuntimeError(f"Prediction error: {str(e)}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            input_data = {
                feature: request.form.get(feature, '')
                for feature in model_loader.feature_names
                if not feature.startswith('feat_') and not feature.endswith('_missing')
            }
            
            try:
                input_data = {k: float(v) if v else 0.0 for k, v in input_data.items()}
            except ValueError:
                return render_template('index.html',
                                    predictions=None,
                                    feature_values=input_data,
                                    error="All inputs must be numbers")
            
            error = validate_input(input_data)
            if error:
                return render_template('index.html',
                                     predictions=None,
                                     feature_values=input_data,
                                     error=error)
            
            processed_data = preprocess_input(input_data)
            predictions = make_predictions(processed_data)
            
            return render_template('index.html',
                                predictions=predictions,
                                feature_values=input_data,
                                error=None)
        
        except Exception as e:
            return render_template('index.html',
                                predictions=None,
                                feature_values=input_data,
                                error=str(e))
    
    return render_template('index.html',
                         predictions=None,
                         feature_values={k: '' for k in model_loader.feature_names 
                                       if not k.startswith('feat_') and not k.endswith('_missing')},
                         error=None)

@app.route('/api/predict', methods=['POST'])
def predict_api():
    try:
        if not request.is_json:
            return jsonify({'status': 'error', 'message': 'Request must be JSON'}), 400
            
        data = request.get_json()
        error = validate_input(data)
        if error:
            return jsonify({'status': 'error', 'message': error}), 400
        
        processed_data = preprocess_input(data)
        predictions = make_predictions(processed_data)
        
        return jsonify({
            'status': 'success',
            'predictions': predictions,
            'input_features': data
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/health')
def health_check():
    try:
        dummy_input = {feature: 0.0 for feature in model_loader.feature_names 
                      if not feature.startswith('feat_') and not feature.endswith('_missing')}
        processed_data = preprocess_input(dummy_input)
        _ = make_predictions(processed_data)
        return jsonify({'status': 'healthy'}), 200
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)