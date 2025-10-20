import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib
import os

class RockfallPredictor:
    def __init__(self):
        self.cnn_model = None
        self.rf_model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10,
            random_state=42
        )
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.gb_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.svm_model = SVC(
            probability=True,
            kernel='rbf',
            random_state=42
        )
        
        self.scaler = StandardScaler()
        self.model_dir = 'model'
        self.ensure_model_dir()
        
        if self.models_exist():
            self.load_models()
        else:
            self.create_models()

    def ensure_model_dir(self):
        """Create model directory if it doesn't exist"""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def models_exist(self):
        """Check if all model files exist"""
        files_needed = [
            'cnn_model.h5',
            'rf_model.joblib',
            'xgb_model.json',
            'gb_model.joblib',
            'svm_model.joblib',
            'scaler.joblib'
        ]
        return all(os.path.exists(os.path.join(self.model_dir, f)) for f in files_needed)

    def create_models(self):
        """Create all models"""
        self._create_cnn()
        # Other models will be trained with actual data

    def _create_cnn(self):
        """Create CNN model for image and DEM data processing"""
        self.cnn_model = models.Sequential([
            # CNN branch for DEM and image processing
            layers.Input(shape=(64, 64, 4)),  # DEM + RGB channels
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        
        self.cnn_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def preprocess_data(self, dem_data, image_data, sensor_data):
        """Preprocess all input data"""
        # Normalize DEM data
        dem_normalized = (dem_data - np.min(dem_data)) / (np.max(dem_data) - np.min(dem_data))
        
        # Normalize image data
        image_normalized = image_data / 255.0
        
        # Scale sensor data
        sensor_scaled = self.scaler.transform(sensor_data.reshape(1, -1))
        
        # Combine DEM and image data for CNN
        cnn_input = np.concatenate([
            dem_normalized[..., np.newaxis],
            image_normalized
        ], axis=-1)
        
        # Create feature vector for traditional models
        traditional_input = np.concatenate([
            sensor_scaled.flatten(),
            np.array([
                np.mean(dem_normalized),
                np.std(dem_normalized),
                np.max(dem_normalized),
                np.mean(image_normalized),
                np.std(image_normalized)
            ])
        ])
        
        return cnn_input, traditional_input

    def train(self, dem_data, image_data, sensor_data, labels):
        """Train all models with the provided data"""
        # Preprocess data
        cnn_input, traditional_input = self.preprocess_data(dem_data, image_data, sensor_data)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            traditional_input, labels, test_size=0.2, random_state=42
        )
        
        # Train CNN
        self.cnn_model.fit(
            cnn_input, labels,
            epochs=50,
            batch_size=32,
            validation_split=0.2
        )
        
        # Train traditional models
        self.rf_model.fit(X_train, y_train)
        self.xgb_model.fit(X_train, y_train)
        self.gb_model.fit(X_train, y_train)
        self.svm_model.fit(X_train, y_train)
        
        # Save all models
        self.save_models()

    def predict(self, dem_data, image_data, sensor_data):
        """Make predictions using all models"""
        # Preprocess input data
        cnn_input, traditional_input = self.preprocess_data(dem_data, image_data, sensor_data)
        
        # Get predictions from all models
        cnn_pred = self.cnn_model.predict(cnn_input)[0]
        rf_pred = self.rf_model.predict_proba(traditional_input.reshape(1, -1))[0][1]
        xgb_pred = self.xgb_model.predict_proba(traditional_input.reshape(1, -1))[0][1]
        gb_pred = self.gb_model.predict_proba(traditional_input.reshape(1, -1))[0][1]
        svm_pred = self.svm_model.predict_proba(traditional_input.reshape(1, -1))[0][1]
        
        # Weighted ensemble prediction
        weights = [0.4, 0.15, 0.15, 0.15, 0.15]  # CNN has higher weight
        ensemble_pred = np.average([
            cnn_pred, rf_pred, xgb_pred, gb_pred, svm_pred
        ], weights=weights)
        
        return {
            'ensemble_probability': float(ensemble_pred),
            'model_probabilities': {
                'cnn': float(cnn_pred),
                'random_forest': float(rf_pred),
                'xgboost': float(xgb_pred),
                'gradient_boosting': float(gb_pred),
                'svm': float(svm_pred)
            }
        }

    def save_models(self):
        """Save all models to disk"""
        self.cnn_model.save(os.path.join(self.model_dir, 'cnn_model.h5'))
        joblib.dump(self.rf_model, os.path.join(self.model_dir, 'rf_model.joblib'))
        self.xgb_model.save_model(os.path.join(self.model_dir, 'xgb_model.json'))
        joblib.dump(self.gb_model, os.path.join(self.model_dir, 'gb_model.joblib'))
        joblib.dump(self.svm_model, os.path.join(self.model_dir, 'svm_model.joblib'))
        joblib.dump(self.scaler, os.path.join(self.model_dir, 'scaler.joblib'))

    def load_models(self):
        """Load all models from disk"""
        self.cnn_model = models.load_model(os.path.join(self.model_dir, 'cnn_model.h5'))
        self.rf_model = joblib.load(os.path.join(self.model_dir, 'rf_model.joblib'))
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(os.path.join(self.model_dir, 'xgb_model.json'))
        self.gb_model = joblib.load(os.path.join(self.model_dir, 'gb_model.joblib'))
        self.svm_model = joblib.load(os.path.join(self.model_dir, 'svm_model.joblib'))
        self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.joblib'))
        sensor_scaled = self.scaler.transform(sensor_data)
        
        return dem_normalized, image_normalized, sensor_scaled

    def train(self, dem_data, image_data, sensor_data, labels):
        """Train the model with the provided data"""
        dem_norm, img_norm, sensor_scaled = self.preprocess_data(dem_data, image_data, sensor_data)
        
        # Combine DEM and image data
        combined_input = np.concatenate([dem_norm[..., np.newaxis], img_norm], axis=-1)
        
        self.model.fit(
            [combined_input, sensor_scaled],
            labels,
            epochs=50,
            batch_size=32,
            validation_split=0.2
        )
        
        # Save the model and scaler
        os.makedirs('model', exist_ok=True)
        self.model.save(self.model_path)
        joblib.dump(self.scaler, self.scaler_path)

    def predict(self, dem_data, image_data, sensor_data):
        """Make predictions using the trained model"""
        dem_norm, img_norm, sensor_scaled = self.preprocess_data(dem_data, image_data, sensor_data)
        combined_input = np.concatenate([dem_norm[..., np.newaxis], img_norm], axis=-1)
        
        return self.model.predict([combined_input, sensor_scaled])

    def load_model(self):
        """Load a pre-trained model"""
        self.model = models.load_model(self.model_path)
        self.scaler = joblib.load(self.scaler_path)