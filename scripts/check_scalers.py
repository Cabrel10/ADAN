import joblib

scalers = joblib.load('./models/training_scalers.pkl')

for tf, scaler in scalers.items():
    print(f"{tf}: {type(scaler).__name__}")
    if hasattr(scaler, 'n_features_in_'):
        print(f"   n_features_in_: {scaler.n_features_in_}")
    if hasattr(scaler, 'scale_'):
        print(f"   scale_ shape: {scaler.scale_.shape}")
