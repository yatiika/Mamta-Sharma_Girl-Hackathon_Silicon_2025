def predict_depth(verilog_file):
    """Predict combinational depth for a given RTL module"""
    features = extract_features_from_rtl(verilog_file)
    model = joblib.load("depth_predictor.pkl")
    
    feature_vector = np.array([[features["num_gates"], features["max_fan_in"], features["max_fan_out"]]])
    
    predicted_depth = model.predict(feature_vector)[0]
    return predicted_depth

# Example usage
verilog_test_file = "test_rtl.v"
predicted_depth = predict_depth(verilog_test_file)
print(f"Predicted Combinational Depth: {predicted_depth}")
