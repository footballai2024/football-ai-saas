from flask import Flask, request, jsonify, render_template
from predictor import full_pipeline
from database import get_all_predictions, get_accuracy_stats

app = Flask(__name__)

@app.route('/')
def home():
    """Show the dashboard"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle screenshot upload and return prediction"""
    try:
        file = request.files.get('image')

        if not file:
            return jsonify({"error": "No image uploaded"}), 400

        result = full_pipeline(file)

        if not isinstance(result, dict):
            return jsonify({"error": "Prediction result is not valid"}), 500

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/live')
def live():
    """Get live prediction"""
    try:
        result = full_pipeline(None, live=True)

        if not isinstance(result, dict):
            return jsonify({"error": "Live prediction result is not valid"}), 500

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/history')
def history():
    """Show prediction history"""
    try:
        preds = get_all_predictions()
        return jsonify({"predictions": preds})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/accuracy')
def accuracy():
    """Show accuracy stats"""
    try:
        stats = get_accuracy_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)