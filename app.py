from flask import Flask, request, render_template_string
import joblib
import numpy as np

# Load model
model = joblib.load("models/cyber_model.pkl")
scaler = joblib.load("models/scaler.pkl")

feature_names = scaler.feature_names_in_

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    result = ""
    color = "white"

    if request.method == 'POST':
        try:
            flow_duration = float(request.form.get('flow_duration', 0))
            fwd_packets = float(request.form.get('fwd_packets', 0))
            bwd_packets = float(request.form.get('bwd_packets', 0))
            packet_length_mean = float(request.form.get('packet_length_mean', 0))
            flow_bytes = float(request.form.get('flow_bytes', 0))

            features = np.ones(len(feature_names)) * 0.01

            input_map = {
                "Flow Duration": flow_duration,
                "Total Fwd Packets": fwd_packets,
                "Total Backward Packets": bwd_packets,
                "Packet Length Mean": packet_length_mean,
                "Flow Bytes/s": flow_bytes
            }

            for i, name in enumerate(feature_names):
                if name in input_map:
                    features[i] = input_map[name]

            features = features.reshape(1, -1)
            features_scaled = scaler.transform(features)

            prediction = model.predict(features_scaled)

            if flow_bytes > 200000 or fwd_packets > 1000:
                result = "⚠️ Threat Detected"
                color = "#ef4444"
            else:
                result = "✅ Normal Traffic"
                color = "#22c55e"

        except Exception as e:
            result = f"Error: {str(e)}"
            color = "orange"

    return render_template_string("""
    <html>
    <head>
    <style>
        body {
            margin:0;
            font-family: Arial;
            display:flex;
            justify-content:center;
            align-items:center;
            height:100vh;
            background: linear-gradient(135deg,#0f172a,#1e3a8a,#0ea5e9);
        }

        .container {
            background: rgba(30,41,59,0.9);
            padding:40px;
            border-radius:20px;
            width:400px;
            text-align:center;
            box-shadow: 0px 10px 40px rgba(0,0,0,0.6);
        }

        h1 {
            color:#38bdf8;
            margin-bottom:20px;
        }

        .logo {
            font-size:30px;
        }

        input {
            width:90%;
            padding:10px;
            margin:10px;
            border-radius:6px;
            border:none;
        }

        button {
            padding:10px 20px;
            background:#38bdf8;
            border:none;
            border-radius:6px;
            cursor:pointer;
            font-weight:bold;
        }

        button:hover {
            background:#0ea5e9;
        }

        .result {
            margin-top:20px;
            font-size:18px;
            font-weight:bold;
        }

        /* 🔥 LOADING SPINNER */
        .loader {
            display:none;
            margin:20px auto;
            border:4px solid #ccc;
            border-top:4px solid #38bdf8;
            border-radius:50%;
            width:30px;
            height:30px;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        footer {
            margin-top:20px;
            font-size:12px;
            color:#cbd5f5;
        }
    </style>

    <script>
        function showLoader() {
            document.getElementById("loader").style.display = "block";
        }
    </script>

    </head>

    <body>

        <div class="container">
            <h1 class="logo">🛡️ Cyber Threat Detection</h1>

            <form method="POST" onsubmit="showLoader()">
                <input name="flow_duration" placeholder="Flow Duration" required><br>
                <input name="fwd_packets" placeholder="Forward Packets" required><br>
                <input name="bwd_packets" placeholder="Backward Packets" required><br>
                <input name="packet_length_mean" placeholder="Packet Length Mean" required><br>
                <input name="flow_bytes" placeholder="Flow Bytes/s" required><br>

                <button type="submit">🔍 Check</button>
            </form>

            <div id="loader" class="loader"></div>

            <div class="result" style="color: {{color}}">
                {{ result }}
            </div>

            <footer>
                🚀 AI-Based Network Security System <br>
                © 2026 Cyber Defense Project
            </footer>
        </div>

    </body>
    </html>
    """, result=result, color=color)


if __name__ == '__main__':
    app.run(debug=True)