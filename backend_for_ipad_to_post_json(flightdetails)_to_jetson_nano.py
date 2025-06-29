from flask import Flask, request, jsonify
import json

app = Flask(__name__)

FLIGHT_FILE = "flight_details.json"

@app.route('/update_flight', methods=['POST'])
def update_flight():
    data = request.json
    if not data:
        return jsonify({"error": "No data received"}), 400

    # Validate and save data
    with open(FLIGHT_FILE, 'w') as f:
        json.dump(data, f, indent=4)

    return jsonify({"status": "success"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) #listens on port 5000 on Jetson Nano; iPad frontend can POST .json data to this API


