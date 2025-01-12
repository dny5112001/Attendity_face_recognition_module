from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from deepface import DeepFace
import io
from PIL import Image

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)


@app.route("/compare_faces", methods=["POST"])
def compare_faces():
    try:
        # Ensure both images are provided in the request
        if "img1" not in request.files or "img2" not in request.files:
            return jsonify({"error": "Both images are required"}), 400

        # Get the images from the request
        img1 = request.files["img1"]
        img2 = request.files["img2"]

        # Read images into memory
        img1_image = Image.open(io.BytesIO(img1.read()))
        img2_image = Image.open(io.BytesIO(img2.read()))

        # Save the images to temporary file-like objects for comparison
        img1_path = "img1.jpg"
        img2_path = "img2.jpg"

        img1_image.save(img1_path)
        img2_image.save(img2_path)

        # Perform face comparison using DeepFace
        result = DeepFace.verify(img1_path, img2_path)

        # Return the verification result
        if result["verified"]:
            return jsonify({"message": "The two images match!", "verified": True}), 200
        else:
            return (
                jsonify({"message": "The two images do not match.", "verified": False}),
                200,
            )

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "An error occurred while processing the images."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
