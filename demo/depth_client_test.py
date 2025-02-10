import requests
import json

# Define the URL of the Flask application
url = "http://localhost:5000/get_depth"

# Define the image path to be sent in the request
img_path = '/mnt/data/Datasets/HOI4D_release/ZY20210800004/H4/C4/N42/S260/s01/T2/align_rgb/00011.jpg'

# Create the payload with the image path
payload = {
    "img_path": img_path
}

# Send a POST request to the Flask application
response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload))

# Check if the request was successful
if response.status_code == 200:
    # Print the response from the server
    print("Response from server:", response.json())
else:
    # Print the error message
    print("Error:", response.status_code, response.text)