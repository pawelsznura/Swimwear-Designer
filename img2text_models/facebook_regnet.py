import requests
import json
import config

API_URL = "https://api-inference.huggingface.co/models/facebook/regnet-y-008"
headers = {"Authorization": f"Bearer %s" %config.api_img2txt}

def query(filename):
    with open(filename, "rb") as f:
        # data = {"inputs": f.read(), "options": {"wait_for_model": True}}
        data = f.read()
    response = requests.request("POST", API_URL, headers=headers, data=data)
    # print(response)
    return json.loads(response.content.decode("utf-8"))

# output = query("insp_img/apple.jpg")

# print(output)