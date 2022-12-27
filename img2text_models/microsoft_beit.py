import requests
import json
import config
import time

API_URL = "https://api-inference.huggingface.co/models/microsoft/beit-base-patch16-224-pt22k-ft22k"
headers = {"Authorization": f"Bearer %s" %config.api_img2txt}

def query(filename):
    with open(filename, "rb") as f:
        # data = {"inputs": f.read(), "options": {"wait_for_model": True}}
        data = f.read()
    response = requests.request("POST", API_URL, headers=headers, data=data)
    # print(response)
    if response.status_code == 503:
        print("retry, sleep 20s")
        time.sleep(20)
        response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))

# output = query("insp_img/apple.jpg")

# print(output)