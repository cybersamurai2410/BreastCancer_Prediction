import time
import json
import joblib
import numpy as np
from PIL import Image
from keras.models import load_model
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

# Load machine learning models
sklearn_model = joblib.load("breast_cancer_sklearn.joblib")
keras_model = load_model("breast_cancer_keras.h5")

# Create an AI Assistant with function calling
assistant = client.beta.assistants.create(
    name="Breast Cancer Prediction Assistant",
    instructions="Classify breast cancer based on either structured numerical data or an image. The assistant must analyze the model's classification results and generate a detailed, context-aware explanation and diagnosis, considering risk factors, potential follow-up steps, and medical recommendations.",
    model="gpt-4o",
    tools=[
        {
            "type": "function",
            "function": {
                "name": "run_sklearn_inference",
                "description": "Predict breast cancer based on structured numerical features.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "features": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "List of numerical features."
                        }
                    },
                    "required": ["features"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "run_keras_inference",
                "description": "Predict breast cancer using a medical image.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {
                            "type": "string",
                            "description": "Path to the medical image."
                        }
                    },
                    "required": ["image_path"]
                }
            }
        }
    ]
)

# Define model inference functions
def run_sklearn_inference(data):
    """Perform breast cancer classification using scikit-learn."""
    features = data.get("features")
    if not features or not isinstance(features, list) or len(features) == 0:
        return {"error": "Invalid or missing features."}

    features_arr = np.array(features).reshape(1, -1)
    prediction = sklearn_model.predict(features_arr)
    return {"prediction": prediction.tolist()}  # Assistant will generate explanation

def run_keras_inference(data):
    """Perform breast cancer classification using Keras (image model)."""
    image_path = data.get("image_path")
    if not image_path:
        return {"error": "No image path provided."}

    # Preprocess image
    image = Image.open(image_path).resize((224, 224))
    image_arr = np.array(image) / 255.0  # Normalize pixel values

    # Ensure the image has 3 channels
    if len(image_arr.shape) == 2:
        image_arr = np.stack([image_arr] * 3, axis=-1)
    elif image_arr.shape[2] == 4:
        image_arr = image_arr[..., :3]

    # Convert grayscale to RGB
    if image_arr.shape[-1] == 1:
        image_arr = np.repeat(image_arr, 3, axis=-1)

    image_arr = np.expand_dims(image_arr, axis=0)
    prediction = keras_model.predict(image_arr)
    predicted_class = int(np.argmax(prediction, axis=1)[0])
    return {"prediction": predicted_class}  # Assistant will generate explanation

# Function to handle assistant interaction
def classify_cancer(user_input):
    """Handles both structured data and image classification requests."""
    
    # Create a thread for the conversation
    thread = client.beta.threads.create()
    
    # Send the user message to the assistant
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_input
    )

    # Run the assistant and wait for the result
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant.id
    )

    # If the assistant requests a function call
    if run.status == "requires_action" and "submit_tool_outputs" in run.required_action:
        tool_calls = run.required_action["submit_tool_outputs"]["tool_calls"]
        tool_outputs = []

        for tool_call in tool_calls:
            func_name = tool_call["function"]["name"]
            func_args = tool_call["function"]["arguments"]
            if not isinstance(func_args, dict):
                func_args = json.loads(func_args)

            # Run the appropriate model
            if func_name == "run_sklearn_inference":
                result = run_sklearn_inference(func_args)
            elif func_name == "run_keras_inference":
                result = run_keras_inference(func_args)
            else:
                result = {"error": f"Unknown function: {func_name}"}

            # Prepare tool output
            tool_outputs.append({
                "tool_call_id": tool_call["id"],
                "output": json.dumps(result)
            })

        # Send function results back to OpenAI if tool_outputs exist
        if tool_outputs:
            try:
                run = client.beta.threads.runs.submit_tool_outputs_and_poll(
                    thread_id=thread.id,
                    run_id=run.id,
                    tool_outputs=tool_outputs
                )
                if run.status != "completed":
                    print(f"Assistant did not complete successfully. Status: {run.status}")
            except Exception as e:
                print("Failed to submit tool outputs:", e)
        else:
            print("No tool outputs to submit.")

    # Retrieve final response
    if run.status == "completed":
        messages = client.beta.threads.messages.list(thread_id=thread.id)

        # Ensure response is printed properly
        if messages.data:
            last_message = messages.data[-1].content
            if last_message and isinstance(last_message, list) and len(last_message) > 0:
                print("Final response:", last_message[0]["text"]["value"])
            else:
                print("Final response: [Empty Message]")
        else:
            print("Final response: [No Messages]")
    else:
        print("Run status:", run.status)

# Main execution block
if __name__ == "__main__":
    print("\nRunning structured data classification...")
    classify_cancer("Classify these features: [5.1, 3.5, 1.4, 0.2]")

    print("\nRunning image classification...")
    classify_cancer("Classify this image: breast_scan.jpg")
