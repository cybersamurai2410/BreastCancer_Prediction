import time
import json
import joblib
import numpy as np
from PIL import Image
from keras.models import load_model
from assistants import client, retrieve_assistant, retrieve_thread, create_assistant  

# Load machine learning models
sklearn_model = joblib.load("breast_cancer_sklearn.joblib")
keras_model = load_model("breast_cancer_keras.h5")

def run_sklearn_inference(data):
    """Perform breast cancer classification using scikit-learn."""
    features = data.get("features")
    if not features or not isinstance(features, list) or len(features) == 0:
        return {"error": "Invalid or missing features."}

    features_arr = np.array(features).reshape(1, -1)
    prediction = sklearn_model.predict(features_arr)
    return {"prediction": prediction.tolist()}

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
    if image_arr.shape[-1] == 1:
        image_arr = np.repeat(image_arr, 3, axis=-1)

    image_arr = np.expand_dims(image_arr, axis=0)
    prediction = keras_model.predict(image_arr)
    predicted_class = int(np.argmax(prediction, axis=1)[0])
    return {"prediction": predicted_class}

def classify_cancer(user_input, thread_id=None):
    """Handles both structured data and image classification requests."""

    # Retrieve existing thread if thread_id is provided, otherwise create a new one
    if thread_id:
        try:
            thread = retrieve_thread(thread_id)
            print(f"Reusing existing thread: {thread.id}")
        except Exception as e:
            print(f"Thread retrieval failed ({e}), creating a new thread.")
            thread = client.beta.threads.create()
    else:
        thread = client.beta.threads.create()

    # Send the user message to the assistant
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_input
    )

    # Retrieve the existing assistant, or create a new one if needed
    assistant_id = "asst_abc123"  # Replace with actual assistant ID
    try:
        assistant = retrieve_assistant(assistant_id)
    except Exception as e:
        print(f"Assistant retrieval failed ({e}), creating a new assistant.")
        assistant = create_assistant()

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

            if func_name == "run_sklearn_inference":
                result = run_sklearn_inference(func_args)
            elif func_name == "run_keras_inference":
                result = run_keras_inference(func_args)
            else:
                result = {"error": f"Unknown function: {func_name}"}

            tool_outputs.append({
                "tool_call_id": tool_call["id"],
                "output": json.dumps(result)
            })

        if tool_outputs:
            try:
                run = client.beta.threads.runs.submit_tool_outputs_and_poll(
                    thread_id=thread.id,
                    run_id=run.id,
                    tool_outputs=tool_outputs
                )
            except Exception as e:
                print("Failed to submit tool outputs:", e)

    # Retrieve final response
    if run.status == "completed":
        messages = client.beta.threads.messages.list(thread_id=thread.id)
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

    return thread.id  # Return the thread ID for reuse

if __name__ == "__main__":
    thread_id = "thread_abc123"  # Use a stored thread ID if available

    print("\nRunning ensemble learning model...")
    thread_id = classify_cancer("Classify these features: [5.1, 3.5, 1.4, 0.2]", thread_id=thread_id)

    print("\nRunning image classification...")
    thread_id = classify_cancer("Classify this image: breast_scan.jpg", thread_id=thread_id)
