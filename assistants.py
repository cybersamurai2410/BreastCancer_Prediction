import os 
from openai import OpenAI

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def retrieve_assistant(assistant_id):
    """Retrieves an existing assistant by its ID."""
    return client.beta.assistants.retrieve(assistant_id)

def retrieve_thread(thread_id):
    """Retrieves an existing conversation thread by its ID."""
    return client.beta.threads.retrieve(thread_id)

def create_assistant():
    """Creates a new assistant with function schemas for classification."""
    assistant = client.beta.assistants.create(
        name="Breast Cancer Prediction Assistant",
        instructions=(
            "Classify breast cancer based on either structured numerical data or an image. "
            "The assistant must analyze the model's classification results and generate a detailed, "
            "context-aware explanation and diagnosis, considering risk factors, potential follow-up steps, "
            "and medical recommendations."
        ),
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
    
    return assistant
