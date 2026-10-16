from smolagents import CodeAgent, LiteLLMModel
import os
import dotenv  

dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 1. Setup the Model
model_id = "gemini/gemini-3.8-flash"
model = LiteLLMModel(model_id=model_id, api_key=GEMINI_API_KEY)

# 3. Initialize the CodeAgent (and pass the system prompt here!)
agent = CodeAgent(
    tools=[],  # Uses built-in python interpreter and base tools
    model=model,
    
    add_base_tools=True,
    additional_authorized_imports=[  # Strictly limit allowed libraries for security
        "pandas", "numpy", "datetime",
        "matplotlib", "plotly", "seaborn", "sklearn"
    ]
)

# 4. Run the Agent
# We pass `input_csv` via `additional_args` so the agent has access to the file path variable
task = "Calculate the average prices by airline from the dataset, create a bar plot visualization, and summarize the findings."

agent.run(
    task,
    additional_args={"input_csv": "flight_price.csv"},  # Replace with your actual CSV path,
    max_steps=4
)