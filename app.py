import os
import dotenv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from smolagents import CodeAgent, LiteLLMModel, tool

dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

CSV_FILE = "flight_price.csv"
MODEL_ID = "gemini/gemini-3.8-flash"

os.makedirs("outputs", exist_ok=True)
df = pd.read_csv(CSV_FILE)
state = {"df": df}

@tool
def describe_data() -> str:
    df = state["df"]
    return f"{df.shape[0]} rows, {df.shape[1]} columns\n{df.dtypes}\n\n{df.describe(include='all').T.head(20)}"

@tool
def check_quality() -> str:
    df = state["df"]
    missing = df.isna().sum()
    return f"Duplicate rows: {df.duplicated().sum()}\nMissing values:\n{missing[missing > 0].to_string() or 'none'}"

@tool
def group_summary(group_col: str, value_col: str, how: str = "mean") -> str:
    out = state["df"].groupby(group_col)[value_col].agg(how)
    return out.sort_values(ascending=False).head(20).to_string()

@tool
def top_correlations() -> str:
    c = state["df"].corr(numeric_only=True)
    pairs = c.where(np.triu(np.ones(c.shape, dtype=bool), 1)).stack()
    return pairs.reindex(pairs.abs().sort_values(ascending=False).index).head(10).to_string()

@tool
def quick_plot(kind: str, x: str, y: str = "") -> str:
    df = state["df"]
    fig, ax = plt.subplots(figsize=(8, 5))
    if kind == "hist":
        df[x].plot.hist(ax=ax)
    elif kind == "bar":
        df.groupby(x)[y].mean().plot.bar(ax=ax)
    else:
        df.plot(kind=kind, x=x, y=y, ax=ax)
    path = f"outputs/{kind}_{x}_{y}.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path

model = LiteLLMModel(model_id=MODEL_ID, api_key=GEMINI_API_KEY)
agent = CodeAgent(
    tools=[describe_data, check_quality, group_summary, top_correlations, quick_plot],
    model=model,
    additional_authorized_imports=["pandas", "numpy"],
    max_steps=6,
)

while True:
    question = input("\nAsk a question (or type quit): ")
    if question.lower() == "quit":
        break
    answer = agent.run(
        f"The data is already loaded as a pandas table called df. "
        f"Its columns are {list(df.columns)}. Use the tools when they fit. "
        f"Charts are saved in the 'outputs' folder; tell me the file name. "
        f"Question: {question}",
        additional_args={"df": df},
    )
    print("\nANSWER:", answer)