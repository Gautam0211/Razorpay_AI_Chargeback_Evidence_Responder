import os
from langsmith import Client
from langsmith.evaluation import evaluate
from openai import OpenAI

# Initialize clients
ls_client = Client()
openai_client = OpenAI()

# 1. Define the target task/LLM application to test
def predict_llm_app(inputs: dict) -> dict:
    """
    Your actual LLM application logic goes here.
    It takes an input dictionary and returns an output dictionary.
    """
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Give short answers."},
            {"role": "user", "content": inputs["question"]}
        ]
    )
    return {"output": response.choices[0].message.content}

# 2. Define a custom evaluator (Optional)
def correctness_evaluator(run, example) -> dict:
    """
    A simple exact/partial match evaluator. 
    You can also use LLM-as-a-judge here.
    """
    student_answer = run.outputs.get("output", "").lower()
    reference_answer = example.outputs.get("answer", "").lower()
    
    score = 1 if reference_answer in student_answer else 0
    return {"key": "correctness", "score": score}

def main():
    dataset_name = "Sample_QA_Dataset"
    
    # 3. Create a dummy dataset if it doesn't exist in LangSmith
    if not ls_client.has_dataset(dataset_name=dataset_name):
        dataset = ls_client.create_dataset(
            dataset_name=dataset_name,
            description="A simple dataset to test tester.py",
        )
        # Add sample data (inputs and expected outputs)
        ls_client.create_examples(
            inputs=[
                {"question": "What is the capital of France?"},
                {"question": "Who wrote Romeo and Juliet?"}
            ],
            outputs=[
                {"answer": "Paris"},
                {"answer": "William Shakespeare"}
            ],
            dataset_id=dataset.id,
        )
        print(f"Created dataset: {dataset_name}")

    # 4. Run the evaluation
    print("Starting LangSmith evaluation...")
    experiment_results = evaluate(
        predict_llm_app,
        data=dataset_name,
        evaluators=[correctness_evaluator],
        experiment_prefix="test-run",
    )
    print("Evaluation complete! View results in your LangSmith dashboard.")

if __name__ == "__main__":
    main()
