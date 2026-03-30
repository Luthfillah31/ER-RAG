import bz2
import json
import os
from datetime import datetime

from loguru import logger
from openai import APIConnectionError, OpenAI, RateLimitError
from prompts.templates import IN_CONTEXT_EXAMPLES, INSTRUCTIONS
from tqdm.auto import tqdm
from transformers import LlamaTokenizerFast

# Load the tokenizer once at the top
tokenizer = LlamaTokenizerFast.from_pretrained("NousResearch/Meta-Llama-3-8B-Instruct")

def get_system_message():
    """Returns the system message containing instructions and in context examples."""
    return INSTRUCTIONS + IN_CONTEXT_EXAMPLES

def attempt_api_call(client, model_name, messages, max_retries=10):
    """Attempt an OpenAI API call with retries."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content
        except (APIConnectionError, RateLimitError):
            logger.warning(f"API call failed on attempt {attempt + 1}, retrying...")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            break
    return None

def parse_response(resp: str):
    """Parse auto-eval output from the evaluator."""
    try:
        resp = resp.lower()
        model_resp = json.loads(resp)
        if model_resp.get("accuracy") is True or str(model_resp.get("accuracy")).lower() == "true":
            return 1
        return -1
    except:
        return -1

def trim_predictions_to_max_token_length(prediction):
    """Trims prediction output to 75 tokens"""
    max_token_length = 100
    tokenized_prediction = tokenizer.encode(prediction)
    # We skip the BOS token (index 0) and take the next 75
    trimmed_tokenized_prediction = tokenized_prediction[1: max_token_length+1]
    return tokenizer.decode(trimmed_tokenized_prediction)

def generate_predictions(dataset_path, participant_model): 
    predictions = [] 
    
    with bz2.open(dataset_path, 'rt', encoding='utf-8') as f:
        # We add 'enumerate' so Python counts which question we are on
        for i, line in enumerate(tqdm(f, desc="Generating Predictions")):
            
 
            if i >= 30:
                break
                
            try:
                data = json.loads(line)
                query = data["query"]
                web_search_results = data.get("search_results", [])
                
                # --- 1. EXTRACT THE TIME FROM THE DATASET ---
                query_time = data.get("query_time", "2024-03-08 00:00:00") 
                
                # --- 2. PASS THE TIME TO THE GENERATOR ---
                prediction = participant_model.generate_answer(query, web_search_results, query_time)
                prediction = trim_predictions_to_max_token_length(prediction)
                
                predictions.append({
                    "query": query,
                    "ground_truth": str(data.get("answer", "")).strip().lower(),
                    "prediction": str(prediction).strip().lower(),
                })
            except Exception as e:
                logger.error(f"Error processing line: {e}")
                continue

    return predictions

def evaluate_predictions(predictions, evaluation_model_name, openai_client):
    n_miss, n_correct, n_correct_exact = 0, 0, 0
    system_message = get_system_message()

    for prediction_dict in tqdm(predictions, desc="Evaluating Predictions"):
        query, ground_truth, prediction = (
            prediction_dict["query"],
            prediction_dict["ground_truth"],
            prediction_dict["prediction"],
        )

        if prediction in ["i don't know", "i don't know."]:
            n_miss += 1
            continue
        
        if prediction == ground_truth:
            n_correct_exact += 1
            n_correct += 1
            continue

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Question: {query}\n Ground truth: {ground_truth}\n Prediction: {prediction}\n"},
        ]

        response = attempt_api_call(openai_client, evaluation_model_name, messages)
        if response:
            eval_res = parse_response(response)
            if eval_res == 1:
                n_correct += 1

    n = len(predictions) if len(predictions) > 0 else 1
    results = {
        "score": (2 * n_correct + n_miss) / n - 1,
        "accuracy": n_correct / n,
        "total": len(predictions),
    }
    logger.info(results)
    return results

if __name__ == "__main__":
    from models.user_config import UserModel

    DATASET_PATH = "example_data/dev_data.jsonl.bz2"

    # 1. Start the Participant Model
    participant_model = UserModel()
    
    # 2. Generate predictions
    predictions = generate_predictions(DATASET_PATH, participant_model)

    # 3. Just print the results (Bypassing OpenAI)
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    for p in predictions:
        print(f"Query: {p['query']}")
        print(f"Prediction: {p['prediction']}")
        print(f"Ground Truth: {p['ground_truth']}")
        print("-" * 30)