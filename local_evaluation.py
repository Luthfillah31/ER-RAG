import bz2
import json
import os
import time
import requests
from datetime import datetime
import re
from loguru import logger
from prompts.templates import IN_CONTEXT_EXAMPLES, INSTRUCTIONS
from tqdm.auto import tqdm
from transformers import LlamaTokenizerFast

# Load the tokenizer once at the top
tokenizer = LlamaTokenizerFast.from_pretrained("NousResearch/Meta-Llama-3-8B-Instruct")

def get_system_message():
    """Returns the system message containing instructions and in context examples."""
    return INSTRUCTIONS + IN_CONTEXT_EXAMPLES

def attempt_ollama_call(model_name, messages, max_retries=3):
    """Attempt an Ollama API call with retries."""
    url = "http://localhost:11434/api/chat"
    
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "format": "json", # Force JSON output
        "options": {
            "temperature": 0.0, # Keep it deterministic for evaluation
            "num_predict": 100
        }
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=120)
            
            if response.status_code == 429:
                logger.warning(f"Rate limit hit on attempt {attempt + 1}, retrying in 2s...")
                time.sleep(2)
                continue
                
            response.raise_for_status()
            
            # Extract the content from the Chat API response
            return response.json().get("message", {}).get("content", "")
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Ollama call failed on attempt {attempt + 1}: {e}")
            
    logger.error("Failed to get response from Ollama after maximum retries.")
    return None

def parse_response(resp: str):
    """Parse auto-eval output from the evaluator, handling Markdown artifacts."""
    if not resp:
        return -1
        
    try:
        resp = resp.lower().strip()
        
        # Safely strip markdown code blocks if the model wrapped the JSON
        if resp.startswith("```json"):
            resp = resp.replace("```json", "").replace("```", "").strip()
        elif resp.startswith("```"):
            resp = resp.replace("```", "").strip()
            
        model_resp = json.loads(resp)
        accuracy = model_resp.get("accuracy")
        
        if accuracy is True or str(accuracy).lower() == "true":
            return 1
        return -1
    except Exception as e:
        logger.error(f"Failed to parse JSON response: {resp}. Error: {e}")
        return -1

def trim_predictions_to_max_token_length(prediction):
    """Trims prediction output to max tokens and strips DeepSeek think blocks."""
    if not prediction:
        return "i don't know"
        
    # Remove DeepSeek <think>...</think> blocks if present
    prediction = re.sub(r'<think>.*?</think>', '', prediction, flags=re.DOTALL).strip()
    
    # If the string is empty after removing the think block
    if not prediction:
        return "i don't know"
        
    max_token_length = 100
    tokenized_prediction = tokenizer.encode(prediction)
    trimmed_tokenized_prediction = tokenized_prediction[1: max_token_length+1]
    return tokenizer.decode(trimmed_tokenized_prediction)

def generate_predictions(dataset_path, participant_model): 
    predictions = [] 
    
    with bz2.open(dataset_path, 'rt', encoding='utf-8') as f:
        # We add 'enumerate' so Python counts which question we are on
        for i, line in enumerate(tqdm(f, desc="Generating Predictions")):

            if i >= 30: # Limit to 30 for testing
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
                logger.error(f"Error processing line {i}: {e}")
                continue

    return predictions

def evaluate_predictions(predictions, evaluation_model_name):
    n_miss, n_correct, n_correct_exact = 0, 0, 0
    system_message = get_system_message()

    for prediction_dict in tqdm(predictions, desc="Evaluating Predictions with Ollama Judge"):
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
        
        # Build the Chat payload
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Question: {query}\n Ground truth: {ground_truth}\n Prediction: {prediction}\n"},
        ]

        response = attempt_ollama_call(evaluation_model_name, messages)
        
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
    logger.info(f"Final Evaluation Results: {results}")
    return results

if __name__ == "__main__":
    from models.user_config import UserModel

    DATASET_PATH = "example_data/dev_data.jsonl.bz2"
    
    # Set your Ollama model here
    OLLAMA_JUDGE_MODEL = "deepseek-v3.1:671b-cloud"

    # 1. Start the Participant Model
    participant_model = UserModel()
    
    # 2. Generate predictions
    predictions = generate_predictions(DATASET_PATH, participant_model)

    # 3. Use the Ollama LLM Judge to Evaluate
    print("\n" + "="*50)
    print(f"EVALUATING WITH OLLAMA JUDGE ({OLLAMA_JUDGE_MODEL})")
    print("="*50)
    
    try:
        results = evaluate_predictions(predictions, OLLAMA_JUDGE_MODEL) 
        
        print("\nFINAL SCORES:")
        print(f"Accuracy: {results['accuracy']:.2%}")
        print(f"CRAG Score: {results['score']:.4f}")
        
    except Exception as e:
        print(f"Failed to run Ollama Evaluation: {e}")
        print("Ensure your Ollama server is running at http://localhost:11434")

    print("\n" + "="*50)
    print("PREDICTION LOG")
    print("="*50)
    for p in predictions:
        print(f"Query: {p['query']}")
        print(f"Prediction: {p['prediction']}")
        print(f"Ground Truth: {p['ground_truth']}")
        print("-" * 30)