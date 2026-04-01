import torch
import time
import requests
import json
import re

# Keep Tokenizer for text formatting and lengths
from transformers import AutoTokenizer

import models.Retriever as Retriever
from models.Parse import parse_answer, finance_parse_answer, music_parse_answer, sports_parse_answer, open_parse_answer
from models.prompt_api import template_map

class RAGModel:
    def __init__(self):
        self.Task = 3
        
        print("-------------------------Configuring Ollama Cloud--------------------------")
        t1 = time.time()
        
        # --- OLLAMA CLOUD CONFIGURATION ---
        self.ollama_api_url = "http://localhost:11434/api/generate"
        self.ollama_model = "deepseek-v3.1:671b-cloud" # Change to deepseek v1 if needed
        
        # Keep Tokenizer to maintain prompt formatting and chunking rules
        model = "NousResearch/Meta-Llama-3-8B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model)

        num_gpus = torch.cuda.device_count()

        if not torch.cuda.is_available() or num_gpus == 0:
            print("[DEBUG] CUDA not compiled or no GPU found. Using CPU for embeddings.")
            self.used1 = "cpu"
            self.used2 = "cpu"
            self.used = "cpu"
        elif num_gpus == 1:
            self.used1 = "cuda:0"
            self.used2 = "cuda:0"
            self.used = "cuda:0"
        else:
            self.used1 = "cuda:1"
            self.used2 = "cuda:1"
            self.used = "cuda:1"

        print("finish configuring LLM via Ollama", time.time() - t1)

        print("-------------------------Loading RET----------------------")
        t1 = time.time()
        
        self.k = 20
        self.r = Retriever.Retriever2(batch_size=64, device1=self.used1, device2=self.used2,
                                      hf_path="models/all-Mini-L6-v2", parent_chunk_size=2000, parent_chunk_overlap=400,
                                      child_chunk_size=200, child_chunk_overlap=50)

        print("finish loading RET", time.time() - t1)
        self.r.clear()

    def llama3_domain(self, query, search_results=None):
        context_hint = ""
        # FIX: Give the classifier a tiny hint of context so it doesn't guess blindly
        if search_results and len(search_results) > 0:
             snippet = search_results[0].get('page_snippet', '')
             context_hint = f"\n Context hint: {snippet[:150]}"
             
        messages = [
            {"role": "system", "content": f"You are an assistant expert in movie, sports, finance and music fields."},
            {"role": "user",
             "content": "Please judge which category the query belongs to, without answering the query. you can only and must output one word in (movie, sports, finance, music) If the question doesn't belong to movie, sports,finance, music, please answer open. \n Query:" + query + context_hint + '\n Category:'},
        ]
        domain, _, _ = self.llam3_output(messages, maxtoken=200)
        
        for key in ['finance', 'music', 'sports', 'movie']:
            if key in domain:
                return key
        return 'open'

    def llam3_output(self, messages, maxtoken=500, disable_adapter=False):
        # Increased maxtoken to 500 so DeepSeek has room to "think" before answering
        import time
        import requests
        t1 = time.time()

        try:
            print(f"\n[DEBUG {time.strftime('%H:%M:%S')}] Sending request to Ollama Chat API ({self.ollama_model})...")
            
            # Use /api/chat and pass the messages list directly!
            # This prevents Llama-3 tokens from poisoning the DeepSeek prompt.
            chat_url = self.ollama_api_url.replace("/api/generate", "/api/chat")
            if not chat_url.endswith("/api/chat"):
                 chat_url = "http://localhost:11434/api/chat"
                 
            response = requests.post(chat_url, json={
                "model": self.ollama_model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.1,  
                    "top_p": 0.9,
                    "num_predict": maxtoken 
                }
            }, timeout=120) 
            
            if response.status_code == 429:
                return "i don't know", 0, 0
                
            response.raise_for_status()
            
            # Extract output correctly from the Chat API response
            output = response.json().get('message', {}).get('content', '').lower().strip()
            
            print(f"[DEBUG {time.strftime('%H:%M:%S')}] Success! Received response in {time.time() - t1:.2f} seconds.")
            return output, 0, 0
            
        except requests.exceptions.Timeout:
            print(f"[DEBUG {time.strftime('%H:%M:%S')}] ERROR: Request timed out.")
            return "i don't know", 0, 0
        except Exception as e:
            print(f"[DEBUG {time.strftime('%H:%M:%S')}] ERROR: {type(e).__name__}: {str(e)}")
            return "i don't know", 0, 0

    def get_batch_size(self) -> int:
        return 16

    def batch_generate_answer(self, batch):
        self.all_st = time.time()
        self.all_time = 16 * 29
        answer = []
        queries = batch["query"]
        batch_search_results = batch["search_results"]
        query_times = batch["query_time"]
        for a, b, c in zip(queries, batch_search_results, query_times):
            if time.time() - self.all_st >= self.all_time:
                answer.append("i don't know")
            else:
                answer.append(self.generate_answer(a, b, c))
        return answer

    def process_api(self, domain, query, query_time):
        print('api prompt')
        t1 = time.time()
        if domain in ['finance']:
            from models.prompt_api import finance_prompt
            filled_template = finance_prompt.format(query_str=query, time_str=query_time)
        elif domain in ['movie']:
            from models.prompt_api import movie_prompt
            filled_template = movie_prompt.format(query_str=query)
        elif domain in ['music']:
            from models.prompt_api import music_prompt
            filled_template = music_prompt.format(query_str=query)
        elif domain in ['sports']:
            from models.prompt_api import sports_prompt
            filled_template = sports_prompt.format(query_str=query, query_time=query_time)
        elif domain in ['open']:
            from models.prompt_api import open_prompt
            filled_template = open_prompt.format(query_str=query)
            
        messages = [
            {"role": "system",
             "content": f"You are a helpful and honest assistant. Please, respond concisely and truthfully in 50 words or less. Now is {query_time}"},
            {"role": "user", "content": filled_template},
        ]
        
        output, minn_logit, mean_logit = self.llam3_output(messages, maxtoken=500)
        print("edn api prompt", output, time.time() - t1)
        
        if domain in ['finance']:
            res, res_str = finance_parse_answer(output, query_time)
        elif domain in ['movie']:
            res, res_str = parse_answer(output)
        elif domain in ['music']:
            res_str = music_parse_answer(output)
        elif domain in ['sports']:
            res_str = sports_parse_answer(output, query_time)
            if len(res_str) == 1 and 'There is no match' in res_str[0]:
                if 'update' in query or 'at the moment' in query or 'week' in query or 'last' in query or 'yesterday' in query or 'previous' in query or 'late' in query or 'today' in query:
                    print('no record')
                    return 'invalid question'
        elif domain in ['open']:
            res_str = open_parse_answer(output)
            
        print("end parse_answer", res_str, time.time() - t1)
        
        if res_str != []:
            context_str = ""
            for snippet in res_str[:]:
                context_str += "<DOC>\n" + snippet + "\n</DOC>\n"
                
            context_str_tokens = self.tokenizer.encode(context_str, max_length=20000, truncation=True, add_special_tokens=False)
            print('len context_str', len(context_str_tokens))
            
            if len(context_str_tokens) >= 4000:
                context_str = self.tokenizer.decode(context_str_tokens) + "\n</DOC>\n"
            else:
                context_str = self.tokenizer.decode(context_str_tokens)
                
            if domain in ["sports"]:
                filled_template = template_map['template_output_answer'].format(context_str=context_str, query_str=query)
                messages = [
                    {"role": "system",
                     "content": f"You are a helpful and honest assistant. Please, respond concisely and truthfully in 70 words or less. Now is {query_time}"},
                    {"role": "user", "content": filled_template},
                ]
            else:
                filled_template = template_map['output_answer_api'].format(context_str=context_str, query_str=query)
                messages = [
                    {"role": "system",
                     "content": f"You are a helpful and honest assistant. Please, respond concisely and truthfully in 50 words or less. If you are not sure about the query, answer i don't know. Now is {query_time}"},
                    {"role": "user", "content": filled_template},
                ]
                
            output, minn_logit, mean_logit = self.llam3_output(messages, maxtoken=500)
            print("edn api", time.time() - t1)
            
            if "i don't know" not in output:
                if 'invalid' in output:
                    output = "i don't know"
                return output
                
        return "i don't know"

    def process_task1(self, domain, query, query_time):
        context_str = ""
        output = ""
        if domain in ['movie']:
            context_str = self.r.get_movie_oscar(query)
            if context_str is not None:
                t1 = time.time()
                filled_template = template_map['output_answer_nofalse'].format(context_str=context_str, query_str=query)
                messages = [
                    {"role": "system",
                     "content": f"You are a helpful and honest assistant. Please, respond concisely and truthfully in 30 words or less. If you are not sure about the query, answer i don't know. There is no need to explain the reasoning behind your answers. Now is {query_time}"},
                    {"role": "user", "content": filled_template},
                ]
                output, minn_logit, mean_logit = self.llam3_output(messages, maxtoken=500)
                print("end oscar", time.time() - t1)
                if "i don't know" not in output and "invalid" not in output:
                    return output, context_str
            else:
                context_str = ""
                t1 = time.time()
                filled_template = template_map['ask_name'].format(query_str=query)
                messages = [
                    {"role": "system",
                     "content": f" You will be asked a lot of questions, but you don't need to answer them, just point out the name of the movie involved."},
                    {"role": "user", "content": filled_template},
                ]
                output, minn_logit, mean_logit = self.llam3_output(messages, maxtoken=500)
                print("end ask movie name", time.time() - t1)
                if "i don't know" not in output:
                    try:
                        for tmpoutput in output.split(' && '):
                            tmpoutput = tmpoutput.replace('"', '').strip()
                            context_str += self.r.get_movie_context(tmpoutput)
                    except:
                        context_str = ""
                else:
                    context_str = ""
        elif domain in ['music']:
            context_str = self.r.get_music_grammy(query)
            if context_str is None:
                context_str = ""
            else:
                t1 = time.time()
                filled_template = template_map['output_answer_nofalse'].format(context_str=context_str, query_str=query)
                messages = [
                    {"role": "system",
                     "content": f"You are a helpful and honest assistant. Please, respond concisely and truthfully in 30 words or less. If you are not sure about the query, answer i don't know. There is no need to explain the reasoning behind your answers. Now is {query_time}"},
                    {"role": "user", "content": filled_template},
                ]
                output, minn_logit, mean_logit = self.llam3_output(messages, maxtoken=500)
                print("edn music", output, time.time() - t1)
                if "i don't know" not in output and "invalid" not in output:
                    return output, context_str
                context_str = ""
        elif domain in ['finance']:
            if 'share' in query or 'pe' in query or 'eps' in query or 'ratio' in query or 'capitalization' in query or 'earnings' in query or 'market' in query:
                context_str = ""
                t1 = time.time()
                filled_template = template_map['ask_name_finance'].format(query_str=query)
                messages = [
                    {"role": "system",
                     "content": f" You will be asked a lot of questions, but you don't need to answer them, just point out the specific stock ticker or company name involved."},
                    {"role": "user", "content": filled_template},
                ]
                output, minn_logit, mean_logit = self.llam3_output(messages, maxtoken=500)
                print("edn ask name", output, time.time() - t1)
                if "i don't know" not in output and 'none' not in output:
                    try:
                        for tmpoutput in output.split(' && '):
                            tmpoutput = tmpoutput.replace('"', '').strip()
                            context_str += self.r.get_finance_context(tmpoutput)
                        t1 = time.time()
                        filled_template = template_map['output_answer_nofalse'].format(context_str=context_str, query_str=query)
                        messages = [
                            {"role": "system",
                             "content": f"You are a helpful and honest assistant. Please, respond concisely and truthfully in 30 words or less. If you are not sure about the query, answer i don't know. There is no need to explain the reasoning behind your answers. Now is {query_time}"},
                            {"role": "user", "content": filled_template},
                        ]
                        output, minn_logit, mean_logit = self.llam3_output(messages, maxtoken=500)
                        print("edn finance", time.time() - t1)
                        if "i don't know" not in output and "invalid" not in output:
                            return output, context_str
                        context_str = ""
                    except:
                        context_str = ""
                else:
                    context_str = ""
        return "", context_str

    def generate_answer(self, query, search_results, query_time=None) -> str:
        print("\n-------------Now Querying----------------")
        print(query)

        self.t_s = time.time()
        self.r.clear()

        print("determine compare")
        # FIX: Pass search results so the classifier has a hint
        domain = self.llama3_domain(query, search_results) 
        print("judge domain", domain)
        context_str = ""
        
        if self.Task >= 2:
            apioutput = self.process_api(domain, query, query_time)
            if ("i don't know" not in apioutput):
                return apioutput
        elif self.Task == 1:
            output, context_str = self.process_task1(domain, query, query_time)
            if output!="":
                return output
                
        # FIX: Restore the actual Retriever pipeline
        print("[DEBUG] Initializing Retriever...")
        retriever_success = self.r.init_retriever(search_results, recall_k=200, task3_topk=30, query=query)
        
        if retriever_success:
            docs = self.r.get_result(query, k=30)
            for doc in docs:
                context_str += f"<DOC>\n{doc}\n</DOC>\n"
        else:
            print("[DEBUG] Retriever failed or no docs. Filtering snippets with LLM.")
            raw_snippets = ""
            # Gather snippets to present to the LLM
            for i, snippet in enumerate(search_results[:8]): 
                text = snippet.get('page_snippet', snippet.get('page_result', ''))[:400]
                raw_snippets += f"[Snippet {i+1}]: {text}\n"
            
            # Ask the LLM to extract only relevant facts
            filter_prompt = f"""You are an expert fact-checker. Read the following search snippets and extract ONLY the facts that directly answer the query. If the snippets do not contain the answer, you MUST reply exactly with "NO_RELEVANT_INFO". Do not guess.

Query: {query}

Snippets:
{raw_snippets}

Extracted Facts:"""

            filter_messages = [
                {"role": "system", "content": "You are a strict information extractor."},
                {"role": "user", "content": filter_prompt}
            ]
            
            # Call Ollama to filter the snippets
            filtered_info, _, _ = self.llam3_output(filter_messages, maxtoken=300)
            
            # If the LLM determines the snippets are garbage, leave context empty to trigger "I don't know"
            if "NO_RELEVANT_INFO" in filtered_info or "no relevant info" in filtered_info.lower():
                print("[DEBUG] LLM determined snippets are irrelevant. Proceeding with empty context.")
                context_str += "" 
            else:
                print(f"[DEBUG] LLM extracted relevant info: {filtered_info[:100]}...")
                context_str += f"<DOC>\n{filtered_info}\n</DOC>\n"
            
        context_str_tokens = self.tokenizer.encode(context_str, max_length=20000, truncation=True, add_special_tokens=False)
        context_str = self.tokenizer.decode(context_str_tokens)
            
        filled_template = template_map['output_answer_nofalse'].format(context_str=context_str, query_str=query)

        messages = [
            {"role": "system",
             "content": f"You are a helpful and honest assistant. Please, respond concisely and truthfully in 70 words or less. Now is {query_time}"},
            {"role": "user", "content": filled_template},
        ]

        output, minn_logit, mean_logit = self.llam3_output(messages)

        if "i don't know" not in output and output not in ['i', "i don't"]:
            return output
        else:
            return "i don't know"