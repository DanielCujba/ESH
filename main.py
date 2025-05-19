import openai
import json
import chromadb  # For ChromaDB
import os  # For checking if requirements file exists
import hashlib  # For creating deterministic IDs

# --- Debug Configuration ---
DEBUG_LOGS = True  # Set to False to hide detailed step-by-step logs

# Configure the OpenAI client to connect to LM Studio
client = openai.OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# --- Model Configuration ---
ESB_MODEL_ID = "mistral-7b-instruct-v0.3"
FEEDBACK_MODEL_ID = "phi-3-mini-4k-instruct"
GUARDRAIL_MODEL_ID = "qwen1.5-1.8b-chat"
EMBEDDING_MODEL_ID = "text-embedding-nomic-embed-text-v1.5"

# --- RAG Configuration with ChromaDB ---
REQUIREMENTS_FILE_PATH = "requirements.txt"
CHROMA_DB_PATH = "chroma_db_store"  # Path where ChromaDB will store its data
CHROMA_COLLECTION_NAME = "esb_requirements_collection"  # Changed name slightly for clarity
chroma_collection = None  # Will be initialized

# --- Prompt Engineering ---
# ESB_SYSTEM_PROMPT, FEEDBACK_LLM_SYSTEM_PROMPT, GUARDRAIL_LLM_SYSTEM_PROMPT
# remain the same as in the previous complete script version.
ESB_SYSTEM_PROMPT = """
You are an Emotional Support Buddy (ESB). Your primary goal is to offer decent, empathetic advice or praise.
Listen attentively to the user. Be kind, understanding, and supportive.
Avoid giving medical or financial advice. Focus on emotional well-being.
Keep your responses concise but warm.
"""

FEEDBACK_LLM_SYSTEM_PROMPT = """
You are a specialized AI that outputs a SINGLE-LINE JSON object. This JSON object is a directive for an Emotional Support Buddy (ESB) LLM.
Your SOLE output MUST be a valid JSON object on a single line. Do NOT use conversational language or markdown code fences.

The JSON object MUST have a top-level key named "directive".
The value of "directive" MUST be an object containing these EXACT three sub-keys:
1. "action_type": (string) A concise verb or verb phrase for the ESB's primary action.
   Examples: "Congratulate", "AcknowledgeFeelings", "ExpressSympathy", "OfferSupport", "ValidateAndExplore", "PositiveReinforcement".
2. "suggested_tone": (string) Adjectives describing the ESB's desired tone.
   Examples: "Enthusiastic and celebratory", "Warm, understanding, and encouraging", "Very gentle, compassionate, and patient", "Supportive and calm".
3. "focus_hint": (string) A BRIEF note (NOT a full message) on what the ESB should address or be mindful of. This helps guide the ESB's unique response.
   Examples: "User's recent promotion; share excitement", "User's mixed feelings about a situation; gently explore", "User's expressed sadness or loss; offer condolences and listening space".

Example of YOUR EXACT JSON OUTPUT (single-line):
User input: "i just got promoted!!!"
Your output:
{ "directive": { "action_type": "Congratulate", "suggested_tone": "Enthusiastic and celebratory", "focus_hint": "User's excitement about their promotion; share joy and ask for details." } }

Another example:
User input: "I'm feeling really down today."
Your output:
{ "directive": { "action_type": "AcknowledgeFeelingsAndOfferSupport", "suggested_tone": "Gentle, empathetic, and patient", "focus_hint": "User feeling down; validate feelings and offer to listen." } }

If your output is not a valid single-line JSON object with the top-level key "directive" whose value is an object containing "action_type", "suggested_tone", and "focus_hint", you are failing.
Analyze the user's last message and recent conversation, then generate the single-line JSON directive.
"""

GUARDRAIL_LLM_SYSTEM_PROMPT = """
You are a Guardrail LLM. Your task is to evaluate the following candidate message from an Emotional Support Buddy.
Determine if the message is:
1. Toxic or harmful in any way.
2. Inappropriate for an emotional support context (e.g., judgmental, dismissive, giving unqualified advice).
3. Generally unhelpful or counterproductive.

Respond with ONLY 'SAFE' if the message is appropriate and helpful.
Respond with ONLY 'UNSAFE' if the message is toxic, inappropriate, or unhelpful.
"""

conversation_history = []
MAX_HISTORY_LEN = 10


# --- RAG Functions with ChromaDB ---
def get_embedding(text_to_embed, model_id=EMBEDDING_MODEL_ID):
    """Gets an embedding for the given text using the specified model."""
    try:
        if not text_to_embed or not text_to_embed.strip():
            if DEBUG_LOGS: print(f"Attempted to embed an empty or whitespace-only string. Returning None.")
            return None
        response = client.embeddings.create(
            input=[text_to_embed],
            model=model_id
        )
        return response.data[0].embedding
    except Exception as e:
        if DEBUG_LOGS:
            print(f"Error getting embedding for text '{text_to_embed[:50]}...': {e}")
        return None


def initialize_and_populate_chroma_collection():
    """Initializes ChromaDB client, gets/creates a collection, and populates it from the requirements file."""
    global chroma_collection
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        chroma_collection = chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}  # Essential for cosine similarity search
        )

        if DEBUG_LOGS:
            print(f"\n----- Initializing/Populating ChromaDB Collection: {CHROMA_COLLECTION_NAME} -----")
            print(f"Collection item count before processing: {chroma_collection.count()}")

        if not os.path.exists(REQUIREMENTS_FILE_PATH):
            if DEBUG_LOGS:
                print(f"Requirements file not found at {REQUIREMENTS_FILE_PATH}. No new items to populate.")
            return

        with open(REQUIREMENTS_FILE_PATH, 'r', encoding='utf-8') as f:
            requirements_texts = [line.strip() for line in f if line.strip()]

        if not requirements_texts:
            if DEBUG_LOGS:
                print("No requirements found in the file to process.")
            return

        ids_to_add = []
        documents_to_add = []
        embeddings_to_add = []

        # Get all existing document IDs from the collection to check for existence
        # This is more efficient than getting all documents if the collection is large.
        existing_ids_in_chroma = set(chroma_collection.get(include=[])['ids'])

        for req_text in requirements_texts:
            # Create a deterministic ID based on the text content
            # This helps in checking for existence and avoiding duplicates if text is the same
            req_id = "req_" + hashlib.md5(req_text.encode('utf-8')).hexdigest()

            if req_id in existing_ids_in_chroma:
                if DEBUG_LOGS:
                    print(f"Requirement ID '{req_id}' ('{req_text[:30]}...') already in ChromaDB. Skipping.")
                continue

            embedding = get_embedding(req_text)
            if embedding:
                ids_to_add.append(req_id)
                documents_to_add.append(req_text)
                embeddings_to_add.append(embedding)
            else:
                if DEBUG_LOGS:
                    print(f"Could not compute embedding for requirement: {req_text}")

        if documents_to_add:
            chroma_collection.add(
                embeddings=embeddings_to_add,
                documents=documents_to_add,
                ids=ids_to_add
            )
            if DEBUG_LOGS:
                print(
                    f"Added {len(documents_to_add)} new requirements to ChromaDB collection '{CHROMA_COLLECTION_NAME}'.")
        else:
            if DEBUG_LOGS:
                print(f"No new requirements to add to the collection.")

        if DEBUG_LOGS:
            print(f"Collection item count after processing: {chroma_collection.count()}")

    except Exception as e:
        if DEBUG_LOGS:
            print(f"Error initializing or populating ChromaDB: {e}")
        chroma_collection = None


def retrieve_relevant_requirements_from_chroma(query_text, top_k=3, similarity_threshold=0.5):
    """Retrieves the top_k most relevant requirements from ChromaDB."""
    global chroma_collection
    if not query_text or chroma_collection is None:
        if DEBUG_LOGS and chroma_collection is None:
            print("ChromaDB collection not initialized. Cannot retrieve.")
        return []

    query_embedding = get_embedding(query_text)
    if not query_embedding:
        if DEBUG_LOGS: print("Could not get embedding for query. Cannot retrieve.")
        return []

    try:
        results = chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'distances']
        )

        retrieved_requirements = []
        if results and results.get('documents') and results.get('distances'):
            documents = results['documents'][0]
            distances = results['distances'][0]

            if DEBUG_LOGS:
                print(f"\n----- ChromaDB RAG Retrieval for query: '{query_text[:50]}...' -----")

            for i, doc_text in enumerate(documents):
                # ChromaDB with "cosine" space returns distance = 1 - cosine_similarity.
                # So, similarity = 1 - distance.
                similarity_score = 1 - distances[i]

                if DEBUG_LOGS:
                    print(
                        f"  Candidate: '{doc_text[:60]}...' (Distance: {distances[i]:.4f}, Similarity: {similarity_score:.4f})")

                if similarity_score >= similarity_threshold:
                    retrieved_requirements.append(doc_text)
                else:
                    if DEBUG_LOGS:
                        print(
                            f"    -> Rejected due to similarity score {similarity_score:.4f} < threshold {similarity_threshold}")

            if DEBUG_LOGS:
                if retrieved_requirements:
                    print(f"Retrieved {len(retrieved_requirements)} requirements meeting threshold:")
                    for req_idx, req_val in enumerate(retrieved_requirements): print(f"  {req_idx + 1}. {req_val}")
                else:
                    print("No requirements met the similarity threshold from ChromaDB query.")
        else:
            if DEBUG_LOGS: print("No results from ChromaDB query or results format unexpected.")

        return retrieved_requirements

    except Exception as e:
        if DEBUG_LOGS:
            print(f"Error querying ChromaDB: {e}")
        return []


# --- Core LLM Functions ---
def get_llm_response(model_id, system_prompt, messages_payload, temperature=0.7, max_tokens=250):
    """Generic function to get a response from an LLM via LM Studio."""
    try:
        all_messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        all_messages.extend(messages_payload)

        completion = client.chat.completions.create(
            model=model_id,
            messages=all_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        response_content = completion.choices[0].message.content.strip()
        return response_content
    except Exception as e:
        if DEBUG_LOGS:
            print(f"Error communicating with LLM ({model_id}): {e}")
        try:
            if DEBUG_LOGS:
                print("Attempting to list available models from LM Studio...")
            models_data = client.models.list()
            if DEBUG_LOGS:
                print("Available models:")
                for m in models_data.data:
                    print(f"  ID: {m.id}, Object: {m.object}, Owned By: {m.owned_by}")
        except Exception as model_list_e:
            if DEBUG_LOGS:
                print(f"Could not retrieve model list: {model_list_e}")
        return None


def get_feedback_instruction(current_history):
    """Gets a structured instruction from the Feedback LLM."""
    if DEBUG_LOGS:
        print("\n----- Getting Feedback Instruction -----")
    if not current_history:
        return "ACTION: Greet warmly. TONE: Friendly and inviting. FOCUS: Ask how the user is feeling."

    feedback_context = []
    for msg in current_history[-(MAX_HISTORY_LEN * 2):]:
        feedback_context.append(f"{msg['role'].capitalize()}: {msg['content']}")
    history_str = "\n".join(feedback_context)

    user_focused_prompt = f"Recent conversation context:\n{history_str}\n\nUSER'S LAST MESSAGE: \"{current_history[-1]['content']}\"\n\nGenerate the single-line JSON directive for the ESB, ensuring the 'directive' object contains 'action_type', 'suggested_tone', and 'focus_hint'."

    raw_json_output = get_llm_response(
        FEEDBACK_MODEL_ID,
        FEEDBACK_LLM_SYSTEM_PROMPT,
        [{"role": "user", "content": user_focused_prompt}],
        temperature=0.05,
        max_tokens=300
    )

    instruction_string = "ACTION: Listen actively and show empathy. TONE: Gentle and understanding. FOCUS: User's current main concern."
    if raw_json_output:
        if DEBUG_LOGS:
            print(f"Feedback LLM ({FEEDBACK_MODEL_ID}) Raw Output:\n'{raw_json_output}'")
        try:
            processed_output = raw_json_output.strip()
            if processed_output.startswith("```json"):
                processed_output = processed_output.removeprefix("```json").strip()
            if processed_output.endswith("```"):
                processed_output = processed_output.removesuffix("```").strip()

            parsed_data = None
            try:
                parsed_data = json.loads(processed_output)
            except json.JSONDecodeError:
                if DEBUG_LOGS:
                    print("Initial JSONDecodeError for Feedback LLM output, attempting to fix quotes (heuristic)...")
                try:
                    processed_output_fixed_quotes = processed_output.replace("'", '"')
                    parsed_data = json.loads(processed_output_fixed_quotes)
                except json.JSONDecodeError as e2_fix_quotes:
                    if DEBUG_LOGS:
                        print(f"Still JSONDecodeError after quote fix attempt for Feedback LLM output: {e2_fix_quotes}")
                    raise e2_fix_quotes from None

            directive_obj = parsed_data.get("directive")
            if directive_obj and isinstance(directive_obj, dict):
                action = directive_obj.get("action_type")
                tone = directive_obj.get("suggested_tone")
                focus = directive_obj.get("focus_hint")

                if action and tone and focus:
                    instruction_string = f"ACTION: {action}. TONE: {tone}. FOCUS: {focus}."
                    if DEBUG_LOGS:
                        print(
                            f"Feedback LLM ({FEEDBACK_MODEL_ID}) Successfully Parsed Instruction:\n{instruction_string}")
                else:
                    missing_keys_details = []
                    if not action: missing_keys_details.append("directive.action_type")
                    if not tone: missing_keys_details.append("directive.suggested_tone")
                    if not focus: missing_keys_details.append("directive.focus_hint")
                    if DEBUG_LOGS:
                        print(
                            f"Feedback LLM ({FEEDBACK_MODEL_ID}) JSON 'directive' object missing required keys. Missing: {', '.join(missing_keys_details)}. Directive Object: {directive_obj}")
            else:
                if DEBUG_LOGS:
                    print(
                        f"Feedback LLM ({FEEDBACK_MODEL_ID}) JSON missing top-level 'directive' object or it's not a dictionary. Parsed: {parsed_data}")
        except json.JSONDecodeError as e_json:
            if DEBUG_LOGS:
                print(
                    f"Feedback LLM ({FEEDBACK_MODEL_ID}) failed to output valid JSON. Error: {e_json}. Raw Output: '{raw_json_output}'")
        except Exception as e_general:
            if DEBUG_LOGS:
                print(
                    f"Feedback LLM ({FEEDBACK_MODEL_ID}) Error processing output: {e_general}. Raw Output: '{raw_json_output}'")
    else:
        if DEBUG_LOGS:
            print(f"Feedback LLM ({FEEDBACK_MODEL_ID}) produced no output.")

    is_fallback = (
                instruction_string == "ACTION: Listen actively and show empathy. TONE: Gentle and understanding. FOCUS: User's current main concern.")
    if DEBUG_LOGS and is_fallback and raw_json_output:
        print(f"Using fallback instruction for ESB because structured feedback from LLM failed or was incomplete.")
    return instruction_string


def generate_esb_response(current_history_for_esb, dynamic_instruction, retrieved_requirements):
    """Generates a response from the ESB LLM, augmented with RAG context."""
    if DEBUG_LOGS:
        print("\n----- Generating ESB Response -----")

    rag_context_str = ""
    if retrieved_requirements:
        rag_context_str = "Consider these specific guidelines for your response (retrieved based on relevance to the current topic):\n"
        for i, req in enumerate(retrieved_requirements):
            rag_context_str += f"- {req}\n"
        rag_context_str += "---\n"

    current_esb_system_prompt = f"""{ESB_SYSTEM_PROMPT}

{rag_context_str}
You have also received the following dynamic guidance for this specific interaction:
---
{dynamic_instruction}
---
Interpret ALL this information (general role, specific guidelines from RAG, and dynamic guidance from feedback LLM) carefully. Use the specified ACTION as your primary goal, adopt the TONE described, and pay attention to the FOCUS point when formulating your empathetic and supportive message.
"""
    if DEBUG_LOGS:
        print(f"Full ESB System Prompt (for potential evaluation):\n{current_esb_system_prompt}")

    response = get_llm_response(
        ESB_MODEL_ID,
        current_esb_system_prompt,
        current_history_for_esb,
        temperature=0.75,
        max_tokens=400  # Slightly increased max_tokens as augmented prompt can be longer
    )
    return response


def validate_with_guardrail(candidate_response):
    """Validates the ESB's candidate response."""
    if DEBUG_LOGS:
        print("\n----- Validating with Guardrail -----")
    if not candidate_response:
        if DEBUG_LOGS:
            print(f"Guardrail ({GUARDRAIL_MODEL_ID}): No response to validate.")
        return False

    validation_result_str = get_llm_response(
        GUARDRAIL_MODEL_ID,
        GUARDRAIL_LLM_SYSTEM_PROMPT,
        [{"role": "user", "content": f"Candidate message: \"{candidate_response}\""}],
        temperature=0.1,
        max_tokens=20
    )

    if DEBUG_LOGS:
        print(f"Guardrail LLM ({GUARDRAIL_MODEL_ID}) Output: '{validation_result_str}'")

    if validation_result_str and "SAFE" in validation_result_str.upper():
        return True
    elif validation_result_str and "UNSAFE" in validation_result_str.upper():
        return False
    else:
        if DEBUG_LOGS:
            print(f"Guardrail LLM ({GUARDRAIL_MODEL_ID}) gave an ambiguous response. Defaulting to UNSAFE.")
        return False


def main_chat_loop():
    """Main loop for the orchestrated chat application with ChromaDB RAG."""
    global conversation_history, chroma_collection  # Ensure chroma_collection is global if modified
    print("Starting Emotional Support Buddy chat with ChromaDB RAG (v6).")
    if DEBUG_LOGS:
        print(f"ESB Model: {ESB_MODEL_ID}")
        print(f"Feedback Model: {FEEDBACK_MODEL_ID}")
        print(f"Guardrail Model: {GUARDRAIL_MODEL_ID}")
        print(f"Embedding Model: {EMBEDDING_MODEL_ID}")

    initialize_and_populate_chroma_collection()
    if chroma_collection is None:
        print("CRITICAL: Failed to initialize ChromaDB collection. RAG will not function. Please check errors.")
        return  # Exit if ChromaDB setup failed

    print("Type 'quit' to end the session.")

    try:
        lm_studio_models = client.models.list().data
        available_model_ids = [m.id for m in lm_studio_models]
        if DEBUG_LOGS:
            print("\nAvailable models in LM Studio:", available_model_ids)

        required_models = {
            "ESB": ESB_MODEL_ID,
            "Feedback": FEEDBACK_MODEL_ID,
            "Guardrail": GUARDRAIL_MODEL_ID,
            "Embedding": EMBEDDING_MODEL_ID
        }
        all_models_found = True
        for role, model_id_needed in required_models.items():
            if model_id_needed not in available_model_ids:
                if DEBUG_LOGS:
                    print(f"WARNING: Configured {role} model ID '{model_id_needed}' not found in LM Studio.")
                all_models_found = False

        if not all_models_found:
            print("CRITICAL: One or more required models are not loaded in LM Studio. Please check your setup.")
            return

    except Exception as e:
        if DEBUG_LOGS:
            print(f"Critical Error: Could not connect to LM Studio or list models: {e}")
        else:
            print("Error connecting to models. Please ensure LM Studio is running and models are loaded.")
        return

    initial_greeting = "Hello! I'm your Emotional Support Buddy. How are you feeling today?"
    print(f"\nESB: {initial_greeting}")
    conversation_history.append({"role": "assistant", "content": initial_greeting})

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'quit':
            print(f"ESB: Take care! It was good talking to you.")
            break
        if not user_input:
            continue

        conversation_history.append({"role": "user", "content": user_input})

        feedback_instruction = get_feedback_instruction(conversation_history)
        retrieved_requirements = retrieve_relevant_requirements_from_chroma(user_input, top_k=3,
                                                                            similarity_threshold=0.5)

        esb_response = None
        attempts = 0
        max_attempts = 2

        while attempts < max_attempts:
            attempts += 1
            if DEBUG_LOGS:
                print(f"\nESB ({ESB_MODEL_ID}) response generation attempt {attempts}...")

            candidate_esb_response = generate_esb_response(
                conversation_history,
                feedback_instruction,
                retrieved_requirements
            )

            if not candidate_esb_response:
                if DEBUG_LOGS:
                    print(f"ESB ({ESB_MODEL_ID}) failed to generate a response.")
                feedback_instruction = "ACTION: Offer a gentle acknowledgement. TONE: Kind and patient. FOCUS: General support if specific topic is hard."
                if attempts == max_attempts:
                    esb_response = "I'm having a little trouble formulating a response right now, but I'm still here to listen."
                continue

            is_safe = validate_with_guardrail(candidate_esb_response)
            if is_safe:
                esb_response = candidate_esb_response
                break
            else:
                if DEBUG_LOGS:
                    print(
                        f"ESB ({ESB_MODEL_ID}) response attempt {attempts} was deemed UNSAFE by {GUARDRAIL_MODEL_ID}.")
                feedback_instruction = "ACTION: Rephrase gently. TONE: Very careful and supportive. FOCUS: Ensuring safety and non-judgmental support."
                if attempts == max_attempts:
                    if DEBUG_LOGS:
                        print(f"ESB ({ESB_MODEL_ID}): Max attempts reached for safe response. Using a fallback.")
                    esb_response = "I want to make sure I'm being helpful and safe. Could you perhaps tell me more about what's on your mind in a different way?"

        print(f"\nESB: {esb_response}")
        conversation_history.append({"role": "assistant", "content": esb_response})

        if len(conversation_history) > MAX_HISTORY_LEN * 2:
            conversation_history = conversation_history[-(MAX_HISTORY_LEN * 2):]


if __name__ == "__main__":
    main_chat_loop()

