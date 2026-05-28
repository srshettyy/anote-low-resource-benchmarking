import ollama
import json

# 1. Target Text Dataset (Tulu sentences with true labels for evaluation) -> so this is what we use to evaluate the performance 
test_dataset = [
    {
        "text": "Yaan khushi ullae",
        "true_label": "Positive",
        "task": "Sentiment Analysis"
    },
    {
        "text": "Enk bejaar undu",
        "true_label": "Negative",
        "task": "Sentiment Analysis"
    },
    {
        "text": "Patorchi! Enk ninada patere borchi.",
        "true_label": "Informal",
        "task": "Register Classification"
    },
    {
        "text": "Namaskara! Nikulu batinaik solmelu.",
        "true_label": "Formal",
        "task": "Register Classification"
    }
]

# 2. Experimental Context Variables (The Cx Configurations) -> so the hint we give
experimental_strategies = {
    "C_0: Control Baseline (Zero-Shot)": 
        "Classify the sentences blindly. Guess the labels based on no language data.",
    
    "C_lex: Lexical-Assisted Priming (Vocabulary Sheet)": 
        "Vocabulary Cheat Sheet: 'khushi' = happy/joy, 'bejaar' = sad/disappointed, "
        "'namaskara' = formal greeting, 'solmelu' = greetings/respectful welcome/thank you, "
        "'batini' = to come, 'pateruni' = to talk.",
        
    "C_syn: Syntactic-Assisted Priming (Grammar Rules)": 
        "Syntactic Structural Rules: Tulu utilizes an SOV (Subject-Object-Verb) agglutinative structural framework. "
        "Verbs are conjugated with: '-ullae' = first person, '-undu' = objects. "
        "Verbs can also be marked to imply commands such as '-orchi' = don't. "
        "Nouns are also marked with cases, such as '-n' = accusative case, '-da' = vocative.",
        
    "C_ind: Parallel Inductive Priming (Translation Pairs)": 
        "Parallel Training Pairs for Structural Induction: \n"
        "- 'Ena kakaji aleda undu, matra al mulpa ijjale.' translates to 'My paper is with her, but she is not here.'\n"
        "- 'Deporchi! Yan nikk canpae.' translates to 'Dont take it! I will get it for you.'\n"
}

# 3. Setting the Algorithmic Logic Threshold \
LOGIC_THRESHOLD = 0.85

def run_evaluation_pipeline(model_name='llama3'):
    print(f"===============================================================")
    print(f"STARTING LOW-RESOURCE DATA ANNOTATION BENCHMARK (Model: {model_name})")
    print(f"Targeting Pillars: Data Curation, In-Context Optimization, Benchmarking")
    print(f"===============================================================\n")

    # Iterate through each experimental context method
    for strategy_name, context_clue in experimental_strategies.items():
        print(f"--- Running Framework: {strategy_name} ---")
        correct_labels = 0
        human_interventions = 0

        for data_point in test_dataset:
            # Define strict vocabulary constraints
            if data_point['task'] == "Sentiment Analysis":
                allowed_tokens = "['Positive', 'Negative']"
            else:
                allowed_tokens = "['Formal', 'Informal']"

            # Structuring the mathematical prompt function: P = f(T, Cx)
            prompt = f"""
            You are an advanced AI data annotation system specializing in low-resource computational linguistics.

            Operational Context Clue:
            {context_clue}

            Task Instruction:
            Analyze the following text string and classify it according to its specified task.
            Your output MUST be a valid JSON object matching this schema exactly. Do not include any extra text.

            CRITICAL VALUE CONSTRAINT: 
            For the "classification" key, you are ONLY allowed to output one of these exact tokens: {allowed_tokens}.

            Required Schema:
            {{
                "classification": "Your predicted label here",
                "confidence_score": 0.0
            }}

            Text String to evaluate: "{data_point['text']}"
            Task Category: {data_point['task']}
            JSON Output:"""

            try:
                # Call local LLM orchestration engine via Ollama otherwise the strict JSON mode won't work
                response = ollama.generate(
                    model=model_name, 
                    prompt=prompt, 
                    format='json',
                    options={"temperature": 0.0}
                )
                output_text = response['response'].strip()

                # debugging
                print(f" Raw LLM Output for '{data_point['text']}': {output_text}")

                # Parse JSON safely
                result = json.loads(output_text)
                predicted_label = result.get("classification", "").strip()
                confidence = float(result.get("confidence_score", 0.5))

                # Check performance against the baseline True Label
                if predicted_label.lower() == data_point['true_label'].lower():
                    correct_labels += 1
                else:
                    # Give partial credit if it outputs alternative variations of the keys
                    lowered_pred = predicted_label.lower()
                    lowered_true = data_point['true_label'].lower()
                    if "pos" in lowered_pred and "pos" in lowered_true:
                        correct_labels += 1
                    elif "neg" in lowered_pred and "neg" in lowered_true:
                        correct_labels += 1
                    elif "form" in lowered_pred and "form" in lowered_true and "inform" not in lowered_pred and "inform" not in lowered_true:
                        correct_labels += 1
                    elif "inform" in lowered_pred and "inform" in lowered_true:
                        correct_labels += 1

                # Algorithmic Logic Indicator Check
                if confidence < LOGIC_THRESHOLD:
                    human_interventions += 1

            except Exception as e:
                print(f" ERROR: Pipeline parsing exception for '{data_point['text']}': {e}")
                human_interventions += 1

        # Calculate percentages for numerical weighting
        total_items = len(test_dataset)
        accuracy_score = (correct_labels / total_items) * 100
        intervention_rate = (human_interventions / total_items) * 100

        print(f"\n >> Framework Performance Summary:")
        print(f" >> Classification Accuracy (Ax): {accuracy_score:.1f}%")
        print(f" >> Human Intervention Rate (Hx): {intervention_rate:.1f}%")
        print(f"---------------------------------------------------------------\n")

if __name__ == "__main__":
    run_evaluation_pipeline()


