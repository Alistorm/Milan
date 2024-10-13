import time

import pandas as pd
import torch
from langchain_ollama.chat_models import ChatOllama
from mistralai import Mistral
from tqdm import tqdm  # Import tqdm for progress tracking
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import MISTRAL_API_KEY


def get_system_prompt():
    return """Vous êtes un assistant médical de niveau expert spécialisé dans les examens médicaux. Vous devez répondre à des questions à choix multiples en sélectionnant uniquement les bonnes réponses parmi cinq options possibles : A, B, C, D, E. 

**Instructions strictes :**

1. **Expertise Médicale** : Utilisez vos connaissances avancées en médecine pour sélectionner **toutes les réponses correctes**. Il y a toujours au moins une réponse correcte, et parfois plusieurs ou même toutes peuvent être correctes.
2. **Réponse Sans Explication** : **Ne fournissez jamais d'explication.** Répondez uniquement avec les lettres correspondant aux bonnes réponses, sans aucun texte supplémentaire.
3. **Format Obligatoire de Réponse** : Répondez en fournissant uniquement les lettres des réponses correctes, séparées par des virgules (exemple : A,C ou A,B,D,E), sans espaces ni texte explicatif.

**Règles Importantes** :
- Vous devez toujours fournir une réponse sous la forme de lettres séparées par des virgules (ex : A,C,D).
- Ne fournissez **aucune explication** ni commentaire supplémentaire. Uniquement les lettres.
- Assurez-vous d'inclure toutes les réponses correctes.

Exemple :
- Si les réponses correctes sont A et D, répondez uniquement : A,D.
- Si toutes les réponses sont correctes, répondez : A,B,C,D,E.

Vous devez respecter ces règles à tout moment et vous appuyer sur vos compétences médicales pour sélectionner la réponse correcte.
"""


def build_question_prompt(
    question,
    a,
    b,
    c,
    d,
    e,
):
    return f"""Question: {question.strip()}
Réponse A: {a.strip()}
Réponse B: {b.strip()}
Réponse C: {c.strip()}
Réponse D: {d.strip()}
Réponse E: {e.strip()}
"""


def get_examples():
    return [
        {
            "question": "Devant un exanthème roséoliforme fébrile de l'enfant, les principales étiologies sont:",
            "answers": [
                "Un exanthème subit",
                "Un mégalérythème épidémique",
                "Une rubéole",
                "Une mononucléose infectieuse",
                "Un syndrome de Kawasaki",
            ],
            "expected_output": "A,C,D",
        },
        {
            "question": "A propos de l’insuffisance cardiaque, quelle(s) est (sont) la (les) proposition(s) vraie(s) ?",
            "answers": [
                "L’auscultation cardiaque peut mettre en évidence un éclat du B2 au foyer aortique en cas d’hypertension",
                "L’auscultation cardiaque peut mettre en évidence un souffle d’insuffisance mitrale",
                "La turgescence jugulaire constitue un signe d’insuffisance cardiaque gauche",
                "Les œdèmes périphériques sont mous, bleus et douloureux",
                "Les râles crépitants sont souvent bilatéraux",
            ],
            "expected_output": "B,C,E",
        },
        {
            "question": "Quelles sont les causes fréquentes de douleurs thoraciques aigües ?",
            "answers": [
                "Infarctus du myocarde",
                "Pneumothorax",
                "Embolie pulmonaire",
                "Dissection aortique",
                "Reflux gastro-œsophagien",
            ],
            "expected_output": "A,B,C,D,E",
        },
    ]


def complete_qa(questions_file):
    client = Mistral(api_key=MISTRAL_API_KEY)
    answers = []
    df = pd.read_csv(questions_file)
    system_prompt = get_system_prompt()
    examples = get_examples()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Questions"):
        messages = [{"role": "system", "content": system_prompt}]

        # Include examples in the messages
        for ex in examples:
            messages.append(
                {
                    "role": "user",
                    "content": build_question_prompt(
                        ex["question"],
                        ex["answers"][0],
                        ex["answers"][1],
                        ex["answers"][2],
                        ex["answers"][3],
                        ex["answers"][4],
                    ),
                }
            )
            messages.append({"role": "assistant", "content": ex["expected_output"]})

        # Add the current question to the messages
        messages.append(
            {
                "role": "user",
                "content": build_question_prompt(
                    row["question"],
                    row["answer_A"],
                    row["answer_B"],
                    row["answer_C"],
                    row["answer_D"],
                    row["answer_E"],
                ),
            }
        )

        # Use the chat client to get a response
        chat_response = client.chat.complete(
            messages=messages,
            model="mistral-large-latest",
            temperature=0,  # Experiment with 0.7 for more creative responses
            max_tokens=5,  # Increase to allow longer responses
            min_tokens=1,
            top_p=1  # Keep it at 1 for maximum output
        )

        # Get and process the response
        response = chat_response.choices[0].message.content.strip()
        answers.append(response)

        # Sleep to avoid rate limits if necessary (adjust as per API limits)
        time.sleep(1.5)  # Reduced sleep time for faster processing

    # Create DataFrame with answers
    output_df = pd.DataFrame(answers, columns=["Answer"])
    output_df.reset_index(
        inplace=True
    )  # Resets the index and creates a new column named 'index'
    output_df.rename(
        columns={"index": "id"}, inplace=True
    )  # Rename the new column to 'id'
    output_df.to_parquet("output.parquet", index=False)  # Save without original index


def complete_qa_local(questions_file):
    chat = ChatOllama(
        model="mistral-nemo:latest",
        temperature=0,  # Experiment with 0.7 for more creative responses
        max_tokens=5,  # Increase to allow longer responses
        min_tokens=1,
        top_p=0.95,
        top_k=5
    )  # Keep it at 1 for maximum output)  # Adjusted temperature for variability
    answers = []
    df = pd.read_csv(questions_file)
    system_prompt = get_system_prompt()
    examples = get_examples()

    # Use tqdm to show progress
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Questions"):
        messages = [{"role": "system", "content": system_prompt}]

        # Include example prompts and expected outputs
        for ex in examples:
            messages.append(
                {
                    "role": "user",
                    "content": build_question_prompt(
                        ex["question"],
                        ex["answers"][0],
                        ex["answers"][1],
                        ex["answers"][2],
                        ex["answers"][3],
                        ex["answers"][4],
                    ),
                }
            )
            messages.append({"role": "assistant", "content": ex["expected_output"]})

        # Add the current question
        messages.append(
            {
                "role": "user",
                "content": build_question_prompt(
                    row["question"],
                    row["answer_A"],
                    row["answer_B"],
                    row["answer_C"],
                    row["answer_D"],
                    row["answer_E"],
                ),
            }
        )

        # Get the response from the chat model
        chat_response = chat.invoke(messages)
        response = chat_response.content.strip()
        answers.append(response)

    # Create a DataFrame and save to Parquet format
    output_df = pd.DataFrame(answers, columns=["Answer"])
    output_df.reset_index(
        inplace=True
    )  # Resets the index and creates a new column named 'index'
    output_df.rename(
        columns={"index": "id"}, inplace=True
    )  # Rename the new column to 'id'
    output_df.to_parquet(
        "output.parquet", index=False
    )  # Save without the original index


def load_model(model_name):
    """Load the specified model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model


def get_answer_from_logits(logits):
    """Extract the correct answers from the model's logits."""
    # Assuming a multi-class classification with answers A-E mapped to indices 0-4
    probabilities = torch.softmax(logits, dim=-1)
    predictions = torch.argmax(probabilities, dim=-1)
    
    return predictions


def complete_qa_biobert(questions_file):
    model_name = "dmis-lab/biobert-v1.1"
    tokenizer, model = load_model(model_name)
    
    df = pd.read_csv(questions_file)
    answers = []

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        # Prepare inputs for the model
        inputs = tokenizer(
            [row["question"]] + [row[f"answer_{letter}"] for letter in ["A", "B", "C", "D", "E"]],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Get predictions
        predictions = get_answer_from_logits(logits)
        answers.append(predictions.tolist())

    # Convert answers to appropriate format
    output_df = pd.DataFrame(answers, columns=["Answer"])
    output_df.to_parquet("output_biobert.parquet")


def complete_qa_clinicalbert(questions_file):
    model_name = "emilyalsup/clinicalbert"
    tokenizer, model = load_model(model_name)

    df = pd.read_csv(questions_file)
    answers = []

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        inputs = tokenizer(
            [row["question"]] + [row[f"answer_{letter}"] for letter in ["A", "B", "C", "D", "E"]],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        predictions = get_answer_from_logits(logits)
        answers.append(predictions.tolist())

    output_df = pd.DataFrame(answers, columns=["Answer"])
    output_df.to_parquet("output_clinicalbert.parquet")


def complete_qa_biomistral(questions_file):
    model_name = "mistral-nemo/biomistral"
    tokenizer, model = load_model(model_name)

    df = pd.read_csv(questions_file)
    answers = []

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        inputs = tokenizer(
            [row["question"]] + [row[f"answer_{letter}"] for letter in ["A", "B", "C", "D", "E"]],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        predictions = get_answer_from_logits(logits)
        answers.append(predictions.tolist())

    output_df = pd.DataFrame(answers, columns=["Answer"])
    output_df.to_parquet("output_biomistral.parquet")

if __name__ == "__main__":
    filepath = "/Users/master/Projects/milan/data/questions.csv"
    complete_qa(filepath)
    
