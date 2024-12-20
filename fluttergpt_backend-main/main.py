import logging
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Configuration du journal (logging)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Création de l'instance de l'application FastAPI
app = FastAPI()

# Activer CORS (Cross-Origin Resource Sharing) pour permettre les requêtes de tous les domaines
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Autorise tous les origines, peut être restreint en production
    allow_credentials=True,
    allow_methods=["*"],  # Autorise toutes les méthodes HTTP
    allow_headers=["*"],  # Autorise tous les en-têtes
)

# Chargement du modèle et du tokenizer lors du démarrage de l'application
logger.info("Chargement du modèle et du tokenizer Qwen/QwQ-32B-Preview...")
try:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    # Utiliser le GPU si disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Définir le token de padding si ce n'est pas déjà fait
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Modèle chargé avec succès sur {device}.")
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle : {str(e)}")
    raise RuntimeError("Échec du chargement du modèle. Assurez-vous que les fichiers du modèle sont accessibles.")

# Définition de la classe Payload pour accepter le texte d'entrée
class Payload(BaseModel):
    text: str

# Fonction de prétraitement de l'entrée
def preprocess_input(input_text: str) -> str:
    """
    Prétraiter le texte d'entrée avant de le passer au modèle.
    """
    logger.info("Prétraitement du texte d'entrée...")
    # Exemple de prétraitement : supprimer les espaces superflus, normaliser la casse, retirer les caractères non désirés
    processed_text = input_text.strip()
    logger.info(f"Texte traité : {processed_text}")
    return processed_text

# Fonction de post-traitement de la sortie du modèle
def postprocess_output(raw_output: str, question: str) -> str:
    """
    Post-traiter la sortie brute générée par le modèle.
    Supprime la question de la réponse si elle est incluse dans la réponse.
    """
    logger.info("Post-traitement de la sortie du modèle...")
    # Exemple de post-traitement : supprimer la question du texte si elle existe dans la réponse
    processed_output = raw_output.strip()

    # Vérifier si la réponse commence par la question et la supprimer
    if processed_output.lower().startswith(question.lower()):
        processed_output = processed_output[len(question):].strip()

    logger.info(f"Sortie post-traitée : {processed_output}")
    return processed_output

# Route POST pour générer une réponse à partir du texte d'entrée
@app.post('/get-response')
async def get_response(payload: Payload):
    """
    Générer une réponse du modèle en fonction du texte d'entrée.
    """
    try:
        # Prétraiter la question reçue dans le payload
        question = preprocess_input(payload.text)
        logger.info(f"Question reçue : {question}")

        # Tokenisation du texte d'entrée
        inputs = tokenizer(
            question,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        # Générer la réponse
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=150,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.pad_token_id,
                temperature=0.7,  # Ajuster la température pour plus de créativité
                top_k=50,        # Utiliser un échantillonnage top-k pour une meilleure diversité
            )

        # Décoder les tokens générés
        raw_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Réponse brute générée : {raw_response}")

        # Post-traiter la réponse en retirant la question si elle existe
        response = postprocess_output(raw_response, question)

        # Retourner la réponse sous forme de JSON
        return JSONResponse({"response": response}, status_code=200)

    except Exception as e:
        # En cas d'erreur, retourner une réponse d'erreur
        logger.error(f"Erreur lors de la génération de la réponse : {str(e)}")
        return JSONResponse({"error": "Une erreur est survenue lors du traitement de votre demande."}, status_code=500)

# Lancer l'application FastAPI (pour le développement uniquement)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
