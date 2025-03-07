import os
import pickle
from typing import List

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

LLM_MODEL_NAME = "claude-3-haiku-20240307"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

print("Loading Embeddings model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
model = AutoModel.from_pretrained(EMBED_MODEL_NAME)

# search for root folders within notebooks dir and ask user which one to use
# should be just the folder name and not full path
ROOT_DIRS = [d for d in os.listdir("notebooks") if os.path.isdir(os.path.join("notebooks", d))]

for i, root_dir in enumerate(ROOT_DIRS):
    print(f"{i + 1}. {root_dir}")

root_dir_idx = int(input("Which notebook do you want to query? \n ")) - 1
DOCS_FOLDER = "notebooks"
DOCS_DIR = os.path.join(DOCS_FOLDER, ROOT_DIRS[root_dir_idx])
EMBEDDINGS_FOLDER = "embeddings"
INDEX_PATH = os.path.join(EMBEDDINGS_FOLDER, f"{ROOT_DIRS[root_dir_idx]}_faiss_index.faiss")
MAPPING_PATH = os.path.join(EMBEDDINGS_FOLDER, f"{ROOT_DIRS[root_dir_idx]}_mapping.pkl")


def retrieve_documents(query: str, k: int = 10) -> List[str]:
    query_embedding = get_sentence_embedding(query).unsqueeze(0).numpy().astype("float32")
    distances, indices = index.search(query_embedding, k)
    retrieved = [mapping[i] for i in indices[0] if i < len(mapping)]
    return retrieved


def build_faiss_index(texts: List[str]) -> faiss.IndexFlatL2:
    embeddings = []
    for doc in texts:
        emb = get_sentence_embedding(doc)
        embeddings.append(emb.numpy())
    embeddings = np.array(embeddings).astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def get_sentence_embedding(text: str, max_length: int = 512) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    token_embeddings = outputs.last_hidden_state  # shape: (1, seq_len, hidden_dim)
    mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * mask, dim=1)
    sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask
    normalized_embeddings = F.normalize(mean_embeddings, p=2, dim=1)
    return normalized_embeddings.squeeze(0)


def load_markdown_files(root_dir: str, ext: str = ".md") -> List[str]:
    texts = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(ext):
                file_path = os.path.join(dirpath, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    texts.append(f.read())
    return texts


if os.path.exists(INDEX_PATH) and os.path.exists(MAPPING_PATH):
    print("Loading existing FAISS index and mapping...")
    index = faiss.read_index(INDEX_PATH)
    with open(MAPPING_PATH, "rb") as f:
        mapping = pickle.load(f)
else:
    print(
        "Creating new FAISS index... This may take sometime. Future runs will use the saved index and will not be this slow")
    documents = load_markdown_files(DOCS_DIR)
    mapping = documents  # Each document's text
    index = build_faiss_index(documents)
    faiss.write_index(index, INDEX_PATH)
    with open(MAPPING_PATH, "wb") as f:
        pickle.dump(mapping, f)
