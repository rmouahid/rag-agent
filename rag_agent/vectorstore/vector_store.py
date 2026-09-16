import numpy as np
import json
import os

class VectorStore:
    def __init__(self):
        """
        Stockage en mémoire :
        - self.vectors  : matrice numpy (N x 384)
        - self.chunks   : liste des chunks associés
        """
        self.vectors = []
        self.chunks  = []

    def add(self, chunks: list[dict], embeddings: np.ndarray):
        """
        Ajoute des chunks et leurs embeddings au store
        """
        for chunk, embedding in zip(chunks, embeddings):
            self.chunks.append(chunk)
            self.vectors.append(embedding)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        Recherche les top_k chunks les plus proches via similarité cosine
        """
        if not self.vectors:
            return []

        matrix = np.array(self.vectors)

        # Similarité cosine = produit scalaire des vecteurs normalisés
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-10)
        matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
        scores = matrix_norm @ query_norm

        # Récupère les top_k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "text"     : self.chunks[idx]["text"],
                "metadata" : self.chunks[idx]["metadata"],
                "score"    : float(scores[idx])
            })

        return results

    def save(self, path: str):
        """
        Sauvegarde le store sur disque
        """
        os.makedirs(path, exist_ok=True)
        np.save(f"{path}/vectors.npy", np.array(self.vectors))
        with open(f"{path}/chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        print(f" VectorStore sauvegardé dans {path}")

    def load(self, path: str):
        """
        Charge le store depuis le disque
        """
        self.vectors = list(np.load(f"{path}/vectors.npy"))
        with open(f"{path}/chunks.json", "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        print(f" VectorStore chargé depuis {path} ({len(self.chunks)} chunks)")
