import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from utils.chatbot import load_vectorstore

QUESTION = "Quels sont les concerts de musique classique à Paris en juillet ?"
K = 4
NEIGHBORHOOD_SIZE = 100
OUTPUT_PATH = "utils/proximity.png"


def main():
    vs = load_vectorstore()
    total = vs.index.ntotal
    all_vectors = vs.index.reconstruct_n(0, total)

    question_vector = np.array(vs.embeddings.embed_query(QUESTION), dtype=np.float32)

    _, neighborhood_indices = vs.index.search(question_vector.reshape(1, -1), min(NEIGHBORHOOD_SIZE, total))
    background_vectors = all_vectors[neighborhood_indices[0]]

    _, indices = vs.index.search(question_vector.reshape(1, -1), K)
    retrieved_vectors = all_vectors[indices[0]]

    stacked = np.vstack([background_vectors, retrieved_vectors, question_vector])
    coords_2d = PCA(n_components=2).fit_transform(stacked)

    n_background = len(background_vectors)
    background_2d = coords_2d[:n_background]
    retrieved_2d = coords_2d[n_background:-1]
    question_2d = coords_2d[-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        background_2d[:, 0], background_2d[:, 1],
        c="black", s=20, alpha=0.5, label="Autres chunks de l'index",
    )

    for i, point in enumerate(retrieved_2d, 1):
        ax.plot(
            [question_2d[0], point[0]], [question_2d[1], point[1]],
            c="tab:orange", linestyle="--", linewidth=1, alpha=0.7,
        )
        ax.scatter(*point, c="tab:orange", s=120, edgecolors="black", zorder=3)
        ax.annotate(str(i), point, textcoords="offset points", xytext=(6, 6), fontsize=11, fontweight="bold")

    ax.scatter(*question_2d, c="tab:red", marker="*", s=400, edgecolors="black", zorder=4, label="Question")

    ax.set_title("Proximité dans l'espace des embeddings : question vs chunks récupérés")
    ax.set_xlabel("Composante principale 1")
    ax.set_ylabel("Composante principale 2")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"Figure sauvegardée dans {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
