import numpy as np
import torch

def main():
    from sentence_transformers import SentenceTransformer

    model_id = "Qwen/Qwen3-Embedding-0.6B"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(model_id, device=device, trust_remote_code=True)

    s1 = "Gia Lai khẩn trương khắc phục thiệt hại trường, lớp sau bão Kalmaegi"
    s2 = "Gia Lai trong bão Kalmaegi: Bài học ứng phó thiên tai bằng bản lĩnh, tình người"

    embs = model.encode(
        [s1, s2],
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=2,
        show_progress_bar=False,
    )
    e1, e2 = embs[0], embs[1]

    cosine = float(np.dot(e1, e2))

    print("Qwen3-Embedding-0.6B (SentenceTransformers)")
    print(f"Kích thước vector embedding: {e1.shape[0]}")
    print(f"Cosine similarity: {cosine:.4f}")
    print("Vector câu 1 (10 phần tử đầu):", np.round(e1[:10], 6))
    print("Vector câu 2 (10 phần tử đầu):", np.round(e2[:10], 6))
    
    np.save("emb_s1.npy", e1)
    np.save("emb_s2.npy", e2)
    print("Đã lưu emb_s1.npy, emb_s2.npy")

if __name__ == "__main__":
    main()
