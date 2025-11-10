# Qwen3-Embedding Cosine Similarity

Tính độ tương tự ngữ nghĩa giữa hai câu tiếng Việt bằng mô hình **[Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)**.

---

## Hai câu mẫu

1. *Gia Lai khẩn trương khắc phục thiệt hại trường, lớp sau bão Kalmaegi*  
2. *Gia Lai trong bão Kalmaegi: Bài học ứng phó thiên tai bằng bản lĩnh, tình người*

---

## Kết quả

- Cosine similarity: 0.6965
- Embedding dimension: 1024

---

## Cách chạy

```bash
pip install -U torch transformers sentence_transformers numpy
python main.py
