(
echo # Qwen3-Embedding Cosine Similarity
echo.
echo Tính độ tương tự ngữ nghĩa giữa 2 câu tiếng Việt bằng mô hình **Qwen/Qwen3-Embedding-0.6B**.
echo.
echo ## 2 câu mẫu
echo 1. Gia Lai khẩn trương khắc phục thiệt hại trường, lớp sau bão Kalmaegi
echo 2. Gia Lai trong bão Kalmaegi: Bài học ứng phó thiên tai bằng bản lĩnh, tình người
echo.
echo ## Cách chạy
echo pip install -U torch transformers sentence_transformers numpy
echo python main.py
echo.
echo ## Kết quả
echo - Embedding dimension: 1024
echo - Cosine similarity: 0.6965
) > README.md
