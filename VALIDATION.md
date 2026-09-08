# Kiểm chứng

- PASS: phân tích cú pháp `cora.py` và các cell Python trong notebook.
- PASS: `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest test_smoke -v`.
- Test dùng đồ thị giả lập 40 node, kiểm tra forward, 2 epoch backward/optimizer, loss hữu hạn, xuất CSV/JSON, hai hình t-SNE và checkpoint. Nạp lại checkpoint cho logits khớp model đã lưu.
- Môi trường test: Python 3.12, torch 2.14.0+cu130 (chạy CPU), torch-geometric 2.8.0.post1.
- Chưa hoàn tất chạy Cora 100 epoch: loader lỗi DNS `Cannot connect to host github.com:443`. Không có accuracy Cora được xác nhận trong repo. Số đo trong smoke test không đại diện cho Cora.
- Notebook chưa được chạy toàn bộ trên Colab; mã nguồn dùng chung đã được kiểm tra offline như trên.

Để kiểm chứng trên máy có mạng: `python cora.py --inspect`, sau đó `python cora.py`. Kết quả thực nằm trong `outputs/metrics.json`.
