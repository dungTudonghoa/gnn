# Kiểm chứng bản triển khai

- 7 unit/integration tests đạt (`python -m unittest discover -s supply_chain/tests -v`).
- Bài tính tay supply/inventory/shortage đạt: objective = 50; trường hợp tồn dư đạt objective = 13.
- Bài regret có hai kịch bản chọn flows khác nhau đạt nghiệm chung q=270/34, regret=7×270/34; kiểm tra không vô tình dùng scenario-specific recourse.
- MILP và LP có assignment cố định khớp trong sai số 1e-6.
- Vét cạn 64 và 256 assignment trên reconstructed-train-03/04 cho cả deterministic và regret: sai khác lớn nhất với MILP dưới 3.3e-11.
- Giải lại assignment tốt nhất của cả 760 runs: objective khớp hoàn toàn; vi phạm ràng buộc vật lý lớn nhất 2.28e-13. Xem `results/reconstruction/verification.json`.
- Kiểm tra singleton, eligibility, cache, exhaustion, tight-budget và entropy; hoán vị thứ tự factory làm hoán vị tương ứng output GNN.
- Các cell notebook được chạy nối tiếp trong IPython cùng process và lưu output. Chưa chạy trên dịch vụ Colab; kernel Jupyter riêng trong môi trường kiểm tra không mở được socket. Cell cài thư viện riêng của Colab chưa được thực thi ở đây.
- 760 runs, 42.400 distinct candidate LP solves không bao gồm solver calls cho nhãn, reference và kiểm chứng.

Các kiểm tra trên xác minh mã triển khai và dữ liệu mới. Không chứng minh khớp dữ liệu, model hoặc số liệu gốc của tác giả.
