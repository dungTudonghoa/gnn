# Thực hành GNN: GCN phân loại bài báo Cora

Triển khai phục vụ học tập theo [bài giới thiệu GNN trên Viblo](https://viblo.asia/p/gioi-thieu-ve-graph-neural-networks-gnns-yZjJYG7MVOE). Code được viết lại thành chương trình chạy độc lập và notebook có giải thích tiếng Việt.

## Chạy nhanh

Python 3.10+; CPU là đủ. Lần đầu cần Internet để cài thư viện và tải Cora.

```bash
git clone https://github.com/dungTudonghoa/gnn.git
cd gnn
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python cora.py --inspect
python cora.py
```

Hoặc mở [notebook trên Colab](https://colab.research.google.com/github/dungTudonghoa/gnn/blob/main/notebooks/cora_walkthrough.ipynb) rồi Run all. Notebook gồm đồ thị NetworkX, xem tensor, huấn luyện, hình trước/sau và dự đoán một node.

## Dữ liệu vào thực sự là gì?

Cora là mạng trích dẫn bài báo. `Planetoid` là lớp tải dữ liệu hỗ trợ Cora, CiteSeer hoặc PubMed; chọn `name='Cora'` chỉ tải Cora, không trộn ba bộ.

| Tensor | Kích thước Cora | Ý nghĩa |
|---|---|---|
| `x` | 2708 × 1433 | Mỗi hàng là đặc trưng từ vựng của một bài báo |
| `edge_index` | 2 × 10556 | Mỗi cột là cặp chỉ số node nguồn, đích |
| `y` | 2708 | Mã lớp của từng bài, từ 0 đến 6 |
| `train_mask` | 2708 | Chọn 140 node để tính loss |
| `val_mask` | 2708 | Chọn 500 node để theo dõi validation |
| `test_mask` | 2708 | Chọn 1000 node để đánh giá cuối |

Các mask không phủ hết node. Toàn bộ đặc trưng và cạnh được dùng khi truyền thông tin, nhưng chỉ nhãn train tham gia loss: đây là thiết lập transductive semi-supervised.

Về mặt xây dựng dữ liệu: danh sách bài báo → gán chỉ số node; đặc trưng hiện diện từ trong từ điển cố định → các hàng `x`; danh sách trích dẫn → các cột `edge_index`; chủ đề bài báo → `y`. Đây không phải embedding ngôn ngữ hay văn bản nguyên gốc. `NormalizeFeatures` chuẩn hóa mỗi hàng đặc trưng theo tổng hàng.

PyG không tải PDF rồi tự trích từ. Nó tải các file Planetoid đã tiền xử lý (`ind.cora.x`, `tx`, `allx`, `y`, `ty`, `ally`, `graph`, `test.index`) và ghép thành `Data`. File `graph` chứa danh sách hàng xóm; `test.index` giúp sắp lại chỉ số test. Các bản Cora dạng `cora.content`/`cora.cites` là cách biểu diễn khác, không phải file đầu vào trực tiếp của loader này. 10556 là số cột cạnh trong biểu diễn PyG có hai chiều, không nên hiểu là 10556 cặp bài báo duy nhất.

Ví dụ tự dựng: A có đặc trưng `[1,0,1]`, B có `[0,1,0]`, C có `[1,1,0]`; A nối B, B nối C. Khi xem như vô hướng:

```python
x = [[1,0,1], [0,1,0], [1,1,0]]
edge_index = [[0,1,1,2], [1,0,2,1]]
```

`edge_index[:, 0] = [0,1]` là cạnh A → B. Hàng đầu không phải danh sách đặc trưng. Chạy `--inspect` để đối chiếu ví dụ này với dữ liệu thật.

## Mô hình và kết quả

Luồng kích thước: `x [2708,1433]` + cạnh → GCN 16 chiều → ReLU → dropout 0.5 → GCN 7 chiều → logits `[2708,7]`. `argmax(dim=1)` trả về một lớp cho mỗi node. Đầu ra GNN ở bài này là ma trận điểm phân loại node, không cần tạo đồ thị mới. `GCNConv` tự xử lý self-loop và chuẩn hóa theo bậc.

Giữ cấu hình bài: seed 1234567, Adam, lr 0.01, weight decay 0.0005, 100 epoch, public split. Dùng CrossEntropyLoss trực tiếp trên logits. Khác bài: không theo dõi test mỗi epoch; đánh giá model cuối một lần, có `no_grad`, và truyền mask đúng khi tính accuracy. Không chọn checkpoint bằng test.

Chương trình ghi `outputs/history.csv`, `metrics.json`, `gcn.pt`, `before.png`, `after.png`. Hình là t-SNE của logits và tô màu nhãn thật để quan sát; không phải đồ thị cạnh, cũng không phải bằng chứng accuracy. Không so sánh tuyệt đối tọa độ giữa hai lần t-SNE.

```bash
python cora.py --epochs 2 --no-plots  # kiểm tra luồng nhanh
python cora.py --device cuda         # nếu đã có PyTorch CUDA phù hợp
```

Accuracy phụ thuộc phiên bản và thiết bị; không cam kết tái tạo đúng con số bài viết. Dependencies giới hạn phiên bản lớn, chưa phải lockfile. Xem `VALIDATION.md` để biết mức kiểm chứng thực tế.

## Tự thực hành

1. In một cột cạnh và hai hàng `x` tương ứng.
2. Kiểm tra tổng hàng `x[0]` sau chuẩn hóa.
3. Giải thích vì sao 140 nhãn vẫn train trên đồ thị 2708 node.
4. Đổi hidden dimension rồi so sánh validation, giữ test cho đánh giá cuối.

Nguồn kỹ thuật: [PyG Planetoid](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html), [GCNConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html), [mã loader Planetoid](https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/io/planetoid.py).
