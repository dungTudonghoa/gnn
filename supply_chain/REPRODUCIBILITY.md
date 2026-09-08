# Mức độ tái hiện arXiv:2608.10245v1

**Đây là mã triển khai độc lập theo bài báo, chưa tái tạo các bảng số liệu gốc.** Dữ liệu trong `results/reconstruction/instances` là dữ liệu mới. Không được gọi các fixture này là Instances 1–15 của tác giả.

Bài: *A Graph Neural Network–Guided Genetic Algorithm for Physical Internet Supply Chain Optimization under Cost Uncertainty*, Faezeh Ardali và Gerald M. Knapp, v1, 10-08-2026. Nguồn: PDF người dùng cung cấp và [arXiv](https://arxiv.org/abs/2608.10245).

## Đối chiếu phương pháp

| Nội dung công bố | Triển khai | Vị trí trong bài |
|---|---|---|
| Chi phí gán, cấp hàng, chuyển hub, giao retailer, thiếu hàng | `optimization.py`, `FlowModel` | Trang 2, (1) |
| Công suất factory, một nguồn mỗi hub, eligibility, giới hạn inbound | MILP có X nhị phân; LP có bounds phụ thuộc chromosome | Trang 2, (2)–(5) |
| Cân bằng hub dạng bất đẳng thức, demand bằng giao + thiếu | Giữ đúng dấu, cho phép dư tồn cuối kỳ | Trang 2, (6)–(9) |
| Min–max regret với X và flows dùng chung 3 kịch bản | Ba bất đẳng thức epigraph; references từ MILP từng kịch bản | Trang 2, (10) |
| Hệ số lower/nominal/upper | `SCENARIOS`, đủ 5 nhóm chi phí | Trang 2 |
| GNN dị thể 3 loại node, 5 loại quan hệ, 2 vòng, hidden 32 | PyTorch, MLP riêng từng quan hệ, mean aggregation, residual, LayerNorm, ReLU, không dropout | Trang 2, II-B |
| Softmax theo factory ứng viên của từng hub | Tensor `[O,H]`, softmax `dim=0`, mask eligibility | Trang 2, (11) |
| AdamW .01, decay .0001, clip 5, seed 3101, tối đa 100 epoch, patience 25 | `gnn.train` | Trang 2 |
| Population min(Pnom,B,assignment space) | `search.search` | Trang 3, (12) |
| 80% guided gồm greedy, 20% uniform; tournament 3; crossover .8; elite 5% | GNN–GA và các ablation | Trang 3, Table I |
| Entropy chuẩn hóa và mutation (13), singleton = 0 | `entropy_mutation` | Trang 3 |
| Cache chromosome, ngân sách distinct LP, HiGHS một thread | Cache độc lập mỗi run; lưu số evaluations và generations | Trang 3 |
| Thử nghiệm nhiều seed và Wilcoxon/Holm, Friedman/Holm | `experiment.statistics` | Trang 4 |

## Thông tin PDF chưa đủ để tái tạo số liệu

Phần III, trang 4 nêu generator, implementation, trained parameters và raw run-level results được cung cấp khi yêu cầu tác giả. Không thấy liên kết tải chúng trong PDF. Những dữ liệu cần bổ sung:

1. Các mảng instance gốc và đầy đủ kích thước từng instance; PDF chỉ nêu một số kích thước, không liệt kê toàn bộ 15 bộ.
2. Generator seed 42: thứ tự lấy mẫu, RNG, số nguyên/thực, phân phối tỷ trọng công suất, tồn kho khởi tạo và quy tắc làm tròn.
3. Danh sách chính xác các feature, thứ tự cột, chi tiết MLP/update và state dict đã chọn.
4. Batching, tiêu chí early stopping và tie breaking nhãn tối ưu.
5. Lịch nhiệt SA, cách xử lý chromosome trùng, chính xác quy tắc rank warm start và làm tròn tỷ lệ quần thể.
6. Incumbents/scenario baselines của Instances 14–15 và logs/chứng nhận tối ưu của các reference.

**Không suy ra các mục này từ việc có cùng seed.** Không điều chỉnh dữ liệu hoặc seed để ép số liệu khớp bài.

## Các lựa chọn bổ sung được công khai

- Generator: PCG64 `default_rng`; uniform liên tục trong khoảng công bố; tỷ trọng capacity/inbound lấy uniform(.5,1.5), chuẩn hóa tổng bằng 1.2/1.1 tổng demand; tồn kho ban đầu bằng 0.
- Chín fixture train và ba fixture validation có kích thước tự chọn trong `demo_manifest`. Ba test có kích thước 3/10/25, 4/12/30, 5/15/40 và seed 202601–202603 giống thông tin T1–T3, **nhưng không phải cùng dữ liệu T1–T3** vì thiếu generator gốc.
- Budget test 75/25/10 là cấu hình minh họa tight-budget mượn từ Table II, không phải ngân sách T1–T3 được xác nhận. Sensitivity B=400 đặt trên reconstructed-test-01, không gọi là Instance 13.
- Feature factory: capacity, supply mean/min, fixed mean/min. Hub: inbound, inventory, supply mean/min, fixed mean/min, delivery mean, transfer-out mean. Retailer: demand, shortage, delivery mean/min. Factory–hub edge: supply/fixed/eligibility; hub–retailer: delivery; hub–hub: transfer. Quan hệ reverse dùng lại edge attributes. Feature thống kê local có cả các cặp không eligible khi dữ liệu sparse; output vẫn mask các cặp đó.
- Chuẩn hóa mean/std chỉ fit trên train, riêng theo loại node/quan hệ; eligibility không chuẩn hóa. MLP thông điệp nhận `[source hidden, target hidden, edge attributes]`; cộng mean của từng loại quan hệ ở node nhận, linear update rồi residual–LayerNorm–ReLU. Hai vòng có trọng số riêng. Scorer MLP một hidden layer.
- Một optimizer step trên toàn bộ train mỗi epoch, CE trung bình theo hub; chọn validation CE nhỏ nhất, dừng sau 25 epoch không cải thiện. Nhãn từ deterministic MILP và dùng chung model cho regret. **Không giả định assignment nominal/regret của dữ liệu mới trùng nhau** như tác giả quan sát trên dữ liệu gốc.
- GA: số guided floor(.8*P), ít nhất 1; elite ceil(.05*P), ít nhất 1; mutation chọn một factory khác đều xác suất. Loại duplicate, sau 100 lần trùng liên tiếp dùng unseen uniform (vét cạn dự phòng nếu không gian ≤100000). Do đó fallback cũng là một thành phần có thể ảnh hưởng kết quả; lưu `fallback_count`.
- SA: T0=max(.01*|initial objective|,1), hạ nhiệt theo tỷ lệ distinct evaluations tới .001*T0; sau 100 proposal trùng dùng unseen candidate. Đây là lịch nhiệt tự chọn.
- Cost warm start: softmax của âm tổng rank supply và fixed theo từng hub, sau đó cùng cơ chế 80/20. Không tuyên bố bằng baseline riêng của tác giả.
- Chỉ nhận MILP được solver xác nhận optimal với `mip_rel_gap=0`; timeout 120 giây báo lỗi thay vì dùng incumbent như exact. Chưa có nhánh thí nghiệm surrogate regret cho Instances 14–15.
- Ma trận ràng buộc LP được tái sử dụng; SciPy tạo lại solver phía trong mỗi lời gọi. Không có persistent HiGHS basis/warm start. Thời gian đo bao gồm inference và search, loại model construction, reference solves, training/loading; không so sánh trực tiếp với tỷ số tốc độ của tác giả.
- Bảng thống kê ở repo là cho các fixture mới: Holm ba cặp phương pháp trong mỗi fixture/formulation; Friedman/Holm trên sáu nhóm test. Sensitivity chỉ có một cặp GA/GNN–GA trong mỗi formulation. Không gọi đây là họ 26 tests của bài.

## Khi có dữ liệu gốc

Thay JSON các instance và manifest (schema trong README), giữ phân tách train 1–9, validation 10–12, test 13–15. Cần thay generator/features/search tie policy bằng mã tác giả để kiểm tra tái lập số học. File `paper_protocol.json` giữ thông số công bố; các kích thước không biết là `null`, không tự đoán. `paper_reported_results.json` chỉ chứa số tác giả báo cáo, không phải đầu ra chương trình.

Kết quả trên dữ liệu mới kiểm chứng triển khai chạy được và cho phép nghiên cứu phương pháp; không xác nhận hay bác bỏ các con số và kết luận thực nghiệm trên dữ liệu gốc.
