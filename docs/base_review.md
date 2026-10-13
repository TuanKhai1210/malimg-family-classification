# Bằng chứng bàn giao base — 13/10/2026

## Phạm vi đã có

Core Task 3: chuẩn bị dữ liệu, EDA, split theo nhóm pixel, frozen ResNet50 và
ConvNeXt-Tiny, lưu/nạp features, LR/SVM, validation search, final test riêng,
metrics/figures/result registry, Colab front end và đóng gói. E0/E1 đã có code
baseline tùy chọn. DL fine-tuning bonus vẫn là hướng mở rộng.

## Kiểm dữ liệu đầy đủ

Đã chạy chuẩn bị và EDA bằng archive MalImg đã kiểm. Không fit classifier trên
MalImg hoặc công bố điểm benchmark trong lượt bàn giao này.

| Chỉ tiêu | Kết quả |
|---|---:|
| Ảnh nguồn / family | 9.339 / 25 |
| Nhóm pixel nguồn | 8.019 |
| Ảnh benchmark / family | 8.539 / 24 |
| Nhóm pixel benchmark | 8.018 |
| File train / val / test | 5.978 / 1.281 / 1.280 |
| Nhóm pixel train / val / test | 5.615 / 1.202 / 1.201 |
| Giao pixel hash giữa các tập | 0 |
| Nhóm tối thiểu mỗi lớp ở val/test | Ít nhất 2 |
| Lệch tỷ lệ file lớn nhất theo lớp | Khoảng 1,038 điểm phần trăm |

Split ứng viên do thuật toán chia nhóm theo nhãn, ưu tiên tiến gần tỷ lệ file
70/15/15, seed 42. Chọn đại diện trước dự đoán bằng đường dẫn nhỏ nhất mỗi nhóm.
Manifest SHA-256:

`e643d7b213ba76277571dbd27da0f361fe9576e2454b503cff3b4eeebadada69`

Thuật toán, lệch từng lớp và provenance đầy đủ nằm trong
`metadata/split_summary_v1.json`. Dữ liệu nguồn vẫn được giữ; 800 file Yuner.A
chỉ excluded khỏi benchmark. Chưa chứng minh độc lập mẫu malware hoặc loại
near-duplicates. Scope/split vẫn cần nhóm và giảng viên duyệt.

## Kiểm phần mềm

- 55 tests thành công, không skip, trên virtual environment local (54 tests ở
  lượt suite đầy đủ và nhóm integration/packaging được kiểm lại sau sửa cuối).
- Kiểm archive/path/ảnh hỏng, reproducibility, leak/cross-label conflict,
  ảnh/artifact bị sửa và đổi policy mà giữ cùng manifest version.
- Kiểm alignment X/y/IDs, cache/checksums, weights/transform và source ảnh.
- Kiểm hình dạng hai kiến trúc torchvision thật bằng weights=None: 2.048 và
  768 chiều. Test không tải pretrained weights; không coi đây là chất lượng pretrained.
- Integration bằng ảnh giả lập: prepare → EDA → extraction → persistence →
  classifier validation → final khóa → draft ZIP. Kiểm train-only scaler/PCA,
  không nạp test arrays khi chọn, không refit final, warnings/failures và stale artifacts.
- Notebook kiểm schema và cú pháp; outputs/execution counts được để trống.
- Đã thực thi đủ 6 code cells của notebook trên dữ liệu giả lập local, gồm
  validation/final, từ thư mục upload không có `.git`. Bước cài/tải bị tắt cho
  kiểm giả lập; chưa coi đây là bằng chứng Colab sạch/GPU hoặc tải weights thật.

Môi trường local: Windows 11, Python 3.12.14, torch 2.8.0+cpu, torchvision
0.23.0+cpu, numpy 2.5.2, scikit-learn 1.9.1, Pillow 12.3.0, matplotlib 3.11.2,
scikit-image 0.26.0, joblib 1.6.0. `pip check` không báo dependency conflict.
CI nhắm Python 3.11 và CPU; không suy ra đã kiểm GPU/Colab từ môi trường local.

## Việc nhóm thực hiện tiếp

Review cấu hình/split và xác nhận scope; chạy frozen features cùng E2/E3/E4 trên
GPU; đọc warnings/validation outputs rồi khóa lựa chọn; đánh giá test; phân tích
và điền báo cáo thật; kiểm Run all Colab sạch, thông tin môn/nhóm, evidence và ZIP.
Không cần viết lại core trước khi train. Tham khảo `review_and_train.md`.
