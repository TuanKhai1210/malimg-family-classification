# Hướng dẫn review base và chạy thực nghiệm

Phiên bản 13/10/2026. Đây là base của pipeline core Task 3, chưa có điểm benchmark.
Không cần viết lại luồng chuẩn bị dữ liệu, extraction, classifier hoặc đánh giá.
Nhóm chịu trách nhiệm review lựa chọn, chạy thật, phân tích và hoàn thiện báo cáo.

## 1. Ba người review gì trước khi train?

| Người | File/đầu ra cần xem | Quyết định cần chốt |
|---|---|---|
| A | `results/eda/`, `metadata/*_v1.*`, `modules/splits.py` | Dataset/24 lớp được duyệt; split/đại diện không leakage; số file và nhóm |
| B | `configs/project.json`, `modules/features.py`, `modules/classifiers.py` | Weights, resize, batch/device, C grid, train unit, PCA nếu dùng |
| C | `modules/experiments.py`, notebook, `tests/`, khung báo cáo | Validation chọn cấu hình; test riêng; output đủ provenance; notebook chạy sạch |

Base đã chuẩn bị đủ 9.339 ảnh; loại 800 file Yuner.A khỏi benchmark, vẫn giữ
trong kiểm kê. Split ứng viên: **5.978 train / 1.281 val / 1.280 test**, tương ứng
**5.615 / 1.202 / 1.201 nhóm pixel**. Seed 42; không giao pixel hash; mỗi lớp
có ít nhất hai nhóm khác pixel ở validation và test. Đọc `docs/base_review.md`
để xem bằng chứng và giới hạn. Nhóm/GV chưa phê duyệt split này.

## 2. Chạy local

Python 3.11 là mục tiêu của CI; kiểm tra local được ghi riêng theo môi trường
thật. Tạo virtual environment, kích hoạt rồi cài:

```bash
python -m pip install -r requirements.txt
python scripts/check_environment.py
python -m unittest discover -s tests -v
python scripts/run_pipeline.py prepare
```

Trên máy đã có ZIP, dùng `prepare --archive PATH` để không tải lại; vẫn kiểm
kích thước/hash. Các đường dẫn dữ liệu/artifact trong config phải nằm trong repo.
Nguồn công khai và checksum đã kiểm nằm trong `configs/project.json`.

Nếu cần GPU local, cài cặp torch/torchvision phù hợp CUDA theo
[hướng dẫn PyTorch](https://pytorch.org/get-started/previous-versions/), rồi chạy
`check_environment.py`. Cặp CPU đã kiểm cho base là torch 2.8.0 / torchvision
0.23.0. Requirements cho phép cặp mới tương thích; artifacts ghi lại phiên bản
thực tế. CI cài cặp CPU này; phiên bản CUDA của Colab cần được kiểm khi chạy thật.

Sau review:

```bash
python scripts/run_pipeline.py train
```

Lệnh này tự chuẩn bị/kiểm dữ liệu, trích hoặc dùng cache khớp, nạp `.npy`, train
E2/E3/E4 và chọn C bằng validation. Không cần một lệnh extraction riêng. Có thể
dùng `extract` trước để chuẩn bị features hoặc kiểm batch/GPU mà chưa fit classifier.

Sau khi khóa mọi lựa chọn:

```bash
python scripts/run_pipeline.py final
```

Lệnh final không fit lại scaler/classifier. Nó kiểm hash/config/split/feature/model,
đánh giá test một lần và khóa output. Chạy lại cùng final chỉ đọc kết quả đã khóa.
Phiên bản numpy/scikit-learn/joblib phải khớp môi trường validation trước khi
nạp model; artifacts có ghi phiên bản để nhóm khôi phục đúng môi trường.
Đổi config hoặc artifact sau khóa sẽ báo lỗi. Không xóa lock để tiếp tục chọn theo
test; nếu có lỗi thật, review provenance và ghi rõ quyết định xử lý.

## 3. Chạy Colab

1. Mở `notebooks/main_colab.ipynb`, chọn GPU runtime.
2. `STAGE='prepare'` rồi Run all để review dữ liệu/EDA. Notebook tự clone nhánh
   `feat/reviewable-core-pipeline`, cài thư viện và tải link công khai, không mount Drive.
3. Sau review đặt `STAGE='all'`; giữ `FINAL_TEST=False` để train/chọn bằng validation.
4. Khóa phương án rồi dùng `STAGE='final'` trong cùng runtime. Với lần kiểm bản nộp
   từ runtime sạch, dùng `STAGE='all'` và `FINAL_TEST=True` để chạy toàn pipeline.
5. Bật `CREATE_DRAFT_ZIP=True` để tải code/artifacts về máy trước khi runtime mất.

Notebook không biến preview thành kết quả hoàn chỉnh. Chưa kiểm Run all trên
Colab thật ở thời điểm bàn giao base. Máy local hiện dùng CPU để kiểm code;
điều đó không xác nhận CUDA/driver trên GPU của nhóm.

## 4. Chỉnh gì ở config?

- `features`: batch size, device (`auto/cpu/cuda/mps`), workers, image size,
  resize (`direct/pad`), weights. Weights mặc định: ResNet50 V2, ConvNeXt-Tiny V1.
- `training`: seed, max_iter, class_weight, train_unit (`file/representative`),
  PCA null hoặc số chiều/tỷ lệ variance. Scaler/PCA chỉ fit train.
- `experiments`: ID, extractor, classifier, C grid. Core là E2/E3/E4, ba C mỗi model.
- `excluded_families`, split_seed/target: quyết định dữ liệu cần A/C review trước
  chạy. Thay quyết định phải dùng manifest_version mới; không âm thầm sửa v1.

`configs/baselines.json` là snippet tùy chọn E0 (kích thước ảnh + LR), E1 (HOG + SVM).
Ghép danh sách vào **bản sao đầy đủ** của project.json rồi dùng `--config PATH`.
Fine-tuning/DL bonus chưa nằm trong base core; nhóm chỉ mở sau khi core ổn định.

Cache ghi manifest, sample IDs, labels, archive hash, weights, transform, versions
và code extraction. Thay một phần tạo cache mới; cache thiếu/sai checksum không
được tái dùng. Runtime/device/version khác có thể cần trích lại. Data preparation
kiểm ảnh thật mỗi lần dùng lại; chi phí này phục vụ kiểm tính toàn vẹn.

## 5. Đầu ra và cách viết báo cáo

| Nơi | Nội dung |
|---|---|
| `metadata/` | Dataset/split manifest, label map, summary, hashes/provenance |
| `results/eda/` | JSON/CSV phân bố lớp, kích thước, trùng pixel; 4 hình EDA |
| `features/<extractor>/<signature>/` | X/y/IDs/metadata train/val/test và completion marker |
| `results/runs/validation_*/` | Config/environment, mọi trial/warning/model, val predictions/metrics/analysis |
| `results/selection.json` | C được chọn và hash khóa từng model/feature store |
| `results/final_test/` | Test predictions, metrics, phân tích lỗi, confusion/per-class figures |
| `results/comparison.csv` | Bảng tổng hợp có cột stage/unit, chi phí/feature dimension |

Macro-F1 chính dùng một đại diện cố định mỗi nhóm pixel; metric toàn bộ file là
cột riêng. Train mặc định dùng các file đã giữ, có class weight; train_unit có thể
đổi trước thực nghiệm. Không gọi decision scores của Linear SVM là xác suất.
Đọc convergence warnings trước khi kết luận; tăng max_iter nếu pilot chưa hội tụ.

Dùng khung `reports/final_report_template.md`. Điền dữ liệu và kết quả thật,
phân tích lỗi/chi phí/giới hạn, nguồn tham khảo và đóng góp thực tế của nhóm.
Kết quả synthetic tests chỉ kiểm phần mềm, không đưa vào bảng MalImg.

## 6. Đóng gói

```bash
python scripts/package_submission.py
```

Tạo `dist/malimg_review_base.zip`: code, notebook, tests, config, metadata,
features và kết quả nếu có; không chứa raw images/ZIP dataset/virtual environment.
Mỗi file có hash trong PACKAGE.json. Model joblib do pipeline tạo được giữ để
nhóm nạp final từ artifacts đã kiểm; chỉ nạp gói tin cậy của nhóm.

Sau khi có báo cáo PDF, thông tin môn/nhóm, scope được duyệt và final results:

```bash
python scripts/package_submission.py --final --output dist/malimg_submission.zip
```

Lệnh final kiểm report tồn tại, approval, final lock/checksum và feature/model
được chọn. Vẫn cần nhóm kiểm nội dung PDF, link truy cập và Run all sạch trước
khi nộp. Mặc định package có features; nếu nộp bằng link ngoài phải review riêng
theo đề. Hạn nội bộ hiện theo thông tin nhóm: **30/11/2026**; lịch LMS có ưu tiên.

## Sao lưu và khôi phục

Xem [checkpoint_and_recovery.md](checkpoint_and_recovery.md) trước một lượt chạy dài.
Notebook hỗ trợ Drive cá nhân tùy chọn, snapshot theo thời điểm, log từng trial và
resume các trial hoàn tất đã kiểm identity. Lần fit đang dở phải chạy lại; lưu
notebook riêng không lưu features/models trên máy Colab.
