# Huấn luyện và khôi phục trên Kaggle

Import [main_kaggle.ipynb](../notebooks/main_kaggle.ipynb) lên Kaggle.
Notebook Colab được giữ riêng cho luồng hiện tại và kiểm tra yêu cầu môn học.

## Hai kiểu huấn luyện

| ENGINE | Mô hình | Resume |
|---|---|---|
| `classical` | LogisticRegression / LinearSVC, grid C cũ | Sau trial hoàn tất; fit đang dở phải chạy lại |
| `sgd_epoch` | SGDClassifier trên frozen features, log-loss hoặc hinge | Sau epoch đã commit; giữ preprocessing, optimizer counter, RNG, history và best epoch |

SGD là thí nghiệm mới, không phải state của solver LR/SVC cũ. SGD dùng `alpha`,
không chuyển C thành cùng một tham số. Split, label map, metric và baseline cũ
được giữ nguyên. GPU trích features; classifier scikit-learn chạy CPU.

## Lần đầu

1. Bật Internet, chọn GPU nếu cần trích features. Giữ notebook/checkpoint inputs
   private khi dùng tài khoản cá nhân.
2. Thêm Kaggle Secrets `KAGGLE_USERNAME` và `KAGGLE_KEY` từ tài khoản của bạn,
   bật quyền cho notebook. Không dán key vào code/output, không commit kaggle.json.
   Environment đã có `KAGGLE_API_TOKEN` cũng được adapter hỗ trợ.
3. Trong **controls**, thay BACKUP_DATASET bằng
   `ten-kaggle-cua-ban/malimg-experiment-checkpoints`. Dùng Dataset riêng, một
   writer. Checkpoint đầu tạo private Dataset nếu chưa tồn tại; public/không
   xác minh được privacy bị từ chối, không tự đổi quyền chia sẻ.
4. Chọn ENGINE='sgd_epoch', ACTION='train', FINAL_TEST=False, RUN_ID=None.
   Mặc định là ResNet50 + SGD log-loss,30epoch,batch256,patience6; đây là cấu hình
   khởi đầu để review, chưa có benchmark MalImg cho nhánh SGD.
5. Chạy từ controls. Kaggle thường có sẵn ML packages; bật INSTALL_DEPENDENCIES
   khi cần. Adapter API được pin kaggle==2.2.5; mặc định không tự nâng PyTorch.
6. Lưu Run ID in ra. Chỉ **REMOTE VERIFIED** xác nhận đã đọc lại bản remote;
   `local checkpoint committed` mới xác nhận lưu trên VM.

MAX_SESSION_MINUTES là ngân sách chủ động, không phải thời lượng Kaggle bảo đảm.
Chỉ pause giữa epoch. EPOCHS_THIS_SESSION chạy một đoạn ngắn mà không thay planned
EPOCH_SETTINGS['epochs']. Khi resume, notebook nạp config/settings của run cũ.

## Mỗi epoch lưu gì

```text
results/epoch_runs/sgd_<timestamp>_<identity>/
  config.json, settings.json, identity.json, environment.json
  status.json, history.csv, selection.json
  epochs/epoch_000001/{state.joblib,metrics.json,commit.json}
  epochs/epoch_000002/...
  validation/{metrics,predictions,per_class,confusion_matrix,errors,...}
```

State có scaler/PCA fit trên train, SGD learning-rate counter, shuffle RNG,
history, best epoch và patience. Class weights tính trên train một lần rồi giữ.
Staging ghi/fsync state và publish thư mục epoch; staging chưa commit được bỏ
qua khi resume và không vào backup. Corrupt committed checkpoint làm dừng thay
vì âm thầm quay về bản cũ. Mất VM giữa minibatch phải replay epoch đang dở.
Chỉ load own-group joblib: checksum kiểm bytes, không xác thực tác giả.

Sau mỗi epoch, tạo một version Dataset private có:

```text
latest.json             # run/checkpoint ID và checksums
recovery.zip
  snapshots/<timestamp>_<label>.zip
  features/<feature-identity>.zip
```

Snapshot gồm source/config/manifest và model/results/history. Không gồm raw
images, dataset ZIP, .git, .venv hoặc credentials. Không tự xóa version cũ.
Uploader chờ ready, kiểm private, tải receipt/payload về và so hash. Lỗi mạng,
quota, writer khác, sai hash hoặc chưa ready làm dừng trước epoch sau; local
checkpoint còn nếu VM còn. Khi VM mất chỉ cứu được mốc đã lưu thành công bên ngoài.

Mỗi version hiện gửi lại snapshot và feature bundle rồi readback; có chi phí
băng thông/thời gian/lưu trữ. Không upload theo minibatch. Giới hạn API/Dataset
có thể thay đổi; không bỏ qua lỗi rồi coi checkpoint đã an toàn. Kaggle Notebook
Output/Save Version là bản sao bổ sung; pipeline không tự bấm Save Version.

## Runtime mới: exact resume

```python
ACTION = 'train'
ENGINE = 'sgd_epoch'
AUTO_BACKUP = True
BACKUP_DATASET = 'ten-kaggle-cua-ban/malimg-experiment-checkpoints'
RESTORE_REMOTE = True
RUN_ID = 'sgd_<ID-ban-da-luu>'
RESTORE_TO = '/kaggle/working/malimg-restored'  # chưa tồn tại
NEW_RUN_FROM_BACKUP = False
FINAL_TEST = False
```

Chạy đủ cell controls trước setup. Restore source trong snapshot, config/settings
đúng run; không tự chọn thí nghiệm theo tên mới nhất. Nếu version cuối thuộc run
khác, run thiếu sẽ báo lỗi. Có thể attach Dataset version cũ làm input, đặt
RESTORE_PAYLOAD='/kaggle/input/.../recovery.zip', RESTORE_REMOTE=False.

Exact resume kiểm Python major/minor, numpy/scipy/sklearn/joblib, source,
config/manifest/sample order/bytes features. Pin thư viện không đổi được Python.
Nếu đã import trước khi đổi version, restart Python rồi dùng workspace đã restore.
Sau restore một lần, trước setup tiếp theo đặt:

```python
RESTORE_REMOTE = False
RESTORE_SNAPSHOT = None
RESTORE_PAYLOAD = None
WORKING_ROOT = '/kaggle/working/malimg-restored'
```

Giữ RUN_ID để tiếp tục; None tạo run mới. Không restore đè thư mục đã tồn tại.

## Import backup Colab cũ

Giữ đúng cặp snapshots/ và features/, attach riêng tư lên Kaggle. Để dùng fixed
features cũ với code/môi trường hiện tại cho một run SGD mới:

```python
RESTORE_REMOTE = False
RESTORE_SNAPSHOT = '/kaggle/input/backup-cua-ban/snapshots/SNAPSHOT.zip'
NEW_RUN_FROM_BACKUP = True
RUN_ID = None
```

Source upgrade được ghi tường minh trong MIGRATION.json; giữ config/artifacts/run
cũ. Không sửa hash để nhận một model khác source/environment như cùng run.
Sau migration và khi resume SGD mới, đặt NEW_RUN_FROM_BACKUP=False.
Muốn chỉ xem baseline: ENGINE='classical',ACTION='analyze', không fit.
Raw images không có trong backup: panel ảnh lỗi có thể vắng, CSV/confusion vẫn đủ.
Khi cần ảnh, prepare lại từ ZIP đã audit. Reuse fixed features không tuyên bố đã
audit lại raw images hoặc trích lại trên GPU hiện tại.

## Các chế độ

- prepare: dữ liệu/split và bảng EDA, chưa fit.
- extract: chuẩn bị/kiểm features, chưa fit classifier.
- train: train/resume; SGD chọn highest validation representative Macro-F1,
  tie chọn epoch sớm nhất, early stopping dùng cùng metric.
- analyze: chỉ đọc history/per-class/confusion/errors và đường học, không
  prepare/extract/fit. Chọn RUN_ID; None ở nhánh SGD liệt kê các run.
- final và FINAL_TEST=True: RUN_ID đã completed, nhóm khóa lựa chọn trước khi
  mở test. Không fit; lock global ngăn tuning hoặc chọn engine khác sau test.
- backup: sao lưu thủ công, không train.

Chọn engine/phạm vi final trước khi mở test. Không dùng test để chọn loss/alpha/
epoch. Final completed không đổi artifacts được đọc lại mà không load/chấm test
lại. Final started/failed cần review lock; không tự xóa để thử tiếp. Nhánh epoch
có kết quả final riêng; final course packaging/report format vẫn cần review.

malimg-review-results.zip thiếu features, chỉ là export đọc/chia sẻ kết quả;
recovery.zip mới là payload resume. Notebook không dump full JSON hàng nghìn dòng.

## Phạm vi xác minh

Tests local dùng ảnh/features synthetic, kiểm optimizer/RNG chạy liền so với
interrupted/segmented/cross-root resume. Upload/readback/privacy/timeout/race dùng
API fixture. Notebook chạy đủ cells rồi restore/resume trong Python process mới
với fit/extraction/download bị cấm. Không coi đây là MalImg SGD benchmark hoặc
live upload trên tài khoản Kaggle của nhóm.

Nguồn: [Kaggle CLI](https://github.com/Kaggle/kaggle-cli),
[Dataset API](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md),
[SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html).
