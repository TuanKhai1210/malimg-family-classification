# Lưu và khôi phục thực nghiệm

Chuyển sang Kaggle: dùng [notebook và hướng dẫn Kaggle](kaggle_training.md).
Nhánh SGD mới có checkpoint theo epoch và readback Dataset private; hướng dẫn
dưới đây giữ cho LR/LinearSVC và Drive theo trial.

Notebook `.ipynb` giữ code/output đã lưu; các file trên `/content` thuộc máy
Colab và có thể mất khi máy bị thu hồi. Lưu notebook không đồng nghĩa lưu
features, model hoặc kết quả. Drive là nơi sao lưu cá nhân tùy chọn; pipeline
vẫn tải dữ liệu từ nguồn công khai và chạy trên ổ đĩa Colab.

## Dùng từ lần chạy mới

1. Mở notebook của nhánh `feat/resumable-drive-checkpoints` đang được review.
2. Trong cell 1 đặt:

   ```python
   STAGE = 'all'
   FINAL_TEST = False
   USE_DRIVE_BACKUP = True
   BACKUP_DIR = '/content/drive/MyDrive/ML_MalImg_Backups'
   RESUME_RUN = None
   RESTORE_SNAPSHOT = None
   ```

3. Chạy các cell theo thứ tự. Cell setup mount Drive để bạn tự duyệt quyền.
   Code, dữ liệu và train dùng thư mục làm việc cục bộ. Mỗi extractor hoàn tất,
   mỗi trial kết thúc, validation hoàn tất và interrupt có xử lý đều tạo bản sao.
4. Giữ cả hai thư mục trên Drive:

   ```text
   ML_MalImg_Backups/
     features/<checksum>.zip
     snapshots/<UTC timestamp>_<checkpoint label>.zip
   ```

Features được đóng bundle theo nội dung, dùng chung giữa các mốc có cùng features.
Khi có thêm extractor, bundle mới chứa tập features mới; bundle cũ vẫn giữ lại.
Snapshot có source, notebook, requirements, config, manifest/split/label map,
metadata, các run, model `.joblib`, metrics/predictions/warnings/figures và môi
trường đã ghi. Không đưa raw images, ZIP dữ liệu, `.git`, `.venv` hoặc `.env` vào
backup. Các paths output phải ở trong repo; nếu đổi `results_dir`/`features_dir`,
lưu chúng vào `configs/project.json` trước khi bật backup.

Mỗi snapshot có checksum từng file và tham chiếu đến đúng feature bundle. Copy
được đọc lại kiểm checksum rồi mới công bố tên cuối. Lỗi Drive/dung lượng làm
dừng pipeline sau checkpoint cục bộ; không được hiểu là đã sao lưu thành công.
Không tự xóa lịch sử: nhóm tự quản lý dung lượng và giữ bundle mà snapshot cần.
Không chạy hai pipeline cùng ghi một workspace/backup đồng thời.

## Các mức checkpoint

| Mức | Lưu được | Khôi phục được |
|---|---|---|
| Feature cache hoàn tất | `.npy`, labels, IDs, metadata/hashes | Bỏ qua extraction nếu identity/environment/device khớp |
| Trial hoàn tất | scaler/PCA/classifier, predictions, metrics, warnings, hashes | Bỏ qua `fit` của trial đó |
| Trial đang chạy | trạng thái, thời điểm bắt đầu, log elapsed mỗi 30 giây | Biết đang ở đâu; phải fit lại trial này |
| Snapshot Drive | source/config/metadata/runs và feature bundle đi kèm | Chuyển sang máy Colab mới |

`LogisticRegression` và `LinearSVC` hiện tại không có cơ chế resume giữa một lần
`fit`. Checkpoint không làm SVM tự nhanh hơn hoặc chữa nguyên nhân hội tụ chậm.
Log elapsed là thời gian đã chờ trong fit, không phải phần trăm hội tụ.
Khi VM mất đột ngột chỉ khôi phục đến snapshot đã upload thành công gần nhất;
không có tiến trình nền nào tiếp tục upload sau khi VM bị mất.

## Resume khi máy Colab còn file

Tìm run ID trong log `Starting validation: validation_...` hoặc
`results/runs/validation_.../resume.json`. Đặt đúng ID, không chọn run chỉ vì tên
mới nhất:

```python
STAGE = 'train'
FINAL_TEST = False
RESUME_RUN = 'validation_<ID thực tế>'
RESTORE_SNAPSHOT = None
```

Chạy controls, setup, configuration rồi pipeline theo thứ tự. Configuration nạp
lại config của run được chọn. Run phải khớp config, manifest/sample order, feature
bytes, implementation, Python major/minor và numpy/scipy/sklearn/joblib. Hoàn tất
không đồng nghĩa hội tụ: trial có `ConvergenceWarning` vẫn được giữ để nhóm review.
Không thay grid/solver/PCA/max_iter rồi dùng resume như cùng một thí nghiệm.

Trial hoàn tất phải có đủ artifact và checksum. Sai hash thì dừng, không âm thầm
chấp nhận hoặc ghi đè. Trial đang dở/failed được giữ vào `trial_XX_attempt_<UTC>`
rồi thử lại ở `trial_XX`. Mỗi run dùng một process lock được hệ điều hành nhả khi
process chết. Không xóa final-test lock để resume validation sau khi đã mở test.

## Khôi phục trên máy Colab mới

1. Mở notebook mới trong runtime mới, chọn cùng Python major/minor và GPU phù hợp.
2. Bật `USE_DRIVE_BACKUP=True`; điền `RESTORE_SNAPSHOT` là đường dẫn đầy đủ của
   file trong `snapshots/`. `RESTORE_TO` phải là thư mục chưa tồn tại, ví dụ
   `/content/malimg-restored`.
3. Điền `RESUME_RUN` của run bên trong snapshot, `STAGE='train'`, `FINAL_TEST=False`.
4. Chạy các cell theo thứ tự. Setup dùng subprocess kiểm cả hai ZIP và từng hash
   trước khi dựng một workspace mới; source của snapshot được dùng để resume.
   Model chưa được deserialize trong bước restore.
5. Setup cài requirements và pin lại numpy/scipy/sklearn/joblib đã ghi. Môi trường
   Python/PyTorch/Torchvision, phiên bản CUDA và loại thiết bị vẫn cần tương thích
   để reuse cache; kiểm `RESTORED_BACKUP.json` và `environment.json`. Nếu thư viện
   đã import trước khi đổi version, restart Python rồi chạy lại setup (không
   restore đè thư mục đã tồn tại; đặt `RESTORE_SNAPSHOT=None` và
   `WORKING_ROOT=RESTORE_TO` để dùng workspace đã phục hồi). Resume sẽ từ chối
   identity không khớp.

Raw dataset không có trong backup: pipeline tải lại nguồn công khai, kiểm archive
và dựng lại dữ liệu trước khi đối chiếu manifest/features. Đó là hành vi dự kiến.
Giữ cả `features/` và `snapshots/` cùng cấu trúc; một snapshot ZIP riêng lẻ chưa đủ.
Chỉ dùng backup do chính nhóm tạo, vì model joblib có thể thực thi mã khi nạp.

CLI tương đương (không cần Google Drive API):

```bash
python scripts/manage_checkpoints.py save --destination /path/outside/repo/backups --label before_restart
python scripts/manage_checkpoints.py restore /path/backups/snapshots/SNAPSHOT.zip --destination /path/new-workspace
cd /path/new-workspace
python scripts/run_pipeline.py train --resume-run RUN_ID --backup-dir /path/outside/repo/backups
```

## Cứu artifacts của notebook cũ trước khi đổi code

Nếu repo cũ và file vẫn còn trong Files, chạy một cell riêng sau khi fit đã dừng.
Cell này dùng `build_submission` đã có trong base cũ, không cần công cụ mới:

```python
from pathlib import Path
from datetime import datetime, timezone
import sys
from google.colab import drive

drive.mount('/content/drive')
candidates = [Path('/content/malimg-family-classification'), Path.cwd()]
old_root = next((p for p in candidates if (p / 'modules/submission.py').is_file()), None)
if old_root is None:
    raise FileNotFoundError('Repo cũ không còn ở các đường dẫn này. Kiểm Files trước khi tạo lại.')
sys.path.insert(0, str(old_root))
from modules.submission import build_submission

target = Path('/content/drive/MyDrive/ML_MalImg_Backups/legacy')
target.mkdir(parents=True, exist_ok=True)
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
backup = build_submission(old_root, target / f'legacy_{stamp}.zip', draft=True, include_features=True)
import zipfile
with zipfile.ZipFile(backup) as saved:
    if saved.testzip() is not None:
        raise OSError('Legacy backup failed CRC check; keep the runtime and inspect Drive.')
print('Legacy backup:', backup)
```

ZIP legacy này độc lập, có `PACKAGE.json`; nó không phải định dạng hai phần của
`restore_backup`. Giữ để review/giải nén riêng và kiểm checksum theo PACKAGE.json.
Run cũ thiếu `resume.json` không được resume bằng code mới: giữ kết quả cũ, bắt đầu
run validation mới. Features chỉ reuse khi source extractor/environment/settings
khớp. E2 cũ đã ghi tổng fit khoảng 12,4 giây trong artifact nhận ngày 23/10/2026;
không thay checksum source để nhận các model cũ như được tạo bởi code mới.

## Lỗi `No module named modules`

Đây là lỗi tìm code/setup, xảy ra trước fit. Chạy controls → setup → configuration
đúng thứ tự; setup thêm repo root vào `sys.path`. Kiểm repo còn tồn tại trong Files
trước khi clone lại. Chỉ lưu notebook không giữ `/content`; không kết luận dữ liệu
còn/mất từ traceback import hoặc từ output cũ trong notebook.

## Lỗi `Path(WORKING_ROOT)` nhận bool

`WORKING_ROOT` là đường dẫn tùy chọn, không phải nút bật/tắt. Ở cell đầu đặt
`WORKING_ROOT = None` để setup tự tìm repo; chạy lại cell đầu rồi cell setup.
Nếu đang dùng workspace đã restore, đặt chuỗi đường dẫn, ví dụ
`WORKING_ROOT = '/content/malimg-restored'`. Bật sao lưu bằng
`USE_DRIVE_BACKUP = True`; không đặt `WORKING_ROOT = True` hoặc `False`.
Notebook mới kiểm giá trị này trước khi mount Drive hoặc clone/cài thư viện.

## Lỗi `Path(...)` nhận bool khi gọi pipeline

Các biến đường dẫn không phải toggle. Với một run validation mới, dùng:

```python
ARCHIVE_PATH = None       # tự tải dữ liệu công khai
RESUME_RUN = None         # run mới
RESTORE_SNAPSHOT = None   # không restore ZIP
FINAL_TEST = False        # còn chọn cấu hình trên validation
USE_DRIVE_BACKUP = True
```

Nếu cần resume/restore, thay `None` bằng run ID hoặc đường dẫn thật trong dấu
nháy. Giữ `BACKUP_DIR` là đường dẫn Drive, không đổi thành `True`. Setup và pipeline
kiểm các path options trước I/O, báo đúng tên biến sai thay vì lỗi sâu trong
`pathlib`. Khi sửa controls, chạy lại controls → setup → configuration → pipeline;
thay code trên GitHub không tự đổi giá trị đang có trong kernel Colab.

Nguồn về runtime/Drive: [Colab FAQ](https://research.google.com/colaboratory/faq.html),
[mã nguồn drive.mount](https://github.com/googlecolab/colabtools/blob/main/google/colab/drive.py).
