# BÀI TẬP LỚN MACHINE LEARNING

Phân loại họ mã độc từ ảnh biểu diễn dữ liệu nhị phân

BẢN BÀN GIAO CHO NHÓM | Phiên bản 1.0 | 30/09/2026

## 01. Đọc trước: mục tiêu và quyết định chung

**Tên đề tài:** Phân loại họ mã độc từ ảnh MalImg bằng đặc trưng pretrained và mô hình học máy truyền thống.

**Tên tiếng Anh:** Malware Family Classification from MalImg Images Using Pretrained Visual Features and Traditional Classifiers.

Nhóm thực hiện **Task 3 - Machine Learning with Image Data**, học kỳ I năm học 2026-2027, môn Machine Learning. Nhóm có 3 người, hạn nộp do nhóm cung cấp là **30/11/2026**. Người phụ trách hiện ký hiệu A/B/C; điền tên thật trong buổi họp đầu tiên.

Mỗi đầu vào là ảnh grayscale biểu diễn nội dung byte của một mẫu malware; đầu ra là nhãn family trong tập lớp đã chọn. Sản phẩm là một nghiên cứu thực nghiệm có thể chạy lại trên Colab. Dataset này không cung cấp bài toán benign/malicious, nên kết luận giới hạn ở phân loại family đã biết trong benchmark.

| Nội dung | Quyết định cho nhóm |
|---|---|
| Dữ liệu chính | MalImg; benchmark chính dùng 24 family, loại Yuner.A |
| Quy mô theo kiểm kê | 8.539 file, 8.018 nhóm pixel khác nhau sau khi loại Yuner.A |
| Core | Frozen ResNet50 và ConvNeXt-Tiny; Logistic Regression và Linear SVM |
| Feature files | Lưu .npy cùng nhãn, sample IDs và metadata; nạp lại để fit classifier |
| Split | Mục tiêu khoảng 70/15/15; chia theo nhãn và nhóm pixel trùng; seed 42 |
| Đánh giá chính | Macro-F1 trên một ảnh đại diện cho mỗi nhóm pixel ở tập đánh giá |
| Phụ trách | A: dữ liệu/EDA; B: đặc trưng/ML; C: đánh giá/Colab |
| Mở rộng | HOG, shortcut audit, resize; fine-tuning và GitHub để hướng tới bonus |

**Đã chốt nội bộ:** phương án 24 lớp và phân vai trên để bắt đầu làm. **Chưa được xác nhận bên ngoài:** dataset không trùng nhóm khác, giảng viên chấp thuận MalImg và tập con 24 lớp, lịch báo cáo tiến độ trên LMS. A chịu trách nhiệm hỏi và ghi lại phản hồi; không ghi trong báo cáo rằng GV đã duyệt khi chưa có bằng chứng.

Đã có bản tải và kiểm kê toàn bộ ảnh; chưa có mô hình được huấn luyện, feature pretrained, split manifest chính thức hoặc kết quả Run all trên Colab. Tài liệu này thay thế các lựa chọn dataset/backbone còn bỏ ngỏ trong bản khung cũ khi nhóm lập kế hoạch. Không dùng nó như báo cáo kết quả đã hoàn thành.

## 02. Yêu cầu của môn học và sản phẩm phải nộp

Theo MLAssignment_ELv2.pdf, nhóm chọn ít nhất một trong ba dạng dữ liệu. Với Task 3, nhóm phải có EDA ảnh, pretrained feature extraction, lưu feature .npy/.h5, classifier downstream, pipeline cấu hình được và so sánh các pretrained feature models. Hai extractor là cách nhóm cụ thể hóa yêu cầu so sánh; đề không bắt buộc đúng ResNet50/ConvNeXt/LR/SVM. [Đề: tr.3-5]

| Sản phẩm hoặc nghĩa vụ | Tiêu chí cần đáp ứng |
|---|---|
| Colab Notebook | Runtime > Run all chạy thành công; tự cài thư viện, tải, giải nén và chuẩn bị dữ liệu, có mô tả |
| Nguồn dữ liệu | Link công khai ngay trong notebook; không phụ thuộc mount kho Drive/Dropbox cá nhân |
| Báo cáo PDF | EDA, phương pháp, thực nghiệm, so sánh, phân tích, bảng phân việc và phần trăm đóng góp |
| Feature files | .npy hoặc .h5, đính kèm hoặc có link tải rõ ràng; được dùng trong bước phân loại |
| Gói nộp | ZIP có notebooks/, modules/, reports/, features/ |
| Quản lý nhóm | Đăng ký tên nhóm và thành viên trên Google Sheet từ LMS |
| Tiến độ | Nộp báo cáo định kỳ theo lịch GV; thu minh chứng họp/làm việc từ đầu |
| Nộp cuối | Drive folder của GV; kiểm hạn và quy cách cụ thể trên LMS |

Tải ZIP bằng link Dropbox công khai là tải HTTP thông thường; điều đề cấm là phụ thuộc việc mount kho cá nhân khi chạy notebook. Nộp bài vào Drive của GV là một bước riêng. [Đề: tr.5-7]

| Tiêu chí chấm | Tỷ trọng |
|---|---:|
| Hoàn thiện pipeline | 40% |
| Chất lượng phân tích và thực nghiệm | 25% |
| Chất lượng báo cáo | 20% |
| Đầy đủ deliverables | 10% |
| Hợp tác nhóm | 5% |

Bonus trong đề: thêm pipeline DL có so sánh **+5%**; GitHub tổ chức tốt, README đủ thông tin **+5%**. Dùng pretrained features đã là phần core của Task 3, không tự động được bonus DL. Cách quy đổi/cộng trần cần GV xác nhận. Đề không đặt ngưỡng accuracy phải đạt. [Đề: tr.6]

## 03. Dữ liệu đã kiểm tra và lý do dùng 24 lớp

Ngày 29/09/2026 đã tải đủ ZIP MalImg, đọc và giải mã toàn bộ PNG, kiểm CRC khi đọc, hash bytes PNG và hash pixel. Kết quả này là bằng chứng sẵn có cho nhóm; A tiếp nhận script/manifest và tái hiện kiểm tra khi dựng pipeline, không cần bắt đầu lại từ việc tìm dataset.

| Chỉ tiêu | Bản gốc đã kiểm |
|---|---:|
| File PNG hợp lệ / family | 9.339 / 25 |
| Ảnh hỏng | 0 |
| Kênh màu | Toàn bộ là grayscale mode L |
| Width: min / median / max | 64 / 256 / 1.024 px |
| Height: min / median / max | 208 / 424 / 5.334 px |
| Pixel khác nhau | 8.019 nhóm |
| Nhóm trùng pixel / số bản sao dư | 67 / 1.320 |
| Nhóm pixel giống nhau nhưng khác nhãn | 0 |

**Yuner.A:** 800 file nhưng chỉ có một ảnh pixel khác nhau. Tách các file đó sang nhiều tập gây rò rỉ ảnh; giữ cùng tập thì không có mẫu khác để đánh giá lớp đó ở các tập còn lại. Quyết định của nhóm là loại toàn bộ family này khỏi benchmark chính, giữ nguyên archive gốc để truy xuất và giải thích quyết định trong báo cáo.

| Family cần chú ý | Số file | Số nhóm pixel khác nhau |
|---|---:|---:|
| Yuner.A - loại khỏi benchmark | 800 | 1 |
| Adialer.C | 122 | 44 |
| Autorun.K | 106 | 16 |
| Fakerean | 381 | 33 |
| Lolyda.AT | 159 | 154 |

Sau khi loại Yuner.A còn **8.539 file, 24 lớp, 8.018 nhóm pixel**, tức còn 521 bản sao dư. Với 24 lớp còn lại, nhóm giữ file nhưng buộc các bản sao cùng partition. Số nhóm pixel khác nhau không chứng minh mẫu malware độc lập về thống kê; near-duplicate và quan hệ biến thể chưa được kiểm tra đầy đủ.

Ảnh khác bytes PNG vẫn có thể cùng pixel vì nén/metadata. Vì vậy khóa nhóm là hash của **mode + width + height + pixel giải mã**, không chỉ tên file hoặc hash bytes PNG. Chi tiết từng file nằm trong MalImg_manifest_verified.csv của gói bàn giao.

## 04. Nguồn tải, tái lập và chính sách chia dữ liệu

**Nguồn đã tải thành công:** [MalImg public ZIP](https://www.dropbox.com/scl/fi/wdb6omeiu2lg796qvt9l7/malimg_dataset.zip?rlkey=63q2xqmtlm66gilf6idd2c9k7&dl=1).

**Kích thước archive:** 1.174.609.734 byte. **SHA-256 bản đã kiểm:**

`9766ae9f1daa520e367fb486ca94728fe1485c0f5cb8314c312d77089a1fe9ec`

Checksum này nhận diện bản nhóm đã kiểm; chưa có checksum do tác giả công bố để xác thực nguồn. Bản ZIP chỉ có các thư mục family, không có split. Bản Figshare đã kiểm ở mức manifest có thêm 924 ảnh lặp và tên nhãn Rbotigen; nhóm thống nhất không thay nguồn mirror tùy người.

**Quy trình khóa dữ liệu do A thực hiện, C kiểm chéo:**

1. Tải, kiểm hash archive, giải nén an toàn, lập danh sách ảnh và label.
2. Kiểm ảnh đọc được; tạo file hash, pixel hash, thống kê trùng/nhãn xung đột.
3. Đánh dấu Yuner.A là excluded, có lý do; không xóa dữ liệu gốc.
4. Tạo label_map chung: 24 tên family còn lại, sắp alphabet, label_id từ 0 đến 23.
5. Chia theo family và nhóm pixel với seed 42; mục tiêu số file khoảng 70% train, 15% val, 15% test. Tất cả file cùng nhóm vào một partition.
6. Kiểm mỗi lớp có mặt trong cả ba tập, không giao hash pixel giữa tập; cố gắng có ít nhất hai nhóm pixel ở val/test mỗi lớp. Báo riêng số file và số nhóm.
7. Chọn một đại diện mỗi nhóm cho val/test bằng đường dẫn nhỏ nhất theo thứ tự từ điển. Lưu cờ eval_representative; không chọn đại diện bằng kết quả model.
8. Xuất manifest, checksum và version v1. C kiểm; cả nhóm dùng đúng phiên bản này.

Tỷ lệ là mục tiêu; không ép đúng từng file bằng cách tách một nhóm trùng. A ghi thuật toán thực tế, tỷ lệ đạt được và lệch từng lớp. **Chưa có số mẫu train/val/test chính thức trong tài liệu này.** Nếu điều kiện tối thiểu không đạt, A/C giải quyết ở mức metadata và khóa lại trước khi chạy model.

Chỉ train được dùng để fit scaler hoặc các bước học từ dữ liệu. Validation dùng chọn cấu hình/checkpoint. Test chỉ mở đánh giá sau khi khóa phương án; không dùng điểm test để chọn split, resize, classifier hoặc epoch. EDA toàn bộ trước split giới hạn ở kiểm chất lượng và mô tả; quyết định tối ưu mô hình dựa trên train/validation.

## 05. Pipeline và quy ước bàn giao giữa các phần

Luồng chung: **URL -> kiểm dữ liệu -> manifest/split -> EDA -> preprocessing -> frozen feature extraction -> lưu/nạp feature -> classifier -> đánh giá -> bảng kết quả và báo cáo.** Colab điều phối toàn bộ các module Python.

Preprocessing: grayscale lặp thành 3 kênh khi backbone yêu cầu; resize theo config; normalization phù hợp pretrained weights. Dùng chung chính sách so sánh, nhưng mỗi weights có thể cần transform riêng. Ghi rõ weights, input size và transform thực dùng. Chưa cần khóa learning rate hoặc batch size ở bản bàn giao này.

Frozen extractor chạy chế độ đánh giá, không cập nhật trọng số và trả một vector mỗi ảnh. Nhãn 24 lớp đến từ classifier của nhóm; không dùng nhãn ImageNet làm đầu ra bài toán. B chọn điểm lấy feature/pooling và ghi kích thước vector thực tế, không đoán.

| Bàn giao | Trường tối thiểu hoặc điều kiện |
|---|---|
| A -> B/C: manifest | sample_id duy nhất, relative_path, label, label_id, width, height, mode, file_sha256, pixel_sha256, included, exclusion_reason, split, eval_representative |
| A -> B/C: label map | JSON 24 tên family và ID; không tự tạo lại theo thứ tự đọc folder |
| B -> C: feature files | X_train/val/test.npy; y_train/val/test.npy; sample_ids_train/val/test.json; metadata.json |
| Metadata feature | Dataset/hash archive, manifest version/hash, weights ID, extractor, transform, feature shape/dtype, thứ tự mẫu, phiên bản thư viện |
| B -> C: prediction | experiment_id, sample_id, true_label, predicted_label; score nếu có và nêu loại score |
| Registry thí nghiệm | ID, config, seed, split version, extractor/weights, classifier, validation score, thời gian, vị trí output, trạng thái |

Điều kiện bắt buộc của bàn giao: **X[i], y[i] và sample_ids[i] luôn nói về cùng một ảnh**. Không shuffle riêng từng mảng. C có quyền từ chối kết quả thiếu version/config hoặc không khớp số mẫu; cần sửa tại nguồn trước khi vào bảng báo cáo.

Feature cache chỉ dùng lại khi manifest, weights và preprocessing khớp metadata. Đổi một thành phần phải tạo cache/version mới. Linear SVM có decision score, không tự gọi đó là xác suất hoặc độ tin cậy đã hiệu chỉnh.

## 06. Câu hỏi nghiên cứu và ma trận thí nghiệm

**Câu hỏi core:** với cùng split và cách đánh giá, pretrained representation và classifier ảnh hưởng ra sao đến chất lượng phân loại 24 family, đặc biệt ở những lớp ít mẫu khác nhau?

Core trả lời: E2 so E3 về classifier trên cùng ResNet50 features; E3 so E4 về pretrained representation khi cùng dùng Linear SVM. E3/E4 có thể khác cả weights, transform và chiều feature, nên kết luận về các cấu hình representation thực tế, không khẳng định chỉ kiến trúc là nguyên nhân.

| ID | Cấu hình | Mục đích | Mức / đầu mối |
|---|---|---|---|
| E2 | Frozen ResNet50 + Logistic Regression | Core classifier comparison | Core / B |
| E3 | Frozen ResNet50 + Linear SVM | Mốc so sánh chung | Core / B |
| E4 | Frozen ConvNeXt-Tiny + Linear SVM | So sánh extractors | Core / B |
| E0 | Width, height, aspect ratio, pixel count + LR | Image-size shortcut audit | Ưu tiên sau core / A, B hỗ trợ |
| E1 | HOG + Linear SVM | Handcrafted baseline | Ưu tiên sau core / A, B hỗ trợ |
| ER | ConvNeXt-Tiny + SVM: direct resize so với resize giữ tỷ lệ + padding | Resize ablation | Sau core / A+B |
| E5a/b | ConvNeXt-Tiny: backbone frozen + neural head được học, rồi partial fine-tuning | DL extension | Bonus / B+C |
| E6 | DINOv2 hoặc representation khác | Mở rộng thêm | Ngoài lịch mặc định |

Mỗi phép so sánh dùng cùng split, cùng tập đánh giá, cùng metric và ngân sách chọn cấu hình được ghi rõ. Chọn một tập hyperparameter nhỏ sau pilot; lưu tất cả trial, lỗi và cảnh báo hội tụ. Không cần chạy hết mọi tổ hợp để hoàn thành core.

**Fine-tuning:** E4 so E5b là so sánh hai pipeline, vì thay cả classifier và cách học representation. Muốn kết luận riêng về tác động cập nhật backbone, dùng thêm E5a: cùng kiến trúc neural head, preprocessing và protocol, nhưng backbone frozen. Không khẳng định giữ cùng backbone đã tự cô lập mọi yếu tố. Checkpoint chọn bằng validation, không bằng test.

Nhóm ưu tiên core và phân tích trước. Tại mốc 25/10, chỉ mở nhánh DL nếu core ổn định và có thời gian kiểm Run all; dừng mở rộng mới sau 08/11. Nếu bỏ một nhánh, bỏ luôn câu hỏi/kết luận tương ứng khỏi abstract và báo cáo. Không hứa so sánh đủ ba mức biểu diễn khi HOG hoặc fine-tuning chưa làm.

## 07. Đánh giá: cách tính và giới hạn kết luận

**Metric chính để chọn cấu hình và báo cáo:** Macro-F1 trên một mẫu đại diện mỗi nhóm pixel trong validation/test. Cách này cho mỗi nhóm ảnh khác pixel một lần đóng góp trước khi lấy trung bình F1 của 24 lớp. C dùng cờ eval_representative trong manifest; mọi model dùng cùng danh sách.

**Metric phụ:** Accuracy; Macro Precision/Recall; Precision/Recall/F1 và support theo lớp; confusion matrix. Báo thêm metric trên toàn bộ file của cùng split để thấy ảnh hưởng của các bản sao. Ghi rõ hai cột là “mỗi nhóm pixel một mẫu” và “toàn bộ file”; không trộn hai cách đếm trong một bảng.

Đây là lựa chọn đánh giá nội bộ dựa trên phát hiện ảnh trùng, không phải metric bắt buộc nguyên văn của đề. Nó không chứng minh đã loại được near-duplicate. Kết quả MalImg 24 lớp theo protocol này không được so trực tiếp như cùng benchmark với paper dùng 25 lớp và random split theo file.

| Phân tích phải có trong core | Sản phẩm |
|---|---|
| So sánh E2/E3/E4 | Một bảng cùng protocol; giải thích được chênh lệch và giới hạn |
| Lớp khó và lớp ít mẫu | Per-class F1 cùng số file/số nhóm thực tế ở tập đánh giá |
| Nhầm lẫn | Confusion matrix và một số ảnh lỗi thuộc các cặp hay nhầm |
| Chi phí tối thiểu | Thời gian extraction, classifier fit, chiều/dung lượng feature; ghi phần cứng, số mẫu và cache state |
| Hạn chế | Few-group classes, near-duplicate chưa kiểm, domain shift, subset 24 lớp, số seed thực chạy |

Nếu đo inference time, tách classifier-only khỏi toàn pipeline gồm preprocessing và extraction; so trên cùng phần cứng hoặc ghi rõ khác biệt. Không cộng thời gian từ môi trường khác nhau rồi gọi là so sánh tốc độ công bằng.

Một fixed split cho kết quả thực nghiệm của split đó; không dùng từ “có ý nghĩa thống kê” nếu chưa đánh giá độ biến thiên/kiểm định phù hợp. Biểu đồ UMAP/t-SNE nếu làm chỉ hỗ trợ quan sát, không thay metric phân loại.

**Dấu hiệu hoàn thành phần đánh giá:** mỗi con số/hình trong report truy về experiment ID và config, không có metric giả, bảng trống được gắn “chưa chạy”, và kết luận chỉ dựa trên kết quả thật.

## 08. Phân công chi tiết: A và B

**A - Dữ liệu, EDA và kiểm soát split. Người review: C.**

| Mã | Việc A sở hữu | Đầu ra / tiêu chí hoàn thành |
|---|---|---|
| A01 | Kiểm LMS, đăng ký và xin GV duyệt MalImg 24 lớp | Ghi câu trả lời, lịch tiến độ và nơi nộp; không tự ghi “đã duyệt” |
| A02 | Tái hiện kiểm kê từ URL/ZIP | Dataset card; ảnh đọc được; hash và counts khớp hoặc giải thích khác biệt |
| A03 | Áp dụng chính sách loại Yuner.A và nhóm trùng | included/exclusion_reason rõ; raw giữ nguyên |
| A04 | Tạo và kiểm split v1, label map | 24 lớp đủ ba tập; không giao pixel hash; bảng file/group theo split |
| A05 | EDA có nhận xét | Phân bố lớp, width/height/channels, ảnh mẫu, duplicate profile, quyết định xử lý |
| A06 | Preprocessing và nhánh phụ | Interface transform cấu hình được; sau core làm E0/HOG và hỗ trợ resize |
| A07 | Viết báo cáo phần dữ liệu | Nguồn, EDA, lý do 24 lớp, split và giới hạn; dẫn số liệu đúng manifest |

**B - Đặc trưng và ML truyền thống. Người review: A.**

| Mã | Việc B sở hữu | Đầu ra / tiêu chí hoàn thành |
|---|---|---|
| B01 | Interface extractor và feature store | Thử trên vài mẫu; shape đúng, frozen đúng; sample IDs khớp |
| B02 | ResNet50 và ConvNeXt-Tiny | Weights/transform rõ, extraction theo batch, kết quả có metadata |
| B03 | Lưu và nạp lại feature | X/y/IDs đủ ba tập; nạp lại đúng nội dung/thứ tự; cache không dùng sai config |
| B04 | LR/SVM và chạy E2/E3/E4 | Cấu hình, validation selection, thời gian và prediction files đầy đủ |
| B05 | Hỗ trợ các phép so sánh | Dùng cùng code classifier cho HOG/shortcut/resize khi phù hợp |
| B06 | Nhánh DL cùng C nếu mở | B phụ trách model và fine-tuning; C phụ trách đánh giá/integration |
| B07 | Viết báo cáo phương pháp | Pipeline, representation, classifier, config; nêu rõ frozen so với fine-tuned |

A bàn giao dữ liệu cho B qua manifest có version. Trong lúc chờ khóa split, B phát triển trên tập mẫu thử và gắn nhãn smoke test; kết quả đó không đi vào bảng thực nghiệm chính thức.

## 09. Phân công C và cách phối hợp nhóm

**C - Đánh giá, Colab và tích hợp. Người review: B.**

| Mã | Việc C sở hữu | Đầu ra / tiêu chí hoàn thành |
|---|---|---|
| C01 | Khung main_colab và môi trường | Tự cài, tải, giải nén; gọi module bằng cấu hình; không đường dẫn cá nhân |
| C02 | Hàm đánh giá chung | Kiểm bằng nhãn nhỏ có đáp án biết trước; class order cố định; hỗ trợ representative/full-file |
| C03 | Registry và tổng hợp kết quả | Mỗi run có ID/version/config; tự tạo bảng/hình từ result files |
| C04 | Phân tích lỗi với A/B | Confusion matrix, per-class support/F1, mẫu lỗi và giới hạn |
| C05 | Review split và feature alignment | Chứng minh không overlap; từ ID tra về ảnh, nhãn, feature và prediction |
| C06 | Kiểm Run all, đóng gói | Chạy runtime sạch; link/artifact hoạt động; ZIP đúng cấu trúc |
| C07 | README và báo cáo đánh giá | C viết protocol/tích hợp; mỗi thành viên viết phần của mình |

**Quy tắc làm việc:**

- Một đầu việc có một người sở hữu và một người review; người review kiểm tiêu chí nghiệm thu rồi mới đánh dấu xong.
- Dùng Git để quản lý code; tài liệu ghi rõ branch/commit của kết quả. Không commit raw dataset, feature cache lớn hoặc credential.
- Mỗi thành viên viết dần phần báo cáo gắn với đầu ra của mình; C tổng hợp bố cục, không viết thay toàn bộ nhóm.
- Họp ngắn mỗi tuần, chốt lịch cụ thể ở buổi đầu. Biên bản ghi việc xong, việc tuần tới, người làm, hạn và vướng mắc; lưu làm minh chứng.
- Chưa biết xử lý lỗi hoặc phát hiện sai split: ghi issue và báo cả nhóm; không sửa âm thầm manifest hay label map.
- DL bonus do B và C phối hợp; A hỗ trợ dữ liệu và kiểm tra. Không giao cả DL, code đánh giá, notebook và báo cáo cuối cho một người.

Tỷ lệ đóng góp cuối dựa trên phần việc và minh chứng thực tế, chưa điền 33,3% mỗi người. Đề ghi điểm cá nhân bằng điểm nhóm nhân Contribution Percentage nhưng chưa giải thích chuẩn hóa; A hỏi GV trước khi điền bảng chính thức.

**Điểm ghép đầu tiên:** notebook tải được dữ liệu, lấy mẫu qua manifest, tạo/lưu/nạp feature đúng ảnh-nhãn, gọi được hàm đánh giá. Mục tiêu này cần hoàn thành sớm để phát hiện bất đồng interface trước khi chạy toàn bộ.

## 10. Lịch và thứ tự ưu tiên tới 30/11

Đây là lịch nội bộ tính từ 30/09/2026. Mốc LMS do GV công bố có ưu tiên; A bổ sung ngay khi tra được. Hạn 30/11 dựa trên thông tin nhóm cung cấp.

| Thời gian | Đầu ra cần có | Người chính / điểm kiểm |
|---|---|---|
| 30/09-04/10 | Điền tên A/B/C; hỏi GV; repo/Colab khung; data card; demo feature nhỏ | Cả nhóm; ghép được đường chạy đầu tiên |
| 05/10-11/10 | EDA bản 1; manifest/split v1; label map và transform chung | A làm, C review; khóa dữ liệu khi phạm vi GV được xác nhận |
| 12/10-18/10 | Hai bộ frozen features lưu/nạp được; đánh giá chung hoạt động | B làm, A/C kiểm alignment |
| 19/10-25/10 | E2/E3/E4 trên validation; registry/bảng kết quả; bản nháp phương pháp | B+C; họp quyết định mở bonus hay củng cố core |
| 26/10-01/11 | Error/cost analysis ban đầu; E0/E1/resize theo sức nhóm | A+B+C; báo cáo viết song song |
| 02/11-08/11 | Nhánh DL nếu đã mở; khóa danh sách thí nghiệm và cấu hình chọn bằng val | B+C; dừng thêm nhánh mới sau 08/11 |
| 09/11-15/11 | Final test theo protocol đã khóa; draft report hoàn chỉnh | C tổng hợp; cả nhóm giải thích kết quả |
| 16/11-22/11 | Run all runtime sạch, tái lập các bảng kết quả, xử lý lỗi và link | C điều phối; một thành viên khác chạy kiểm |
| 23/11-27/11 | Review chéo PDF/README, bảng đóng góp, evidence và ZIP | Cả nhóm; khóa bản nộp ứng viên |
| 28/11-29/11 | Nộp sớm, tải/mở lại kiểm tra, giữ xác nhận nộp | Người nộp do nhóm chỉ định |
| 30/11 | Dự phòng lỗi nộp bài | Không để công việc chính đến ngày này |

**Ưu tiên nếu chậm:** bảo đảm pipeline core, chất lượng dữ liệu, đánh giá và Run all; dừng E6, rồi DL nếu chưa ổn định; chỉ giữ các nhánh phụ có đủ kết quả để phân tích. Việc cắt scope phải ghi lại, sửa câu hỏi nghiên cứu và tiêu đề nếu cần.

**Buổi họp đầu tiên nên chốt:** ai là A/B/C, ai đại diện hỏi GV, ai là người nộp; nơi quản lý code/tài liệu; lịch họp; ngày kiểm đầu ra đầu tiên. Có thể phân công ba vai ngay trong khi chờ GV, nhưng chưa ghi kết quả benchmark như đã được phê duyệt.

## 11. Cấu trúc code, notebook và báo cáo

Thư mục tối thiểu đề yêu cầu là notebooks/, modules/, reports/, features/. Các thư mục còn lại dưới đây phục vụ việc phối hợp:

```text
malimg-family-classification/
  README.md, requirements.txt
  configs/       cấu hình thí nghiệm
  notebooks/     main_colab.ipynb
  modules/       data.py, validation.py, eda.py
                 preprocessing.py, features.py, feature_io.py
                 classifiers.py, evaluation.py, visualization.py
                 dl_extension.py (nếu làm)
  metadata/      dataset_manifest, split_manifest, label_map
  features/      mỗi extractor có X, y, sample IDs và metadata
  results/       experiments, metrics, predictions, figures
  reports/       final_report.pdf, progress/, contributions
```

**Notebook chính:** thông tin môn/nhóm; hướng dẫn chạy; cài môi trường; config/seed; tải/giải nén; kiểm dữ liệu; manifest/split; EDA; preprocessing/extraction; lưu rồi nạp feature; classifier; đánh giá/so sánh; mở rộng nếu bật; tổng hợp output. Các notebook khám phá khác chỉ là hỗ trợ.

Core phải chạy được bằng Run all từ runtime sạch với nguồn công khai. Runtime > Run all chỉ chạy subset demo không tự đáp ứng bài chính. Nếu cần tải feature cache công khai hoặc có full/demo mode vì thời gian, C ghi rõ chế độ mặc định và hỏi GV; người chạy phải biết kết quả nào được tính lại và kết quả nào tải từ artifact.

**Khung báo cáo PDF:**

1. Thông tin môn, GV, nhóm; abstract phản ánh việc thực sự đã làm.
2. Bài toán, mục tiêu, phạm vi 24 lớp và câu hỏi nghiên cứu.
3. Kiến thức liên quan và nguồn tham khảo có trích dẫn.
4. Dataset/EDA: nguồn, kiểm kê, duplicate audit, quyết định Yuner.A, split.
5. Phương pháp: preprocessing, extractor, feature persistence, classifier.
6. Thiết kế thực nghiệm: cấu hình, metric/đơn vị đánh giá, val/test protocol.
7. Kết quả: bảng core, so sánh, per-class metrics, chi phí; bonus nếu có.
8. Phân tích lỗi, giới hạn và kết luận trả lời đúng câu hỏi đã thực hiện.
9. Phân công/đóng góp, minh chứng nhóm, tham khảo và hướng dẫn tái chạy.

Đề không ấn định số trang. Chất lượng bảng, hình và giải thích có trọng số cao; không kéo dài phần lý thuyết để bù thiếu kết quả.

## 12. Nghiệm thu, thông tin còn thiếu và nguồn bàn giao

**Checklist trước khi chốt bản nộp:**

- [ ] GV duyệt dataset/tập con 24 lớp, không trùng nhóm khác; đăng ký nhóm đầy đủ.
- [ ] Có EDA và giải thích chính sách dữ liệu; các mốc tiến độ LMS được nộp.
- [ ] Split đủ lớp, không giao pixel hash; version/label map dùng chung.
- [ ] Hai pretrained extractors, feature .npy/.h5, nạp lại khớp sample IDs và nhãn.
- [ ] E2/E3/E4 có config, validation selection, final metrics và phân tích thật.
- [ ] Báo cáo phân biệt metric theo nhóm pixel với toàn bộ file; không hứa thống kê vượt bằng chứng.
- [ ] Colab tự cài/tải/giải nén/chuẩn bị, Run all từ runtime sạch thành công.
- [ ] PDF, features hoặc link, ZIP đúng thư mục, mọi link cho người chấm truy cập được.
- [ ] Bảng đóng góp theo cách GV xác nhận; có minh chứng cộng tác.
- [ ] Nếu xin bonus: có DL comparison thực hiện đầy đủ; GitHub/README đáp ứng yêu cầu.

**Thông tin còn thiếu, giao người phụ trách để không chặn cả nhóm:**

| Thông tin | Ai bổ sung / khi nào |
|---|---|
| Họ tên, MSSV, email và tên nhóm | Cả nhóm, buổi họp đầu; hiện dùng A/B/C |
| Mã môn, GV, lớp học phần | A từ LMS trước khi hoàn thiện README |
| Dataset/tập con 24 lớp và đăng ký không trùng | A hỏi GV ngay; lưu phản hồi |
| Mốc báo cáo tiến độ, nơi nộp, giới hạn Run all/dung lượng | A tra LMS/GV; C áp dụng vào notebook |
| Cách tính contribution, quy đổi/trần bonus | A hỏi GV trước báo cáo cuối |

**Tài liệu kèm gói bàn giao:** bản này ở PDF và Markdown; báo cáo kiểm kê ngày 29/09; manifest 9.339 ảnh gốc; đề MLAssignment_ELv2.pdf để đối chiếu; script tải và kiểm kê đã dùng để tái hiện số liệu. Script kiểm kê là bằng chứng hỗ trợ, chưa phải pipeline dự án đã tích hợp. Không có raw dataset trong gói gửi nhóm.

**Nguồn:** đề MLAssignment_ELv2.pdf tr.4-5 (Task 3), tr.5-6 (deliverables/nhóm), tr.6-7 (rubric/tiến độ/nộp); kiểm kê bản tải 29/09/2026; [Nataraj et al., Malware Images: Visualization and Automatic Classification, VizSec 2011](https://doi.org/10.1145/2016904.2016908).

README trong ZIP yêu cầu trích dẫn bài báo và ghi “This dataset is not be distributed.” Nhóm dùng URL hiện hữu để tải, không đưa ảnh/ZIP gốc lên GitHub. Điều khoản đối với chia sẻ feature embeddings chưa được làm rõ; hỏi cách nộp/chia sẻ phù hợp trước khi công bố công khai. GitHub bonus yêu cầu README đủ tên/mã môn, học kỳ/năm học, GV, họ tên/MSSV/email thành viên, mục tiêu, cách chạy/cài/tải, cấu trúc, link report/Colab; link repo đặt trong notebook và report.

**Thuật ngữ nhanh:** EDA = khám phá dữ liệu; frozen = giữ nguyên trọng số backbone; feature/embedding = vector biểu diễn ảnh; classifier = bộ dự đoán family từ vector; fine-tuning = cập nhật pretrained model theo bài toán; leakage = thông tin tập đánh giá lọt vào quá trình học/chọn model; ablation = thay một thành phần để đo ảnh hưởng; manifest = bảng ánh xạ mẫu, nhãn và split dùng chung.
