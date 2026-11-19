# Phân vùng tướng địa chấn trên ảnh lát cắt

## Bản nâng cấp

Giao diện hiện gồm bốn khu vực: **Phân tướng một ảnh**, **Xử lý thư mục**, **Kiểm tra dữ liệu** và **Thí nghiệm**. Có bản đồ confidence/entropy, ngưỡng pixel cần rà soát, sửa polygon ngoại tuyến và đối chiếu số liệu giữa các lần huấn luyện. Chi tiết lựa chọn từ dự án tương tự nằm trong [ghi chú nghiên cứu](docs/RESEARCH.md); danh sách thay đổi ở [CHANGELOG.md](CHANGELOG.md).

### Quy trình khi nhận bộ ảnh chuyên gia

Kiểm tra dữ liệu trước:

```powershell
python -m facies.audit --manifest data/dataset.csv --classes classes.json --out runs/audit
```

Chạy mô hình nền và cấu hình attention để bắt đầu thí nghiệm. Hai cấu hình này khác cả cách lấy mẫu/loss; để chỉ so sánh kiến trúc, dùng cùng một tệp config và đổi riêng --arch.

```powershell
python -m facies.train --config configs/baseline.json --manifest data/dataset.csv --out runs/baseline
python -m facies.train --config configs/attention.json --manifest data/dataset.csv --out runs/attention
```

Hoặc dùng cấu hình CPU ngắn để kiểm tra hệ thống, không dùng điểm của lượt này để kết luận hiệu quả:

```powershell
python -m facies.train --config configs/cpu-smoke.json --manifest data/dataset.csv --out runs/check
```

Lưu và tiếp tục huấn luyện với cùng cấu hình. epochs là tổng số epoch đích, không phải số epoch cộng thêm:

```powershell
python -m facies.train --config configs/attention.json --manifest data/dataset.csv --out runs/attention --resume runs/attention/last.pt --epochs 100
```

last.pt lưu optimizer, scheduler và trạng thái ngẫu nhiên; best.pt lưu mô hình được chọn trên validation. Thay đổi ảnh, nhãn hoặc cách chia làm thay đổi fingerprint và bị chặn khi resume. Chọn thư mục mới cho thí nghiệm mới; không ghi đè một lượt đã có kết quả.

### Đánh giá và xử lý hàng loạt

Sau khi đã chọn cấu hình trên validation, dùng tập test được chuyên gia giữ riêng:

```powershell
python -m facies.evaluate --model runs/attention/best.pt --manifest data/dataset.csv --split test --out runs/evaluation --previews 3
python -m facies.batch --model runs/attention/best.pt --input data/new-images --out runs/predictions --confidence-threshold 0.65 --tta
```

CLI evaluate cần split rõ ràng trên tất cả dòng CSV. Nó không dùng train thay cho test khi thiếu nhãn test. metrics.csv ghi điểm theo lớp, sections.csv ghi từng lớp trên từng lát cắt, metrics.json lưu tổng hợp. Boundary F1 dùng sai số pixel mặc định 2, chỉ đo biên bên trong vùng đã có nhãn và tránh mép vùng chưa biết. Nó là chỉ số hình học ảnh, không phải độ chính xác địa chất theo mét.

Batch lưu index.csv/index.json và kết quả riêng từng ảnh, xếp ưu tiên rà soát bằng entropy/tỷ lệ pixel dưới ngưỡng. Các ảnh lỗi được báo riêng. Nhánh TTA lật ngang là tùy chọn; giữ tắt để xử lý nhanh hơn khi cần.

| Kết quả bổ sung | Nội dung |
| --- | --- |
| raw_mask.png | Mã lớp trước khi áp dụng ngưỡng, giữ nguyên độ phân giải. |
| confidence.png | PNG 16-bit: softmax lớn nhất × 65535. |
| uncertainty.png | PNG 16-bit: entropy chuẩn hóa × 65535. |
| review.png | 255 tại pixel cần rà soát, 0 ở phần còn lại. |
| labelme.json | Gợi ý polygon có cờ máy dự đoán, cần chuyên gia duyệt. |
| source.png | Ảnh đầu vào để ghép với JSON. |
| trace.html | Ảnh và polygon tích hợp trong một trang hiệu chỉnh ngoại tuyến. |

Nếu có ngưỡng rà soát, mask.png dùng ID 255 ở phần dưới ngưỡng; raw_mask.png vẫn giữ mã dự đoán. Không suy ra pixel dưới ngưỡng là nền. Với polygon có lỗ hoặc không tái tạo mask chính xác, công cụ giữ nguyên mask và liệt kê vùng bỏ qua trong JSON/trang hiệu chỉnh.

Mở trace.html của từng kết quả để kéo đỉnh, đổi nhãn, vẽ bổ sung và xác nhận vùng đã duyệt. Lưu JSON hiệu chỉnh và source.png cùng thư mục. Polygon dự đoán chưa được duyệt sẽ được bỏ qua khi nhập lại thành nhãn huấn luyện; đây không phải nhãn chuyên gia tự động. Sau khi chỉnh sửa, tạo cặp source.png/JSON trong manifest và chạy audit lại.

Trong mục **Thí nghiệm**, chọn các metrics.json cần đối chiếu. Báo cáo khác fingerprint dữ liệu được cảnh báo. Chỉ số không có trong báo cáo cũ giữ trống; không được điền thành 0. Chỉ số dùng chọn checkpoint mới là mean_iou_facies_union, tính cả lớp chỉ xuất hiện trong dự đoán; mean_iou_facies cũ vẫn được lưu để đọc báo cáo trước.

Đưa một ảnh lát cắt vào mô hình để nhận ảnh có màu, đường viền và tên tướng. Mô hình gán mã lớp cho từng pixel, rồi vẽ viền theo hình dạng dự đoán của vùng. Mô hình dùng kiến trúc U-Net và xử lý ảnh lớn theo các ô chồng lấp để giữ nguyên kích thước đầu ra. [Bài báo U-Net](https://arxiv.org/abs/1505.04597).

Thư mục hiện có mô hình và ảnh mô phỏng để chạy thử. Cần bộ lát cắt thực đã được chuyên gia gán nhãn để huấn luyện và đo chất lượng trên dữ liệu thực.

## Chạy mẫu

Mở PowerShell tại thư mục dự án và chạy hai lệnh sau.

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 127.0.0.1
```

Trong giao diện, chọn **Mô hình mô phỏng**, giữ tùy chọn dùng lát cắt có sẵn rồi bấm **Chạy phân tướng**. Có thể đối chiếu ảnh trước và sau và tải kết quả. Lệnh dưới đây chạy cùng mẫu.

```powershell
python -m facies.predict --model demo/model.pt --image demo/sample.png --out runs/demo
```

Báo cáo đánh giá của mẫu chỉ đo khả năng phân vùng trên ảnh mô phỏng. Xem ảnh khoanh vùng trong ứng dụng hoặc tạo bằng lệnh trên; các ảnh kết quả sinh tự động không được đưa vào Git. [Báo cáo mẫu](demo/metrics.json).

## Khi chuyên gia giao ảnh đã khoanh và ghi tên

Với khoảng 300 lát cắt, nên thử quy cách trên vài ảnh trước khi xuất cả bộ. Mỗi lát cắt nên có hai bản PNG cùng tên, cùng kích thước và cùng khung tọa độ. Đặt bản có viền và chữ trong `data/annotated/`, bản gốc trong `data/clean/`. Giữ cách hiển thị biên độ, tỷ lệ và hướng trục nhất quán giữa ảnh huấn luyện và ảnh cần dự đoán.

Nếu các vùng có **viền kín với màu RGB riêng theo từng tướng**, công cụ bên dưới có thể lấy vùng bên trong viền làm mask. Dùng PNG hoặc TIFF và giữ nguyên màu viền theo [bảng lớp](classes.json). Đặt tên tướng bên trong vùng, ưu tiên tránh đè lên viền. Tạo `data/groups.csv` từ [CSV nhóm mẫu](data/groups.example.csv) để mỗi lát cắt được gắn với tuyến hoặc khối khảo sát.

Chữ nên cùng màu với viền tướng tương ứng và nằm gọn trong vùng. Nếu chữ có màu khác viền, khai báo thêm, ví dụ `--text-color 255,255,255` cho chữ trắng khi có ảnh sạch. Công cụ chỉ lấy lớp từ màu viền; chưa đọc chữ bằng OCR.

```powershell
python -m facies.import_outlines --annotated data/annotated --clean data/clean --classes classes.json --groups data/groups.csv --out data/imported
```

Xem `report.json` và ảnh kiểm tra trong thư mục đầu ra trước khi huấn luyện với `--manifest data/imported/dataset.csv`. Quy trình tự động dựa vào màu viền được khai báo. Với viền dùng chung màu, vùng bị hở hoặc tướng chỉ được phân biệt bằng tên in trên ảnh, mở [công cụ vẽ lại vùng](trace.html) trong trình duyệt. Đọc tên trên ảnh tham chiếu, chọn lớp, bấm các đỉnh theo viền rồi xuất JSON LabelMe.

Nếu chỉ còn ảnh đã khoanh, tùy chọn `--inpaint` tạo một bản đầu vào gần đúng từ các pixel xung quanh phần đánh dấu. Tùy chọn này thay cho `--clean` và cần kiểm tra ảnh sau khi xử lý, nhất là vùng có chữ trắng hoặc đen trùng với màu tín hiệu. Phần phản xạ bị che không thể được phục hồi chính xác từ ảnh đã xuất. Dùng một nhóm ảnh sạch độc lập để kiểm tra chất lượng trước khi sử dụng kết quả trên dữ liệu thực.

## Chuẩn bị dữ liệu khi đã có mask hoặc JSON

Mỗi lát cắt cần một **ảnh gốc chưa khoanh vùng** và **tệp nhãn tương ứng**. Đặt ảnh gốc và nhãn về cùng mặt cắt, cùng phạm vi và cùng lưới pixel. Nếu cắt trục, lề hoặc khung tên, áp dụng đúng phần cắt cho cả ảnh lẫn nhãn.

| Kiểu nhãn | Cách chuẩn bị |
| --- | --- |
| Mask PNG hoặc TIFF theo ID | Giữ chiều rộng và chiều cao bằng ảnh gốc. Mỗi pixel chứa ID lớp. Mask thang xám dùng giá trị pixel. Mask có bảng màu dùng chỉ số màu làm ID. |
| Mask màu RGB hoặc RGBA | Mỗi vùng dùng đúng bộ màu RGB trong bảng lớp. Tắt làm mịn viền khi xuất. Với RGBA, alpha 0 là pixel bỏ qua và alpha 255 là pixel có nhãn. |
| LabelMe JSON | Vẽ `polygon` hoặc `rectangle` trên ảnh gốc bằng tên hoặc bí danh trong bảng lớp. Pixel ngoài vùng vẽ được bỏ qua. Nếu các vùng chồng nhau, vùng đứng sau trong JSON được dùng. Có thể đặt nhãn `__ignore__` cho vùng cần bỏ qua. |

ID `0` dành cho nền đã xác nhận, ví dụ khoảng ngoài vùng tướng chuyên gia đã xem và xác định là nền. ID `255` dành cho vùng chưa gán nhãn hoặc còn nghi ngờ. Những pixel `255` được bỏ qua khi huấn luyện và tính điểm. Nếu dùng LabelMe, chỉ vẽ nhãn `Nền` ở vùng nền đã xác nhận.

| ID | Tướng | RGB |
| ---: | --- | --- |
| 0 | Nền | `0, 0, 0` |
| 1 | Hỗn loạn | `238, 101, 95` |
| 2 | Song song | `76, 197, 250` |
| 3 | Á song song | `133, 204, 114` |
| 4 | Gò đồi | `248, 187, 79` |
| 5 | Phân kì | `220, 144, 251` |
| 6 | Sigma | `73, 221, 191` |

Sửa `classes.json` cho phù hợp với cách phân tướng của chuyên gia. Giữ ID liên tiếp từ `0`, dùng tên và màu riêng cho mỗi lớp, rồi đồng bộ tệp nhãn với bảng ID hoặc tên đã chọn. [Bảng lớp](classes.json).

Sao chép tệp CSV mẫu thành `data/dataset.csv` và thay các đường dẫn mẫu. Mỗi dòng có `image`, `mask`, `group` và tùy chọn `split`. Đường dẫn tương đối được tính từ thư mục chứa CSV. Chẳng hạn, `images/line-a.png` trong `data/dataset.csv` trỏ đến `data/images/line-a.png`. [Tệp CSV mẫu](data/dataset.example.csv).

- Đặt lát cắt gần nhau hoặc chồng lấp vào cùng `group`, theo tuyến hoặc khối khảo sát. Các bản cắt hoặc phiên bản xuất từ cùng một lát cắt cũng thuộc cùng nhóm.
- Để chương trình tự chia theo nhóm, bỏ cột `split` hoặc để trống toàn bộ cột. Cần ít nhất hai nhóm và mặc định khoảng 20% số nhóm được dành cho `val`.
- Để chủ động đánh giá, điền `train`, `val` hoặc `test` cho mọi dòng, mỗi nhóm chỉ thuộc một tập. Dành `val` để chọn mô hình và giữ một tuyến hoặc khối độc lập ở `test` để đo khả năng áp dụng trên dữ liệu mới.

## Huấn luyện và đánh giá

```powershell
python -m facies.train --manifest data/dataset.csv --classes classes.json --out runs/facies
```

Chương trình tự chọn CUDA nếu PyTorch nhận GPU. Lệnh dưới đây chạy thử trên CPU để kiểm tra dữ liệu và cấu hình.

```powershell
python -m facies.train --manifest data/dataset.csv --classes classes.json --out runs/check --device cpu --epochs 2 --tile-size 128 --batch-size 2 --patches-per-image 4 --width 12
```

Lượt chạy ngắn giúp phát hiện lỗi đọc dữ liệu và xem thời gian xử lý. Với lượt huấn luyện chính, chọn kích thước ô đủ rộng để thấy kiểu phản xạ và hình dạng của từng tướng. Giảm `--batch-size` trước nếu thiếu bộ nhớ.

Chương trình lưu **`runs/facies/best.pt`** tại epoch có mean_iou_facies_union cao nhất trên tập `val`. Chỉ số này tính các tướng có pixel chuyên gia hoặc pixel được dự đoán, để phạt cả lớp dự đoán thừa. Báo cáo **`runs/facies/metrics.json`** ghi số pixel có nhãn, IoU/Dice/precision/recall/F1 theo lớp và ma trận nhầm lẫn. IoU và Dice đo mức chồng lấp giữa dự đoán và vùng chuyên gia vẽ.

Khi xem báo cáo, đối chiếu IoU và Dice với `support_pixels` của từng tướng. Lớp có `support_pixels` bằng 0 trong một tập chưa có mẫu dương để đánh giá. Mô hình chỉ dự đoán các lớp đã xuất hiện trong tập train. Kiểm tra số lượng lát cắt có từng tướng, ranh giới vùng và các cặp dễ nhầm như Song song và Á song song. Dùng kết quả từ nhóm `test` độc lập để kiểm tra khả năng áp dụng sang lát cắt mới.

## Phân tướng ảnh mới

Chọn **Mô hình đã huấn luyện** trong giao diện, trỏ tới `runs/facies/best.pt`, tải ảnh gốc và bấm **Chạy phân tướng**. Lệnh dưới đây xử lý một ảnh gốc.

```powershell
python -m facies.predict --model runs/facies/best.pt --image data/images/line-a.png --out runs/prediction
```

| Tệp | Nội dung |
| --- | --- |
| `overlay.png` | Ảnh gốc có màu, đường viền và chữ tên tướng. |
| `mask.png` | PNG một kênh chứa mã tướng của từng pixel. |
| `regions.json` | Tên tướng, diện tích, hộp bao theo tọa độ pixel và điểm dự đoán trung bình của các vùng được khoanh. |

`--min-region-area` đặt diện tích nhỏ nhất của một vùng được khoanh và ghi tên, mặc định 64 pixel. Mask PNG vẫn lưu toàn bộ dự đoán. Nếu ảnh lớn khiến suy luận thiếu bộ nhớ, thử `--batch-size 1` trước và dùng kích thước ô lưu trong mô hình khi có thể.

Điểm dự đoán trong JSON là softmax trung bình trên vùng, chưa được hiệu chỉnh thành xác suất đúng. Với khoảng 300 ảnh, phân bố đủ ví dụ của từng tướng và tính độc lập giữa các tuyến quan trọng hơn việc chia cố định theo số ảnh.

Chạy kiểm tra pipeline bằng `python -m unittest discover -s tests -v`. Bản mẫu đã được chạy từ huấn luyện đến xuất ảnh trên CPU.
