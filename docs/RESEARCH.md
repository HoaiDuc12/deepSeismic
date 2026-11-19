# Cơ sở nâng cấp và đối chiếu dự án

Ngày rà soát: 19/11/2026. Các nguồn dưới đây được đọc để chọn phương pháp; không sao chép toàn bộ mã hay tải trọng số của họ vào sản phẩm.

Đợt khảo sát bổ sung đã tìm nguồn gần hơn về hình thái và nhãn tương tác. Xem [danh mục nguồn và mức độ phù hợp](REFERENCES.md), đặc biệt cigFacies/cigFaciesNet và các giới hạn ánh xạ sang sáu tướng của dự án.

## Dự án có cùng bài toán

| Nguồn gốc | Điều tham khảo | Áp dụng trong sản phẩm |
| --- | --- | --- |
| [Microsoft DeepSeismic](https://github.com/microsoft/seismic-deeplearning) | Tách pipeline dữ liệu, mô hình và thí nghiệm; hỗ trợ nhiều kiến trúc phân đoạn. Repo đã được lưu trữ, chỉ dùng làm tài liệu phương pháp. | Bộ tạo mô hình dùng chung, cấu hình JSON, fingerprint dữ liệu, báo cáo và checkpoint tiếp tục huấn luyện. |
| [Alaudah và cộng sự — Facies Classification Benchmark](https://github.com/yalaudah/facies_classification_benchmark) | Có mô hình theo patch/lát cắt và yêu cầu giữ test để đánh giá sau khi chọn mô hình. | Đánh giá toàn lát cắt, split theo tuyến/khối, kiểm tra ảnh trùng giữa tập; CLI evaluate chỉ nhận val/test rõ ràng. |
| [Seismic Segmentation Methods Benchmark](https://github.com/gabriel-cangucu/seismic-segmentation-benchmark) | So sánh SegNet/U-Net/DeconvNet, chia các đoạn lát cắt liên tiếp, có trọng số lớp, early stopping và resume. | Giữ U-Net làm đối chứng, kiến trúc bổ sung có thể hoán đổi, weighted loss, ưu tiên tướng ít pixel, scheduler và dừng sớm. |

Các repo dùng dữ liệu khảo sát và quy ước nhãn riêng. Bộ nhãn hình thái Hỗn loạn/Song song/Á song song/Gò đồi/Phân kì/Sigma của dự án này vẫn do chuyên gia xác nhận trong classes.json. Không tự lấy số đo hay trọng số của dự án khác làm bằng chứng cho bộ ảnh đang chờ cung cấp.

## Kiến trúc và huấn luyện

- [U-Net — Ronneberger và cộng sự](https://arxiv.org/abs/1505.04597): mô hình nền encoder–decoder với kết nối bỏ qua, giữ làm đối chứng.
- [Attention U-Net — Oktay và cộng sự](https://arxiv.org/abs/1804.03999): tham khảo ý tưởng gate không gian trên đặc trưng encoder. Bài gốc đánh giá trên ảnh y khoa; không suy diễn mức tăng điểm sang địa chấn.
- [Residual learning — He và cộng sự](https://arxiv.org/abs/1512.03385): tham khảo kết nối cộng residual. Kiến trúc attention_unet tại đây kết hợp residual, gate và context giãn nhiều mức; đây là biến thể cục bộ, không phải bản tái tạo nguyên gốc của một paper.
- [PyTorch 2.6 AMP](https://docs.pytorch.org/docs/2.6/amp.html): dùng autocast cùng GradScaler khi chạy CUDA; máy hiện tại kiểm tra trên CPU, nhánh CUDA chưa được đo trên phần cứng GPU.

Loss là weighted cross entropy + Dice. Tùy chọn boundary_weight tăng trọng số CE trong lân cận ranh giới có nhãn; nó không phải bản cài đặt Distance Transform Loss hay Boundary Loss của một paper khác. Pixel 255 bị loại khỏi loss và chỉ số. Chỉ lật ngang và thay gain/độ sáng ở mức nhỏ trong huấn luyện; không quay hoặc lật chiều sâu.

## Độ bất định và vòng hiệu chỉnh

Độ tin cậy là xác suất softmax lớn nhất; entropy chuẩn hóa là mức phân tán giữa các lớp đã học. Cả hai chưa được hiệu chuẩn thành xác suất địa chất đúng. Ngưỡng rà soát chỉ quyết định pixel nào mang mã 255; raw_mask vẫn giữ toàn bộ dự đoán. TTA lật ngang là tùy chọn và tăng số lần forward.

Danh sách ưu tiên xem lại dùng công thức công khai 0.6 × tỷ lệ pixel dưới ngưỡng + 0.4 × entropy trung bình. Đây là heuristic hỗ trợ phân bổ công sức, chưa phải một thí nghiệm active learning được chứng minh. Polygon máy chưa được chuyên gia xác nhận không được dùng làm nhãn chuyên gia khi nhập lại.

## Cách xác nhận nâng cấp trên dữ liệu thật

1. Xuất ảnh sạch và ảnh diễn giải cùng lưới pixel; ưu tiên nhãn mask/polygon gốc. Nếu chỉ có ảnh khoanh, kiểm tra từng bản chuyển đổi/nội suy.
2. Nhóm các lát cắt trùng, gần hoặc giao nhau theo khối/tuyến trước khi chia. Hash chỉ phát hiện bản sao chính xác, không chứng minh độc lập không gian.
3. Chạy audit; bảo đảm từng tướng có ví dụ đủ và đại diện ở train/val/test. Với khoảng 300 ảnh, số nhóm độc lập và độ phủ nhãn phải được xem cùng số lượng ảnh.
4. Chạy U-Net và attention_unet với cùng dữ liệu, seed, số bước, kích thước ô và loss để tách ảnh hưởng kiến trúc. Để phân tích tác dụng lấy mẫu/loss, thay từng lựa chọn riêng ở các thí nghiệm sau.
5. Chọn cấu hình trên validation; giữ test cuối cùng, báo cáo IoU/Dice/precision/recall/F1 từng tướng, Boundary F1 và ảnh lỗi. Chạy nhiều seed trước khi tuyên bố một mô hình tốt hơn.

Thí nghiệm demo chỉ dùng ảnh mô phỏng tạo bằng công thức. Chúng không kiểm tra nhiễu và thiết bị thực, cấu trúc 3D, tính đúng đắn diễn giải hoặc khả năng khái quát khảo sát mới.
