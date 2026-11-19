# Thay đổi

## Bản nâng cấp 19/11/2026

- Hai kiến trúc: U-Net cũ tương thích checkpoint trước và U-Net residual có attention/context giãn.
- Cấu hình JSON, ưu tiên lấy mẫu tướng, CE tăng trọng số biên + Dice, giảm learning rate, dừng sớm, gradient clipping và AMP tùy CUDA.
- best.pt để suy luận, last.pt để tiếp tục đúng trạng thái optimizer/RNG; config và metrics lưu mỗi epoch.
- Kiểm tra nội dung trùng SHA-256 giữa các tập và fingerprint cả nhãn/ảnh/cách chia.
- Audit độ phủ nhãn, lỗi cặp ảnh và phân bố lớp trước khi huấn luyện.
- Đánh giá độc lập toàn lát cắt, precision/recall/F1 và Boundary F1 bên cạnh IoU/Dice.
- TTA lật ngang, bản đồ confidence/entropy, mặt nạ pixel cần rà soát và lưu raw_mask.
- Xử lý cả thư mục, bảng ưu tiên rà soát và hồ sơ mô hình tạo ra từng kết quả.
- Xuất LabelMe, source.png và HTML hiệu chỉnh ngoại tuyến; vùng không biểu diễn chính xác được liệt kê thay vì lấp lỗ sai.
- Giao diện bốn khu vực: một ảnh, thư mục, dữ liệu và thí nghiệm.

Các cải tiến kỹ thuật đã được kiểm thử; hiệu quả trên ảnh chuyên gia vẫn cần được đo khi dữ liệu được cung cấp. Trọng số demo cũ được giữ để đối chiếu.
