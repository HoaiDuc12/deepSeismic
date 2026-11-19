# Thay đổi

## Bản tối ưu suy luận và giao diện — 19/11/2026

- Gom batch qua nhiều hàng ô, đặc biệt hữu ích với lát cắt hẹp. Vẫn ghép ô chồng lấp và giữ kích thước ảnh gốc.
- Giữ dự đoán trong phiên làm việc: đổi ngưỡng rà soát, độ đậm hoặc diện tích hiển thị chỉ vẽ lại kết quả.
- Xem trước giới hạn 1600 px, tách thời gian suy luận khỏi thời gian dựng ảnh; tạo gói tải đầy đủ theo yêu cầu.
- Giao diện workbench hai cột với bốn metric chia hai hàng để giữ nhãn rõ trên màn hình thường.
- Chế độ xuất gọn cho cả thư mục giữ ảnh khoanh, mask và bản đồ điểm; chế độ rà soát giữ thêm ảnh nguồn, LabelMe và trang sửa vùng.
- Nén ZIP nhanh hoặc gọn, và gom đúng các tệp của lượt vừa chạy.
- Bộ kiểm thử phủ gom batch và tương đương mask/score trên lát cắt hẹp; toàn bộ 45 kiểm thử đạt.

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
