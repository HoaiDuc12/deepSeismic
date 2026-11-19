# Đo hiệu suất suy luận

Đối chiếu commit 63d0a80 với phiên bản gom batch qua nhiều hàng ô, dùng cùng Attention U-Net mô phỏng, PyTorch 2.6 CPU, 4 luồng, tile 112, overlap mặc định, batch 4, ngưỡng 0,55 và TTA tắt. Đây là thời gian suy luận đã tải checkpoint, gồm ghép ô và tính entropy. Chạy khởi động trước, lấy trung vị 5 lần trên cùng tiến trình và luân phiên thứ tự bản cũ/bản mới.

| Ảnh thử (rộng × cao) | Trước | Sau | Giảm thời gian |
| --- | ---: | ---: | ---: |
| Mẫu 576 × 352 | 0,366 s | 0,299 s | 18% |
| Ảnh hẹp 96 × 1760 | 0,410 s | 0,317 s | 23% |
| Mẫu ghép 1728 × 704 | 2,325 s | 2,231 s | 4% |

Các ảnh hẹp và ảnh lớn được cắt/ghép từ mẫu mô phỏng để kiểm tra hình dạng đầu vào. Mask giữ kích thước gốc; trên ba ảnh, hai mask khớp hoàn toàn và ảnh hẹp có một pixel đổi mã. Độ lệch điểm tối đa dưới 0,000024 trên thang [0, 1], do sai số làm tròn khi PyTorch đổi cách gom batch. Kết quả phụ thuộc bộ xử lý, kích thước ảnh và cấu hình; đây không phải phép đo trên GPU hoặc ảnh chuyên gia.

Trên ảnh mẫu 576 × 352, ảnh đã suy luận được dùng lại giữa các phép đo bên dưới; trung vị 5 lượt sau khi chạy khởi động.

| Dựng ảnh hoặc xuất tệp | Thời gian | Dung lượng tệp |
| --- | ---: | ---: |
| Gói rà soát trước tối ưu | 0,167 s | 1,77 MB |
| Gói rà soát hiện tại | 0,168 s | 1,77 MB |
| Gói gọn hiện tại | 0,117 s | 1,01 MB |
| Dựng bản xem trước trong bộ nhớ | 0,061 s | Không ghi tệp |

Gói rà soát giữ ảnh nguồn và công cụ chỉnh polygon. Gói gọn giảm khoảng 43% dung lượng ở phép đo này nhờ bỏ ba tệp chỉ dùng để sửa vùng. Bản xem trước không tạo các gói tải. Giao diện dùng lại dự đoán trong phiên khi điều chỉnh lớp phủ; các tệp tải vẫn giữ nguyên độ phân giải đầu vào.
