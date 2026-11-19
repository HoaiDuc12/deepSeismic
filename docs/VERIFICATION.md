# Kiểm tra bản nâng cấp

Ngày kiểm tra: 19/11/2026. Môi trường thực thi: Windows, Python 3.13, PyTorch 2.6 CPU.
Không có CUDA trong môi trường này; nhánh AMP được triển khai theo API nhưng chưa xác nhận tốc độ/bộ nhớ trên GPU.

## Lệnh kiểm tra tự động

    python -m unittest discover -s tests -v

Kết quả lượt cuối: **41 kiểm thử đạt**, thời gian 54.275 giây trên máy hiện tại.

Bộ kiểm tra bao phủ:

- Tương thích checkpoint U-Net cũ và checkpoint Attention U-Net.
- Ghép ô ảnh kích thước lẻ và nhỏ hơn một ô; căn chỉnh TTA, entropy và mã 255.
- Pixel chưa biết có gradient 0; mask thô không bị thay đổi bởi lọc hiển thị.
- Nhập viền màu/chữ làm mịn, nhãn polygon, rò rỉ nhóm và ảnh trùng nội dung.
- Precision/recall/F1, IoU/Dice, dung sai biên và báo cáo audit có lỗi.
- Chạy 2 epoch liên tục và 1 epoch rồi resume đến 2 cho trọng số giống nhau từng bit trên CPU.
- Đổi một pixel nhãn làm đổi dấu vân tay bộ dữ liệu.
- Bốn vùng giao diện, inference demo thật, batch, báo cáo cũ/mới và báo cáo đánh giá phẳng.
- Lưu nhãn máy chưa duyệt không tự biến thành nhãn chuyên gia; kiểm tra trình duyệt Edge nếu máy có Playwright và Edge.

## Các luồng chạy thực tế

Đã chạy huấn luyện Attention U-Net từ cấu hình CPU ngắn, suy luận từ checkpoint vừa tạo,
đọc lại checkpoint v1, chạy TTA/ngưỡng trên ảnh gốc, xử lý thư mục và xuất ZIP.
Đã chạy audit và evaluate trên bộ mô phỏng; các ảnh/mask giữ nguyên 576 × 352 pixel.

Bản mô phỏng đối chiếu hai kiến trúc nằm trong demo/baseline/ và demo/attention/.
Cấu hình chung và bảng tổng hợp nằm trong demo/comparison.json.
Chọn checkpoint theo validation, số đo test chỉ để kiểm tra thí nghiệm mô phỏng đã cố định.
Số liệu mô phỏng không chứng minh tính đúng địa chất hoặc khả năng khái quát dữ liệu thật.

## Những phần cần dữ liệu chuyên gia để kiểm tra tiếp

Tính chính xác của nhãn phẳng sau chuyển đổi; phần tín hiệu bị chữ/viền che;
khả năng phân biệt tướng gần nhau; tính độc lập không gian giữa các lát cắt;
khả năng áp dụng giữa các khảo sát và lựa chọn ngưỡng có ý nghĩa thực tế.

Chưa có bộ khoảng 300 ảnh thực trong dự án để kết luận các điểm này.
