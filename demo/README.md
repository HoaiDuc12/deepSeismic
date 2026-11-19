# Dữ liệu mô phỏng để kiểm tra pipeline

## Hai mô hình đối chiếu của bản nâng cấp

baseline/ và attention/ chứa checkpoint và báo cáo thí nghiệm kiểm tra, cùng dữ liệu,
seed 42, width=8, tile=112, batch=8, patches_per_image=10 và 12 epoch.
Khác biệt duy nhất về cấu hình huấn luyện là kiến trúc. Cả hai dùng sampling=balanced,
boundary_weight=1, lr=0.003, cùng phép chia 16/3/3 ảnh.
Trong giao diện mô phỏng, có thể chọn Attention U-Net để thử mô hình mới.

Ảnh khoanh của attention/preview dùng ngưỡng rà soát 0.65 và diện tích vùng 128 pixel;
raw_mask.png giữ mã dự đoán trước ngưỡng. Mục attention/evaluation có IoU/Dice/F1 biên
trên ba ảnh mô phỏng test. Các con số này không chứng minh tăng chất lượng trên địa chấn thật.
Audit trong attention/audit chỉ kiểm tra bộ mô phỏng, không phải bộ 300 ảnh đang chờ.

Để chạy lại đối chiếu (chọn thư mục đích chưa có thí nghiệm):

    python -m facies.simulate --out runs/simulated
    python -m facies.train --manifest runs/simulated/dataset.csv --out runs/compare-unet --arch unet --epochs 12 --tile-size 112 --batch-size 8 --patches-per-image 10 --width 8 --lr 0.003 --cpu-threads 3 --demo --patience 0 --boundary-weight 1 --sampling balanced
    python -m facies.train --manifest runs/simulated/dataset.csv --out runs/compare-attention --arch attention_unet --epochs 12 --tile-size 112 --batch-size 8 --patches-per-image 10 --width 8 --lr 0.003 --cpu-threads 3 --demo --patience 0 --boundary-weight 1 --sampling balanced

## Mẫu gốc giữ để tương thích

model.pt được huấn luyện từ đầu bằng U-Net width=12 trên 16 ảnh mô phỏng.
Ba ảnh mô phỏng validation dùng để chọn epoch; ba ảnh test giữ riêng để đo.
sample.png lấy từ test, kích thước 576 × 352 pixel.
Các kiểu sóng chỉ mô phỏng hoa văn phản xạ; chúng không thay thế dữ liệu và phân tướng của chuyên gia.

Các vùng nhỏ dự đoán sai có thể tạo thêm nhãn phụ trong ảnh preview/overlay.png.
Điểm trong metrics.json chỉ áp dụng cho bộ mô phỏng.

Để tái tạo, chạy lần lượt tại thư mục dự án:

    python -m facies.simulate --out runs/simulated
    python -m facies.train --manifest runs/simulated/dataset.csv --out runs/demo --epochs 22 --tile-size 112 --batch-size 8 --patches-per-image 10 --width 12 --lr 0.003 --cpu-threads 4 --demo
    python -m facies.predict --model runs/demo/best.pt --image runs/simulated/images/section_19.png --out runs/demo/preview

Khác biệt phiên bản PyTorch/phần cứng có thể làm số đo thay đổi.
