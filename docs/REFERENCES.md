# Danh mục tham khảo GitHub và nguồn nghiên cứu

Rà soát ngày 19/11/2026 cho bài toán: ảnh lát cắt vào, vùng tướng và tên lớp ra; dự kiến khoảng 300 ảnh chuyên gia khoanh trực tiếp. Đây là khảo sát tài liệu/mã nguồn công khai, không phải kết quả tái lập các mô hình bên ngoài.

## Nguồn ưu tiên

### 1. cigFacies / cigFaciesNet — gần nhất về hình thái phản xạ

- GitHub tác giả: https://github.com/huigcig/cigFaciesNet
- Bài báo ESSD, xuất bản 10/02/2025: https://essd.copernicus.org/articles/17/595/2025/
- Dữ liệu gốc: https://zenodo.org/records/10777460
- Bản lưu mã có DOI: https://zenodo.org/records/13150879

Nguồn phát hành có 8.000 mẫu skeleton 128 × 128, với năm lớp: parallel/subparallel gộp chung, clinoform, fill, hummocky và chaotic. Đây là nguồn phân loại mẫu ảnh; pixel đường phản xạ không phải mask tướng. Tôi đánh giá đây là ứng viên để học trước đặc trưng hình thái, rồi kiểm chứng chuyển giao sang bộ nhãn của mình.

Đối chiếu tạm, cần chuyên gia xác nhận:

| Nhãn dự án | Phạm vi đối chiếu |
| --- | --- |
| Hỗn loạn | chaotic là ứng viên gần về tên/hình thái |
| Song song và Á song song | Nguồn gộp hai tướng, không dạy sẵn việc phân biệt chúng |
| Gò đồi | Cân nhắc hummocky/mound, không mặc định tương đương hoàn toàn |
| Phân kì | divergent không phải lớp phát hành riêng |
| Sigma | Nếu chuyên gia dùng với nghĩa sigmoid: là kiểu con của clinoform, không đại diện toàn bộ clinoform |

Không đổi toàn bộ fill thành Phân kì hoặc clinoform thành Sigma. Cấu trúc phân cấp trong bài báo dùng để thảo luận hệ nhãn, không phải một phép ánh xạ tự động.

### 2. Alaudah — Facies Classification Benchmark / Netherlands F3

- Mã nguồn: https://github.com/yalaudah/facies_classification_benchmark
- Bài gốc: https://arxiv.org/abs/1901.07659
- Dữ liệu chính thức: https://zenodo.org/records/3755060

Có mô hình theo patch và toàn lát cắt, dữ liệu 3D đã diễn giải và tập kiểm tra riêng. Hữu ích làm benchmark pipeline và quy trình đánh giá. Các nhãn địa chất của F3 không mặc định trùng sáu nhãn hình thái do chuyên gia của dự án định nghĩa. Giữ hệ nhãn riêng và đánh giá khả năng chuyển giao thay vì gộp nhãn bằng số ID.

### 3. FaciesSAM — hỗ trợ tạo/sửa vùng bằng tương tác

- Repo: https://github.com/joshua-atolagbe/FaciesSAM
- Bài báo: https://doi.org/10.1109/ACCESS.2025.3547931

Repo có train.py, đánh giá instance/semantic, ứng dụng tương tác và liên kết checkpoint. README nêu môi trường NVIDIA/CUDA, Linux và Python 3.10; có giấy phép AGPL-3.0. Ưu tiên nghiên cứu cách tạo gợi ý vùng và cho chuyên gia sửa, chọn lớp. Không coi prompt tên tướng tiếng Việt là một bộ nhận diện sáu lớp đã có sẵn; cần huấn luyện/kiểm chứng theo đúng nhãn của mình. Chưa tải hoặc chạy checkpoint.

### 4. AdaSemSeg — ít mẫu và chuyển sang khảo sát khác

- Repo tác giả: https://github.com/Surojit-Utah/AdaSemSeg
- Bài báo: https://arxiv.org/abs/2501.16760
- DOI tạp chí: https://doi.org/10.1109/TGRS.2025.3595010

Hướng few-shot phân vùng với ví dụ support có nhãn, thích ứng số lớp khác nhau; kết hợp đặc trưng tự giám sát và dự đoán theo từng lớp. Đáng nghiên cứu nếu sau này có nhiều khảo sát nhưng ít nhãn ở mỗi khảo sát. README phân biệt kiểm tra nhanh theo patch với đánh giá toàn lát cắt. Không diễn giải “few-shot” thành vài ảnh đã khoanh chữ là đủ dùng ngay; vẫn cần support mask và đúng quy trình học trước. Chưa tái lập số đo hoặc xác nhận checkpoint tải được.

### 5. seisvit-decoding — hướng encoder thị giác học trước

- Repo: https://github.com/jinyisong123/seisvit-decoding
- Bản lưu dữ liệu kèm dự án: https://data.mendeley.com/datasets/gnvyh3msrj/1

README mô tả DINOv3, LoRA và nhiều decoder cho F3/Parihaka; có script train/eval và chỉ mục split. Dữ liệu/trọng số pretrained không đi kèm repo. Có thể dùng làm ứng viên thí nghiệm encoder học trước sau khi chuẩn hóa nhãn và có tài nguyên phù hợp. Chưa có cơ sở chọn nó thay U-Net chỉ từ tên kiến trúc; cần đối chiếu cùng split, ngân sách và seed.

## Bài báo và dữ liệu ngoài GitHub

### 6. UmixClick — nhấp điểm dương/âm để sửa biên

- Bài chính thức: https://doi.org/10.1038/s41598-025-32016-8
- Toàn văn: https://pmc.ncbi.nlm.nih.gov/articles/PMC12816730/

Phân vùng tương tác sử dụng điểm dương, điểm âm và mask trước đó. Phù hợp tham khảo bước giảm công vẽ nhãn/sửa ranh giới. Bài được công bố ngày 02/01/2026 dù DOI có chuỗi 2025. Thực nghiệm trên lát cắt 2D không đồng nghĩa tự động nhận tên tướng trên mọi vùng. Chưa xác minh được một repo triển khai chính thức từ bài, nên xếp là tài liệu phương pháp.

### 7. Leakage-aware full-section evaluation — thiết kế đánh giá

- Bài chính thức, 25/09/2026: https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2026.1910345/full

Nghiên cứu tách liên tục theo không gian, khoảng đệm, đánh giá toàn lát cắt và nhiều seed; cũng khảo sát học bán giám sát weak-to-strong. Cần tham khảo cho bộ ảnh gần nhau/giao nhau. Tôi chưa xác minh được repo tác giả trong trang bài. Không lấy mức tăng điểm trong khảo sát của họ làm mức tăng dự kiến cho 300 ảnh của mình.

### 8. AIcrowd — Seismic Facies Identification / Parihaka

- Nguồn cuộc thi: https://www.aicrowd.com/challenges/seismic-facies-identification-challenge

Nguồn dữ liệu 3D và bài toán nhận diện theo voxel do chuyên gia diễn giải. Hữu ích kiểm chứng pipeline/encoder trên dữ liệu công khai. Sáu lớp của cuộc thi không tự động đồng nghĩa sáu tướng hình thái của dự án. Dùng mask được cung cấp, không dùng ảnh chụp hình minh họa làm ground truth.

## Nguồn kỹ thuật vẫn giữ để đối chiếu

- Microsoft DeepSeismic: https://github.com/microsoft/seismic-deeplearning — cấu trúc pipeline nhiều mô hình. Repo đã archive ngày 16/11/2023, nên ưu tiên học cách tổ chức hơn là cài nguyên môi trường cũ.
- Seismic Segmentation Methods Benchmark: https://github.com/gabriel-cangucu/seismic-segmentation-benchmark — so sánh nhiều encoder–decoder và dữ liệu F3/Parihaka/Penobscot.

## Thứ tự đề xuất cho sản phẩm

1. Chốt định nghĩa sáu tướng, nhất là Sigma/sigmoid, Gò đồi và Phân kì, bằng một sổ tay ảnh ví dụ được chuyên gia duyệt.
2. Thử phương pháp tương tác để tạo mask từ vài ảnh mẫu, đo thời gian sửa của chuyên gia trước khi áp dụng cả bộ.
3. Giữ U-Net/Attention U-Net hiện tại làm đối chứng. Thử encoder học trước từ mẫu hình thái hoặc tự giám sát như các nhánh thí nghiệm riêng.
4. Chia theo tuyến/khối và khoảng đệm trước khi cắt patch. Các lát cắt giao nhau cần được xét chung để tránh kiểm tra trên phần địa chất đã xuất hiện ở train.
5. Đánh giá toàn lát cắt, theo từng tướng và nhiều seed; kèm ảnh lỗi và thời gian chuyên gia sửa. Chỉ sau đó quyết định tích hợp mô hình phức tạp hơn.

Các mục trên là lựa chọn nghiên cứu sau đợt rà soát. Chưa tích hợp cigFacies, FaciesSAM, AdaSemSeg hoặc DINOv3 vào bản chạy hiện tại. Chưa tải bộ dữ liệu/trọng số lớn hay thực thi mã repo bên ngoài trong lượt này.
