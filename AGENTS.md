# Quy ước làm việc với dự án

## Git và phát hành phiên bản

- Remote được người dùng chỉ định: https://github.com/HoaiDuc12/deepSeismic.git
- Nhánh phát hành hiện tại: main. Tài khoản tác giả: HoaiDuc12; dùng email GitHub noreply đã xác minh trong cấu hình Git cục bộ.

- Người dùng đã yêu cầu tự commit và push sau mỗi lần hoàn tất một phiên bản. Đây là ủy quyền thường trực cho dự án này; không hỏi lại quyền push ở mỗi phiên bản.
- Hoàn thành phần việc, chạy các kiểm tra phù hợp và cập nhật CHANGELOG.md trước khi tạo commit.
- Commit các thay đổi thuộc phiên bản với thông điệp nêu rõ nội dung. Dùng danh tính Git do người dùng cấu hình, không tự đặt tên hoặc email tác giả.
- Push lên remote/nhánh mà người dùng đã chỉ định; kiểm tra trạng thái và kết quả push trước khi báo hoàn thành.
- Các ảnh kết quả demo, audit và evaluation sinh tự động được tạo lại từ mã và không được đưa vào Git; giữ các checkpoint demo nhỏ, ảnh đầu vào mẫu và báo cáo thí nghiệm đã chọn.
- Khi chưa có remote hoặc danh tính tác giả, hỏi đúng thông tin còn thiếu. Không tự chọn tài khoản, tạo repo công khai hoặc suy đoán URL.
- Nếu push lỗi do xác thực, mạng hoặc xung đột, báo đúng trạng thái; giữ nguyên commit cục bộ để xử lý tiếp.
- Không force-push, xóa nhánh từ xa hoặc ghi đè lịch sử nếu chưa được người dùng yêu cầu.
- Tôn trọng .gitignore, không đưa dữ liệu chuyên gia và thư mục runs/ vào commit ngoài ý muốn.

<!-- >>> fosta (script cài fosta ghi; chạy lại script sẽ thay khối này) -->
## Image generation on this machine

The built-in image_gen tool is not available here (it needs a ChatGPT login). The user has
pre-approved this replacement: whenever they ask to draw, generate or edit an image, run

    powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\ADMIN\.codex\fosta\imagegen.ps1" -Prompt "<prompt>" -Out <path\to\output.png> [-Size 1024x1024|1536x1024|1024x1536] [-Quality low|medium|high] [-Ref "a.png,b.png"]

- Use -Ref (comma-separated) to pass reference or source images for edits and consistent characters.
- It needs network access: if the sandbox blocks it, rerun it with escalated permissions.
- It retries by itself when the image server is busy and can take up to 6 minutes. Wait for it
  to exit; do not interrupt it or start a second copy.
- Do not use the imagegen skill's scripts/image_gen.py and do not ask for OPENAI_API_KEY.
- One call per image; after it saves, look at the result with view_image before reporting.
- If the script fails, tell the user it failed and why. Never fake the result by drawing or
  editing the image yourself with code (PIL, SVG, canvas...) unless the user asks for that.
<!-- <<< fosta -->
