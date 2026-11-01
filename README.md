# 👁️ Real-Time Face Detection

Một ứng dụng nhận diện khuôn mặt theo thời gian thực sử dụng Python và OpenCV.  
Chương trình truy cập webcam, phân tích từng khung hình và xác định vị trí khuôn mặt, sau đó vẽ khung chữ nhật xung quanh những khuôn mặt được phát hiện.

Dự án sử dụng Haar Cascade Classifier có sẵn trong OpenCV để thực hiện face detection mà không cần huấn luyện model riêng.

## ✨ Features

- **Real-time Detection:** Phát hiện khuôn mặt trực tiếp từ webcam.  
- **Face Localization:** Xác định vị trí và kích thước khuôn mặt bằng bounding boxes.  
- **Image Processing:** Chuyển đổi khung hình sang grayscale trước khi nhận diện.  
- **Detection Configuration:** Điều chỉnh `scaleFactor` và `minNeighbors` để kiểm soát kết quả nhận diện.  
- **Live Preview:** Hiển thị kết quả trực tiếp trên cửa sổ OpenCV.  

## 🛠 Tech Stack

**Python** – Ngôn ngữ lập trình chính;  
**OpenCV** – Xử lý hình ảnh và nhận diện khuôn mặt;  
**Haar Cascade** – Thuật toán phát hiện khuôn mặt.

## 🚀 Getting Started

Cài đặt OpenCV:

```bash
pip install opencv-python
```

Chạy ứng dụng:

```bash
python main.py
```

Đảm bảo webcam được kết nối và cấp quyền truy cập. Nhấn `Q` để thoát chương trình.

## 💡 What I Learned

Thông qua dự án này, mình thực hành xử lý video từ webcam, làm việc với image frames,  
chuyển đổi ảnh sang grayscale và sử dụng Haar Cascade để phát hiện đối tượng trong Computer Vision.  
Ngoài ra, dự án giúp hiểu thêm cách điều chỉnh các tham số detection để cân bằng giữa khả năng phát hiện và độ chính xác.
