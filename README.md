# Roadmap 60 ngày trở thành AI Engineer cùng Aleotron

1. ## Tuần 1 - 2: Ôn tập Lý thuyết AI và Machine Learning cơ bản

    * Ngày 1: Giới thiệu về AI và ML: 
        
        * Phân biệt AI, ML, DL
        * Lịch sử phát triển của AI
        * Ứng dụng thực tế của AI

    * Ngày 2-3: Các loại học trong ML:
        * Học có giám sát: Supervised Learning
        * Học không giám sát: Unsupervised Learning
        * Học bán giám sát: Semi-supervised Learning
        * Học tăng cường: Reinforcement Learning

    * Ngày 4-6: Quy trình xây dựng hệ thống ML:

        * Bước 1: Xác định yêu cầu bài toán
        * Bước 2: Thu thập dữ liệu
        * Bước 3: Tiền xử lí
        * Bước 4: Chọn mô hình
        * Bước 5: Huấn luyện mô hình
        * Bước 6: Đánh giá mô hình
        * Bước 7: Triển khai mô hình

    * Ngày 7-9: Bài toán hồi quy

        * Hồi quy tuyến tính: Linear Regression
        * Hồi quy đa thức: Polynomial Regression
        * Hồi quy Ridge: Ridge Regression (Chuẩn hóa L2)
        * Hồi quy Lasso: Lasso Regression (Chuẩn hóa L1)
        * Hồi quy ElasticNet: ElasticNet Regression
        * Hồi quy cây quyết định: Decision Tree Regression
        * Hồi quy rừng ngẫu nhiên: Random Forest Regression
        * Hồi quy tăng cường dần: Gradient Boosting Regression
        * Hồi quy K láng giềng gần nhất: K-Nearest Neighbors Regression (KNN)
        * Hồi quy vector hỗ trợ: Support Vector Regression (SVR)

    * Ngày 10-12: Bài toán phân loại

        * Phân loại nhị phân: Binary Classification
        * Phân loại đa lớp: Multi-class Classification
        * Phân loại đa nhãn: Multi-label Classification
        * Phân loại mất cân bằng: Imbalanced Classification
        * Phân loại trực tuyến: Online Classification
        * Phân loại xác xuất: Probabilistic Classification

    * Ngày 13: Chỉ số đánh giá mô hình
        * **Mô hình phân loại**
            * Độ chính xác: Accuracy
            * Độ chính xác dương tinh: Precision
            * Độ nhạy: Recall
            * F1 Score
            * Ma trận nhầm lẫn: Confusion Matrix
            * Đường cong ROC và diện tích AUC: ROC Curve & AUC
        * **Mô hình hồi quy**
            
            * Sai số tuyệt đối trung bình: Mean Absolute Error (MAE)
            * Sai số bình phương trung bình: Mean Squared Error (MSE)
            * Căn bậc hai của MSE: Root Mean Squared Error (RMSE)
            * Hệ số xác định: R² Score (Coefficient of Determination)

    * Ngày 14: Tổng kết và kiểm tra kiến thức

2. ## Tuần 3 - 5: Học sâu (Deep Learning) và mạng Nơ-ron (Neural Network)
    * Ngày 15: Giới thiệu Deep Learning:
        * Khái niệm học sâu, mạng nơ-ron
        * Phân biệt ML và DL
        * Ví dụ và ứng dụng của DL
    
    * Ngày 16: Kiến trúc Perceptron
        * Neuron
        * Layer
        * Weights
        * Bias
        * Forward Propagation
     * Ngày 17: Hàm kích hoạt và hàm mất mát
        * Sigmoid
        * ReLU
        * Softmax
        * Cross-Entropy Loss
    * Ngày 18: Backpropagation & Gradient Descent
        * Đạo hàm riêng
        * Tối ưu hóa bằng SDG (Stochastic Gradient Descent)
    * Ngày 19: Cài đặt môi trường
        * Cài TensorFlow/Keras/PyTorch
        * Chạy thử Hello World trong AI
    * Ngày 20: Dataset MNIST
        * Load dữ liệu
        * Tiền xử lí ảnh số
        * Train model đơn giản
    * Ngày 21: Đánh giá và tối ưu model
        * Độ chính xác: Accuracy
        * Loss Curves: Hàm mất mát
        * Overfitting: Học thuộc
        * Underfitting: Học chưa đủ
        * Early Stopping: Dừng sớm
    * Ngày 22: Giới thiệu CNN
        * Tại sao CNN phù hợp với ảnh
        * Convolution Layer
        * Pooling Layer
    * Ngày 23: Các kiến trúc nổi tiếng
        * LeNet
        * AlexNet
        * VGGNet
        * GoogLeNet
        * ResNet
        * DenseNet
        * Xception
        * MobileNet
        * EfficentNet
    * Ngày 24: Xây dựng CNN đơn giản
        * Train model nhận diện chữ số MNIST
        * Sử dụng Keras Sequential API
    * Ngày 25: Nhận diện hình ảnh phức tạp hơn (chó, mèo, ...)
    * Ngày 26: Data Agumentation
        * Tăng cường dữ liệu bằng ImageDataGenerator
    * Ngày 27: Fine-tune & Pretrained Model
        * Sử dụng MobileNet/ResNet từ TensorFlow Hub
    * Ngày 28: Dự án nhỏ:
        * Datasets Dogs & Cats từ Kaggle
        * Huấn luyện CNN đơn giản
    * Ngày 29: Giới thiệu RNN
        * Mạng nơ-ron hồi tiếp
        * Bài toán chuỗi thời gian
    * Ngày 30: Long Short-Term Memory & Gated Recurrent Unit (LSTM & GRU)
        * Giải thích cách quên/thêm thông tin
        * So sánh với RNN truyền thống
    * Ngày 31: Tokenization & Embedding
        * Mã hóa văn bản
    * Ngày 32: Text Classification
        * Phân loại cảm xúc (Dataset IMDB)
    * Ngày 33: Text Generator
        * Tự động tạo văn bản
        * Đoán kí tự tiếp theo
    * Ngày 34: PyTorch
        * Viết lại 1 bài toán cũ bằng PyTorch
        * So sánh với TensorFlow
        * Làm quen với tensor, autograd
    * Ngày 35: Tổng kết và kiểm tra kiến thức
3. ## Tuần 6-7:
    * Ngày 36: Giới thiệu NLP
        * Tokenization
        * Lemmatization
        * StopWords
        * Bag-Of-Words
        * Term Frequency - Inverse Document Frequency (TF-IDF)
    * Ngày 37: Word Embedding
        * Word2Vec
        * GloVe
        * FastText
        * Embedding Layer trong Keras
    * Ngày 38: Transformer & Attention
        * Ý tưởng cốt lõi của mô hình Transformer
        * Giới thiệu BERT
    * Ngày 39: Text Generation
        * RNN
        * LSTM
        * GRU
        * Tự động sinh văn bản đơn giản
    * Ngày 40: Chatbot đơn giản
        * Rule-based Chatbot
        * ML-based Chatbot
        * Xây dựng Intent Recognition
    * Ngày 41: Dự án: Phân tích cảm xúc trên review sản phẩm hoặc tạo chatbot đơn giản
    * Ngày 42: Giới thiệu về Computer Vision
        * Classification
        * Detection
        * Segmentation
        * Sử dụng datasets phổ biến: COCO, ImageNet, CIFAR
    * Ngày 43: Object Detect
        * YOLO
        * SSD
    * Ngày 44: Transfer Learning trong CV
        * Sử dụng Model Pretrained như ImageNet, ResNet, EffencientNet
    * Ngày 45: Image Segmentation
        * Giới thiệu U-Net
        * Phân vùng ảnh y tế hoặc vật thể
    * Ngày 46: Face Regconition
        * Nhận diện khuôn mặt bằng OpenCV + Deep Learning
    * Ngày 47: Tăng cường dữ liệu (Data Agumentation)
        * Albumentations
        * ImageDataGenerator
    * Ngày 48: Dự án: Nhận diện khuôn mặt hoặc phát hiện đối tượng đơn giản
4. ## Tuần 8: Hoàn thiện dự án và tổng kết
    * Demo Jupiter
    * Deploy Streamlit App