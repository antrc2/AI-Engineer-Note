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
        
        * **Mô hình hồi quy**
            
            * Sai số tuyệt đối trung bình: Mean Absolute Error (MAE)
            * Sai số bình phương trung bình: Mean Squared Error (MSE)
            * Căn bậc hai của MSE: Root Mean Squared Error (RMSE)
            * Hệ số xác định: R² Score (Coefficient of Determination)

        * **Mô hình phân loại**
            * Độ chính xác: Accuracy
            * Độ chính xác dương tinh: Precision
            * Độ nhạy: Recall
            * F1 Score
            * Ma trận nhầm lẫn: Confusion Matrix
            * Đường cong ROC và diện tích AUC: ROC Curve & AUC

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
3. ## Tuần 6 - 7: Chuyên sâu Natural Language Processing (NLP)

    * Ngày 36: Tổng quan về NLP
        * NLP là gì? Ứng dụng thực tế
        * Các tác vụ chính: Text Classification, NER, Summarization, QA, Chatbot
            - Text Classification: Phân loại văn bản
            - NER: Nhận diện và phân loại các thực tể
            - Summarization: Tóm tắt văn bản
            - QA: Hỏi đáp
            - Chatbot
        * NLP truyền thống vs NLP hiện đại (Deep Learning-based)
        * Pipeline xử lý văn bản
            - Tiền xử lí văn bản

    * Ngày 37: Tiền xử lý văn bản
        * Tokenization: từ, subword (BPE, WordPiece)
            - BPE (Byte Pair Encoding): `unhappiness → [u, n, h, a, p, p, i, n, e, s, s]`
                - Đếm tần xuất các kí tự lặp liên tiếp
                - Tìm cặp xuất hiện nhiều nhất: "p p"
                - Ghép lại thành một token mới: `pp`
            
            Code: 
            ```
            import tensorflow as tf

            def get_pairs_tf(tokens):
                """Lấy các cặp ký tự liên tiếp trong tensor"""
                pairs = tf.stack([tokens[:-1], tokens[1:]], axis=1)
                return pairs

            def count_pairs_tf(pairs):
                """Đếm tần suất các cặp"""
                pairs_str = tf.strings.reduce_join(pairs, axis=1, separator=" ")
                unique_pairs, _, counts = tf.unique_with_counts(pairs_str)
                return unique_pairs, counts

            def merge_pair_tf(tokens, merge_pair):
                """Gộp cặp merge_pair thành một token mới"""
                i = 0
                new_tokens = []
                while i < tf.shape(tokens)[0] - 1:
                    current = tf.strings.join([tokens[i], tokens[i+1]], separator=" ")
                    if current == merge_pair:
                        new_tokens.append(tf.strings.join([tokens[i], tokens[i+1]]))
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                # Thêm phần tử cuối nếu chưa xử lý
                if i == tf.shape(tokens)[0] - 1:
                    new_tokens.append(tokens[i])
                return tf.convert_to_tensor(new_tokens)

            # Bắt đầu
            word = "unhappiness"
            tokens = tf.strings.unicode_split(word, 'UTF-8')

            print("Tokens ban đầu:", tokens.numpy())

            # Bước 1: Tạo cặp
            pairs = get_pairs_tf(tokens)

            # Bước 2: Đếm cặp
            unique_pairs, counts = count_pairs_tf(pairs)

            # Lấy cặp phổ biến nhất
            top_pair_index = tf.argmax(counts)
            most_common_pair = unique_pairs[top_pair_index]
            print("Cặp phổ biến nhất:", most_common_pair.numpy())

            # Bước 3: Gộp cặp đó lại
            new_tokens = merge_pair_tf(tokens, most_common_pair)
            print("Tokens sau merge:", new_tokens.numpy())
            ```
            Kết quả: 
            ```output
            Tokens ban đầu: [b'u' b'n' b'h' b'a' b'p' b'p' b'i' b'n' b'e' b's' b's']
            Cặp phổ biến nhất: b'u n'
            Tokens sau merge: [b'un' b'h' b'a' b'p' b'p' b'i' b'n' b'e' b's' b's']
            ```
            - WordPiece: `playing`
                - Sau khi tách từ: `play` và `##ing`
                - `##` dùng để đánh dấu rằng token này không phải đứng đầu từ
        * Lowercasing, stemming, lemmatization
            - Lowcasing: chuyển thành viết thường
            - Stemming: 
                - Cắt bỏ hậu tố để đưa về từ gốc, nhưng không nhất thiết là từ đúng trong từ điển
                - `running` -> `runn`
            - Lemmatization:
                - Cắt bỏ hậu tố để đưa về từ gốc, nhưng từ sẽ là đúng trong từ điển
                - `easily` -> `easy`

        * Loại bỏ stopwords, punctuation
            - Stopwords: là các từ không mang nhiều nghĩa trong ngữ cảnh, được lược bỏ để giảm nhiễu
            - Punctuation: là các dấu dâu
        * Thực hành với NLTK, SpaCy, tokenizer từ Hugging Face

    * Ngày 38: Biểu diễn văn bản
        * Bag-of-Words, TF-IDF
            - Bag-of-Words: Biểu diễn văn bản bằng tần xuất xuất hiện của từng từ trong văn bản
                - Ưu điểm: Đơn giản, dễ cài đặt, hiệu quả với các mô hình truyền thống (SVM, Logistic Regression)
                - Nhược điểm: Không giữ được ngữ nghĩa và thứ tự các từ, Vector thường rất thưa 
            - TF-IDF: 
                - Cái tiến BoW bằng cách giảm trọng số của các từ phổ biến
                - Là sự kết hợp của:
                    - TF (Term Frequency): Độ thường xuyên của từ trong một văn bản
                    - IDF: Độ hiếm của từ trong toàn bộ văn bản
                - Ưu điểm: Nhấn mạnh được các đặc trưng, phân biệt nội dung. Hạn chế ảnh hưởng của các stopwords
                - Nhược điểm: Không xử lí được ngữ cảnh, đồng nghĩa - trái nghĩa. Vector thưa


        * Word2Vec: CBOW, Skip-Gram
            - CBOW (Continuous Bag of Words): Dự đoán từ ở trung tâm. Ví dụ: `["Tôi","thích","học","ở","trường"]`. Thì CBOW sẽ dự đoán từ trung tâm là `"NLP"`. Và câu hoàn chỉnh sau khi dự đoán là `"Tôi thích học NLP ở trường"`
            - Skip-Gram: Dùng để dự đoán các từ lân cận, dựa theo từ ở trung tâm
        * GloVe: Là mô hình học biểu diễn từ dựa trên thống kê toàn cục của văn bản, đặc biệt dựa trên ma trận đồng xuất hiện từ
        * Contextual Embeddings: 
            - Giúp mô hình hiểu được nghĩa của từ dựa trên ngữ cảnh của câu thay vì chỉ gán 1 vector tĩnh như Word Embedding.
            - Danh sách các mô hình liên quan đến Contextual Embeddings
                - ELMo:
                    - Sử dụng mô hình ngôn ngữ 2 chiều: Bi-Directional Language Model (BiLM)
                    - Dùng làm embedding đầu vào cho các mô hình NLP: Classification, NER, QA
                - BERT:
                    - Được phát triển từ mô hình ngôn ngữ 2 chiều, giúp hiểu rõ ngữ cảnh của một từ hơn rất nhiều
                - RoBERTa:
                    - Là biến thể của BERT, cải tiến hiệu xuất của BERT bằng cách thay đổi phương pháp huấn luyện, sử dụng nhiều dữ liệu hơn và huấn luyện lâu hơn
                - XLNet:
                    - Là mô hình tự hồi quy. Là sự kết hợp giữa 2 mô hình: BERT + GPT
                    - Thay vì dự đoán 1 số token bị mask như BERT, thì XLNet dự đoán tất cả các token theo thứ tự hoán vị ngẫu nhiên
                - ALBERT:
                    - Là một bản Lite của BERT, giảm kích thước mô hình và tăng tốc độ huấn luyện, trong khi hiệu xuất vẫn bằng BERT, hoặc có thể cao hơn
                - ELECTRA:
                    - Thay vì sử dụng mask như BERT, thì ELECTRA có khả năng phát hiện ra các token bị thay thế
                    - Là một mô hình nhỏ mà mạnh, hiệu quả huấn luyện tốt hơn
                - T5: 
                    - Ý tưởng cả T5 là chuyển tất cả các bài toán về NLP về dạng Text To Text
                    - Sử dụng kiến trúc chuẩn Transformers
                - ERNIE
        * Thực hành so sánh các loại embedding

    * Ngày 39: Mạng nơ-ron cho NLP
        * RNN, LSTM, GRU: hoạt động và kiến trúc
        * Bi-directional RNN
        * Cơ chế Attention cơ bản
        * Thực hành: Text Classification bằng LSTM (IMDb/Yelp)

    * Ngày 40: Transformer và BERT
        * Self-Attention là gì
        * Kiến trúc Transformer Encoder-Decoder
        * Giới thiệu BERT, GPT, T5
        * Pre-training vs Fine-tuning
        * Thực hành: Fine-tune BERT phân loại văn bản

    * Ngày 41: Gán nhãn chuỗi – Named Entity Recognition (NER)
        * Bài toán Sequence Labeling
        * BiLSTM-CRF là gì
        * Hugging Face pipeline cho Token Classification
        * Thực hành: Huấn luyện BERT cho NER (CoNLL-2003)

    * Ngày 42: Hỏi đáp tự động (Question Answering)
        * Extractive QA vs Generative QA
        * Mô hình: DistilBERT, BERT, T5
        * Dataset: SQuAD
        * Thực hành: hỏi đáp văn bản với Hugging Face pipeline

    * Ngày 43: Tóm tắt văn bản
        * Extractive vs Abstractive Summarization
        * Giới thiệu BART, T5
        * Thực hành: tóm tắt tin tức, tài liệu dài

    * Ngày 44: Sinh văn bản (Text Generation)
        * Language Model truyền thống: n-gram, RNN
        * Mô hình hiện đại: GPT-2, GPT-Neo
        * Thực hành: Fine-tune GPT-2 để sinh thơ, văn, truyện

    * Ngày 45: Chatbot với RAG (Retrieval-Augmented Generation)
        * Tổng quan hệ thống RAG
        * Kết hợp mô hình sinh và truy xuất
        * Sử dụng thư viện: Haystack, LangChain
        * Demo chatbot hỏi đáp từ tài liệu nội bộ

    * Ngày 46: NLP nâng cao – Prompt Engineering
        * Zero-shot, few-shot learning
        * Prompting: In-context Learning, Chain-of-Thought
        * Sử dụng OpenAI API hoặc Transformers
        * Thực hành: Viết prompt hiệu quả cho bài toán NLP

    * Ngày 47: Dự án 1 – Phân loại cảm xúc từ bình luận
        * Thu thập dữ liệu: IMDb, Facebook, Youtube comments
        * Tiền xử lý văn bản
        * Huấn luyện mô hình BERT hoặc BiLSTM
        * Trực quan hóa kết quả bằng Streamlit

    * Ngày 48: Dự án 2 – Chatbot hỏi đáp tài liệu
        * Thu thập dữ liệu nội bộ (PDF, DOC, TXT)
        * Xây dựng hệ thống RAG (retriever + generator)
        * Triển khai demo chatbot với giao diện (Gradio/Streamlit)

4. ## Tuần 8: Hoàn thiện dự án và tổng kết
    * Demo Jupiter
    * Deploy Streamlit App