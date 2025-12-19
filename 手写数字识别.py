import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_and_preprocess_data():
    """
    加载并预处理数据
    """
    # 1. 准备数据
    train_data = pd.read_csv('digit-recognizer/train.csv')
    test_data = pd.read_csv('digit-recognizer/test.csv')
    
    # 2. 分离标签和特征
    y_train = train_data['label'].values
    X_train = train_data.drop(columns=['label']).values / 255.0
    X_test = test_data.values / 255.0
    
    # 3. 重塑数据为图像格式 (样本数, 高度, 宽度, 通道数)
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    
    # 4. 对标签进行独热编码
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    
    return X_train, y_train, X_test

def create_model():
    """
    创建改进的CNN模型
    """
    model = Sequential([
        # 第一个卷积块
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # 第二个卷积块
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # 全连接层
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    return model

def plot_training_history(history):
    """
    绘制训练历史图表
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 绘制准确率
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # 绘制损失
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    """
    主函数
    """
    # 加载和预处理数据
    X_train, y_train, X_test = load_and_preprocess_data()
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # 创建模型
    model = create_model()
    
    # 编译模型
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 显示模型结构
    model.summary()
    
    # 设置回调函数
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=0.0001
        )
    ]
    
    # 训练模型
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=128,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # 绘制训练历史
    plot_training_history(history)
    
    # 评估模型
    predictions = model.predict(X_test, verbose=1)
    predictions_labels = np.argmax(predictions, axis=1)
    
    # 生成提交文件
    submission = pd.DataFrame({
        'ImageId': np.arange(1, len(predictions) + 1), 
        'Label': predictions_labels
    })
    
    print("预测结果前10行:")
    print(submission.head(10))
    
    # 保存提交文件
    submission.to_csv('submission.csv', index=False)
    print(f"提交文件已保存，共 {len(submission)} 行数据")

if __name__ == "__main__":
    main()