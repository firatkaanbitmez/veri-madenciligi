import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score



def sigmoid(x):
    """
    Sigmoid aktivasyon fonksiyonu
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """
    Sigmoid aktivasyon fonksiyonunun türevi
    """
    return x * (1 - x)

def initialize_parameters(input_size, hidden_size, output_size):
    """
    Model parametrelerini başlatma
    """
    W1 = np.random.randn(input_size, hidden_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size)
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

def forward_propagation(X, W1, b1, W2, b2):
    """
    İleri yayılım işlemi
    """
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2

def calculate_cost(A2, Y):
    """
    Hata hesaplama
    """
    m = Y.shape[0]
    cost = -np.sum(np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), (1 - Y))) / m
    return cost

def backward_propagation(X, Y, Z1, A1, Z2, A2, W1, W2, b1, b2, learning_rate):
    """
    Geri yayılım işlemi
    """
    m = X.shape[0]
    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(A1.T, dZ2)
    db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
    dZ1 = np.dot(dZ2, W2.T) * sigmoid_derivative(A1)
    dW1 = (1 / m) * np.dot(X.T, dZ1)
    db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2

def train_neural_network(X_train, y_train, X_test, y_test, input_size, hidden_size, output_size, learning_rate, epochs):
    """
    Sinir ağı modelinin eğitimi
    """
    # Parametreleri başlatma
    W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)
    y_train = y_train.astype(int)  # y_train'i tamsayıya dönüştür
    y_test = y_test.astype(int)    # y_test'i tamsayıya dönüştür
 
    # Eğitim sırasında kaydedilecek metriklerin listeleri
    train_costs = []
    test_costs = []
    train_accuracies = []
    test_accuracies = []
    
    # Eğitim döngüsü
    for epoch in range(epochs):
        # İleri yayılım ve geri yayılım
        Z1, A1, Z2, A2 = forward_propagation(X_train, W1, b1, W2, b2)
        W1, b1, W2, b2 = backward_propagation(X_train, y_train.reshape(-1, 1), Z1, A1, Z2, A2, W1, W2, b1, b2, learning_rate)
        
        # Eğitim verisi üzerinde maliyeti hesapla ve kaydet
        train_cost = calculate_cost(A2, y_train.reshape(-1, 1))
        train_costs.append(train_cost)
        
        # Eğitim verisi üzerinde doğruluk hesapla ve kaydet
        train_predictions = predict(X_train, W1, b1, W2, b2)
        train_predictions = (train_predictions > 0.5).astype(int)

        train_accuracy = accuracy_score(y_train, train_predictions)
        train_accuracies.append(train_accuracy)
        
        # Test verisi üzerinde maliyeti hesapla ve kaydet
        Z1_test, A1_test, Z2_test, A2_test = forward_propagation(X_test, W1, b1, W2, b2)
        test_cost = calculate_cost(A2_test, y_test.reshape(-1, 1))
        test_costs.append(test_cost)
        
        # Test verisi üzerinde doğruluk hesapla ve kaydet
        test_predictions = predict(X_test, W1, b1, W2, b2)
        test_accuracy = accuracy_score(y_test, test_predictions)
        test_accuracies.append(test_accuracy)
        
        # Her 100 epoch'ta bir eğitim ve test verisi için maliyeti ve doğruluğu yazdır
        if epoch % 100 == 0:
            print(f"Epoch {epoch}:")
            print(f"  Train Cost: {train_cost}, Train Accuracy: {train_accuracy}")
            print(f"  Test Cost: {test_cost}, Test Accuracy: {test_accuracy}")
    
    return W1, b1, W2, b2, train_costs, test_costs, train_accuracies, test_accuracies

def predict(X_test, W1, b1, W2, b2):
    """
    Modelin test verisi üzerinde tahmin yapması
    """
    _, _, _, predictions = forward_propagation(X_test, W1, b1, W2, b2)
    predictions = (predictions > 0.5).astype(int)
    return predictions

# Veriyi yükleme
data_path = "C:\\Users\\FIRAT\\Desktop\\myProject\\veri-madenciligi\\Backpropagation-Projesi\\data.txt"
data = np.loadtxt(data_path)
X = data[:, :-1]
y = data[:, -1]

# Veriyi eğitim ve test kümelerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hiperparametreler
input_size = X_train.shape[1]
hidden_size = 10  # Değiştirilebilir
output_size = 1   # İkili sınıflandırma varsayımı
learning_rate = 0.01
epochs = 1000

# Modeli eğitme
W1, b1, W2, b2, train_costs, test_costs, train_accuracies, test_accuracies = train_neural_network(X_train, y_train, X_test, y_test, input_size, hidden_size, output_size, learning_rate, epochs)

# Sonuçları görselleştirme
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(range(epochs), train_costs, label='Train')
plt.plot(range(epochs), test_costs, label='Test')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('Cost over epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(epochs), train_accuracies, label='Train')
plt.plot(range(epochs), test_accuracies, label='Test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over epochs')
plt.legend()

plt.tight_layout()
plt.show()
