import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

def compute_mAP(outputs, labels):
    """
    Tính toán mAP (mean Average Precision) cho các đầu ra và nhãn.
    """
    outputs = outputs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    
    # Tính toán mAP cho từng lớp
    mAP = average_precision_score(labels, outputs, average='macro')  # 'macro' cho trung bình trên tất cả các lớp
    return mAP

# Dữ liệu
x = torch.tensor([[-1.0], [1.0], [3.0], [5.0]])
y = torch.tensor([[1.0], [0.0], [0.0], [1.0]])

# MLP đơn giản
model = nn.Sequential(
    nn.Linear(1, 4),      # Tầng ẩn đầu tiên với 4 node
    nn.Tanh(),            # Hàm kích hoạt Tanh (hoặc ReLU)
    nn.Linear(4, 1),      # Đầu ra
    nn.Sigmoid()          # Phân loại nhị phân
)

# Hàm mất mát và tối ưu
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Huấn luyện
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Dự đoán
with torch.no_grad():
    preds = model(x)
    predicted_classes = (preds >= 0.9).float()
    print("Predictions:", predicted_classes.view(-1).tolist())

    #Print mAP
    print(f"mAP: {compute_mAP(preds, y):.4f}")

    