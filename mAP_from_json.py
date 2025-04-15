import json

def calculate_precision_and_map(output_infer, output_true):
    """
    Tính Precision và Mean Average Precision (mAP) dựa trên kết quả mô hình và ground truth.
    """
    total_precision = 0
    ap_sum = 0
    count = 0
    
    for key, true_labels in output_true.items():
        predicted_labels = output_infer.get(key, [])
        
        if not true_labels:
            continue
        
        # Precision
        true_positive = sum(1 for label in predicted_labels if label in true_labels)
        precision = true_positive / len(predicted_labels) if predicted_labels else 0
        total_precision += precision
        
        # AP (Average Precision)
        relevant_labels = set(true_labels)
        retrieved_labels = []
        correct_count = 0
        ap = 0
        
        for i, label in enumerate(predicted_labels, start=1):
            retrieved_labels.append(label)
            if label in relevant_labels:
                correct_count += 1
                ap += correct_count / i
        
        ap = ap / len(relevant_labels) if relevant_labels else 0
        ap_sum += ap
        count += 1
    
    precision_avg = total_precision / count if count else 0
    mean_ap = ap_sum / count if count else 0
    
    return precision_avg, mean_ap

if __name__ == "__main__":
    # Đọc dữ liệu từ file JSON
    with open("output_infer.json", "r", encoding="utf-8") as f:
        output_infer = json.load(f)
    
    with open("output_true.json", "r", encoding="utf-8") as f:
        output_true = json.load(f)
    
    # Tính Precision và mAP
    precision_avg, mean_ap = calculate_precision_and_map(output_infer, output_true)
    
    print(f"Precision: {precision_avg:.4f}")
    print(f"Mean Average Precision (mAP): {mean_ap:.4f}")
