import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = "./outputs"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

CONFIDENCE_THRESHOLD = 0.75

labels_map = {
    0: "Tiêu cực",
    1: "Tích cực"
}

def answer_question(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    confidence = torch.max(probs).item()

    if confidence < CONFIDENCE_THRESHOLD:
        return "Câu hỏi của bạn có vẻ không liên quan hoặc mô hình không chắc chắn. Vui lòng hỏi lại."
    pred_label = torch.argmax(probs).item()
    label_str = labels_map.get(pred_label, "Không xác định")
    return f"Label: {label_str}\n Confidence: {confidence:.2f}"

iface = gr.Interface(fn=answer_question,
                     inputs=gr.Textbox(lines=2, placeholder="Nhập câu hỏi của bạn..."),
                     outputs="text",
                     title="Demo Hỏi Đáp Mô Hình AI",
                     description="Nhập câu hỏi về chủ đề để nhận câu trả lời.")

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=8000)