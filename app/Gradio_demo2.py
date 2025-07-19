import numpy as np
from PIL import Image
import gradio as gr
import tensorflow as tf

# 1. Load your saved model
model = tf.keras.models.load_model('mnist_cnn.keras')  # or .h5 if that's what you used

# 2. Prediction helper
def predict_digit(image):
    """
    image: HxW or HxWxC NumPy array from Gradio.
    """
    # Convert to PIL, grayscale & resize to 28×28
    img = Image.fromarray(image)
    img = img.convert("L").resize((28, 28))
    arr = np.array(img, dtype='float32') / 255.0
    arr = arr.reshape(1, 28, 28)

    # Predict
    preds = model.predict(arr)[0]
    top3 = np.argsort(preds)[-3:][::-1]
    return { str(i): float(preds[i]) for i in top3 }

# 3. Gradio interface
iface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Image(type="numpy"),         # simple upload
    outputs=gr.Label(num_top_classes=3),
    title="MNIST Digit Recognizer",
    description="Upload or draw a digit image; I'll show my top 3 guesses."
)

# 4. Launch with a public link
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7861, share=False)

# if __name__ == "__main__":
#     iface.launch(share=True)
