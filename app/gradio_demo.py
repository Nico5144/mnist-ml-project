# gradio_demo.py
# function to run demo of completed proj

import tensorflow as tf
import numpy as np
import gradio as gr
from tensorflow.keras.models import load_model

# 1. Load your saved model
#model = load_model('mnist_cnn.h5')
model = tf.keras.models.load_model('mnist_cnn.keras')

# 2. Preprocessing + prediction function
def predict_digit(image):
    """
    image: numpy array HxWxC or HxW from Gradio input.
    Returns a dict of top-3 digit probabilities.
    """
    # If color channels present, convert to grayscale
    if image.ndim == 3:
        image = image[:,:,0]
    # Resize / normalize to 28x28 in [0,1]
    image = np.array(image, dtype='float32')
    image = image.reshape(1,28,28) / 255.0

    # Run prediction
    preds = model.predict(image)[0]
    # Get top-3 indices
    top3_idx = preds.argsort()[-3:][::-1]
    # Build label→confidence dict
    results = { str(i): float(preds[i]) for i in top3_idx }
    return results

# 3. Build the Gradio interface
iface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Image(
        # shape=(28,28),
        image_mode='L',
        source='canvas',
        invert_colors=False
        type="numpy"
    ),
    outputs=gr.Label(num_top_classes=3),
    title="MNIST Digit Recognizer",
    description="Draw a digit below (or upload); I’ll show my top 3 guesses."
)


# 4. Launch the app
# if __name__ == "__main__":
#     iface.launch()

if __name__ == "__main__":
    iface.launch(share=True)

