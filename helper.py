import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model
import tensorflow as tf
import keras
import matplotlib.cm as cm
import cv2
class Helper(object):

    def classification(self, img_path):
        model = load_model("models/model7.h5")
        image = load_img("static/uploaded_img/" + img_path, target_size=(224, 224), color_mode="grayscale")
        last_conv_layer_name = "block5_conv3"
        img = img_to_array(image)
        img = np.expand_dims(img, axis=0)
        img = img.astype('float32') / 255
        img = np.repeat(img, 3, axis=-1)

        preds = model.predict(img)
        i = np.argmax(preds, axis=1)

        classes = ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"]
        max_probabilities = np.max(preds, axis=1)
        predictions = [(classes[i], round(probability * 100, 2)) for i, probability in zip(i, max_probabilities)]
        prediction, prob = predictions[0]

        heatmap = self.make_heatmap(img, model, last_conv_layer_name)
        self.save_gradcam(img_path, heatmap)

        return prediction, prob

    def make_heatmap(self, img_array, model, last_conv_layer_name, pred_index=None):
        grad_model = keras.models.Model(
            model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

        threshold = 0.4
        heatmap = np.where(heatmap < threshold, 0, heatmap)
        heatmap = (heatmap - threshold) / (1 - threshold)
        heatmap = np.clip(heatmap, 0, 1)

        return heatmap

    def save_gradcam(self, filename, heatmap, alpha=1):
        # Load the original image
        img = keras.utils.load_img("static/uploaded_img/" + filename)
        img = keras.utils.img_to_array(img)

        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)

        # Use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # Create an image with RGB colorized heatmap
        jet_heatmap = keras.utils.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = keras.utils.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = keras.utils.array_to_img(superimposed_img)

        # Save the superimposed image
        superimposed_img.save("static/grad_cam/" + filename)
