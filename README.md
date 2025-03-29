
This is a web app that recognises handwritten digits between 0 and 9.  The prediction is made using a Convolutional Neural Network (CNN) trained on the MNIST database

Try it out here: URL: http://138.199.200.113:8501



## 🧠 Model Weights

This repo includes a pre-trained CNN for MNIST digit recognition:

- `app/mnist_cnn.pth` — weights for a small CNN trained on the MNIST dataset

You can retrain your own using `CNNModelMNIST.py` or swap in a different `.pth`.

---

## 🗃️ Database Init

On first run, the `init.sql` script (in `db/init.sql`) sets up the PostgreSQL schema for logging predictions.

Docker Compose handles this automatically.
