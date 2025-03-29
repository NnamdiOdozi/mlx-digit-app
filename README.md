
This is a web app that recognises handwritten digits between 0 and 9.  The prediction is made using a Convolutional Neural Network (CNN) trained on the well-known MNIST database.

The user draws a digit on the slate and then presses the predict button.  If the app predicts wrong, the user types in the correct number and presses the submit feedback button. To try another digit, click on the delete icon below the slate and draw another digit as before.  A history of previous digits drawn by the user is shown in a table below the slate together with a running prediction accuracy of the app. However this table is lost once the app is refreshed. Persistent storage of past predictions is done in a Postgres database hosted on the same web server,  

Try it out here: URL: http://138.199.200.113:8501


## Project Structure

MLX_PROJECT/
├── app/
│   ├── main.py               # Contains UI + model prediction logic
│   ├── CNNModelMNIST.py      # PyTorch model definition
│   ├── model.pth             # Saved model weights
│   ├── Dockerfile            # For Streamlit app
│   └── ...
│
├── db/                       # Optional Postgres init scripts
│   └── init.sql
│
├── docker-compose.yml        # Streamlit app + Postgres
├── requirements.txt          # App + ML dependencies
├── .env                      # DB and port settings
└── README.md



## Data


## Model Architecture
 - File
- ![image](https://github.com/user-attachments/assets/69745b30-4741-4dc2-8dd4-614bbcf26b06)


## Training
 - In order to make use of GPUs and so speed up training, model was run in a Google Colab Notebook
## Model Evaluation
 - File

![image](https://github.com/user-attachments/assets/b52b9aa2-f333-4744-9858-90f60bd6d844)

![image](https://github.com/user-attachments/assets/774a7021-7221-4332-a1bf-40490dc6136f)



## Inference
File

## 🧠 Model Weights

This repo includes a pre-trained CNN for MNIST digit recognition:

- `app/mnist_cnn.pth` — weights for a small CNN trained on the MNIST dataset

You can retrain your own using `CNNModelMNIST.py` or swap in a different `.pth`.

## Deployment
  2 Docker containers
  Hosted on Hetzner VPS Instance
 - Streamlit
 - PostGres DB

## 🗃️ Database Init

On first run, the `init.sql` script (in `db/init.sql`) sets up the PostgreSQL schema for logging predictions.

Docker Compose handles this automatically.


## Ideas for future
 - trying out different models eg LeNet5, Vision Transformers, LLMs like ChatGPT
 - Splitting out the model from the Streamlit app so that there are 3 containers instead of 2
