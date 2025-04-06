
This is a web app that recognises handwritten digits between 0 and 9.  The prediction is made using a Convolutional Neural Network (CNN) trained on the well-known MNIST database.

The user draws a digit on the slate and then presses the predict button.  If the app predicts wrong, the user types in the correct number and presses the submit feedback button. To try another digit, click on the delete icon below the slate and draw another digit as before.  A history of previous digits drawn by the user is shown in a table below the slate together with a running prediction accuracy of the app. However this table is lost once the app is refreshed. Persistent storage of past predictions is done in a Postgres database hosted on the same web server,  

Try it out here: URL: http://138.199.200.113:8501

![image](https://github.com/user-attachments/assets/fafc085f-82c0-4bb0-805d-01b5934e387d)



## Project Structure
![image](https://github.com/user-attachments/assets/c1578f63-b976-426c-9739-3267c0a1a002)

![image](https://github.com/user-attachments/assets/d586ff46-ba9a-4b70-99ae-42d2b03a41c1)


![image](https://github.com/user-attachments/assets/ae7f3438-d2ff-429c-aefb-c40bbeb3168d)


## Data
The MNIST dataset was used

Preprocessing:

## Model Architecture
 - 
- ![image](https://github.com/user-attachments/assets/69745b30-4741-4dc2-8dd4-614bbcf26b06)
  

## Training
 - In order to make use of GPUs and so speed up training, model was run in a Google Colab Notebook, Model was run for 10 epochs. Othe training details including hyper-parameters are as per the screenshot below:

 - ![image](https://github.com/user-attachments/assets/4a8ccb3d-c03f-4322-9241-ca66b3819682)

   
## Model Evaluation
 - 
Model accuracy of over 90% was achieved on the MNIST test dataset.
![image](https://github.com/user-attachments/assets/b52b9aa2-f333-4744-9858-90f60bd6d844)

![image](https://github.com/user-attachments/assets/774a7021-7221-4332-a1bf-40490dc6136f)



## Inference
File

## 🧠 Model Weights

This repo includes a pre-trained CNN for MNIST digit recognition:

- `app/mnist_cnn.pth` — weights for a small CNN trained on the MNIST dataset

You can retrain your own using `CNNModelMNIST.py` or swap in a different `.pth`.

## Deployment
  Two Docker containers were used as per the project structure.  One for the PyTorch Model and Streamlit App and the second for the initialisiation of the Postgres DB
  Containers were built and hosted on Hetzner VPS Instance
  Git Hub actions were used to push any changes to the app folder automatically to the VPS using ssh login and there to re-build the container
  Docker-compose was used to automatically restart the App anytime the VPS was rebooted
  On first run, the `init.sql` script (in `db/init.sql`) sets up the PostgreSQL schema for logging predictions. Docker Compose handles this automatically.

 The Postgres database table is shown below after over 20 attemopts had been logged

 ![image](https://github.com/user-attachments/assets/279e00a1-b5a3-4bef-9fa4-c37f7370bb12)


## Ideas for future
 - trying out different models eg LeNet-5, Vision Transformers, Capsule Networks, Google Vision API, LLMs like ChatGPT
 - Splitting out the model from the Streamlit app so that there are 3 containers instead of 2
