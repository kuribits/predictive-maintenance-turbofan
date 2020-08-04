echo "Cloning original Turbofan POC repository..."
git clone https://github.com/matthiaslau/Turbofan-Federated-Learning-POC.git
echo "Merging improved files into cloned repository..."
rsync -avzhP federated-trainer Turbofan-Federated-Learning-POC/federated-trainer
echo "Downloading and preprocessing data..."
cd ./Turbofan-Federated-Learning-POC && python data_preprocessor.py --worker_count 1
echo "Initialisation done."
