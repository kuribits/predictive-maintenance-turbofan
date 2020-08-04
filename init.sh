echo "Cloning original Turbofan POC repository..."
git clone https://github.com/matthiaslau/Turbofan-Federated-Learning-POC.git
echo "Downloading and preprocessing data..."
cd ./Turbofan-Federated-Learning-POC && python data_preprocessor.py --worker_count 1
echo "Initialisation done."
