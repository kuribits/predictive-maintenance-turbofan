echo "Cloning original Turbofan POC repository..."
git clone https://github.com/matthiaslau/Turbofan-Federated-Learning-POC.git
echo "Merging improved files into cloned repository..."
echo "Downloading and preprocessing data..."
mv Turbofan-Federated-Learning-POC turbofanpoc
cd turbofanpoc && python data_preprocessor.py --worker_count 1
echo "Initialisation done."
