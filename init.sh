git clone https://github.com/matthiaslau/Turbofan-Federated-Learning-POC.git
rsync -avzh --remove-source-files src/ Turbofan-Federated-Learning-POC
cd ./Turbofan-Federated-Learning-POC
python data_preprocessor.py --worker_count 1
