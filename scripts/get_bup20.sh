echo Creating datasets path...

mkdir -p datasets
cd datasets

echo Downloading BUP_20 dataset...
wget -O BUP20.tar.gz -c https://uni-bonn.sciebo.de/s/dbETJWamSqyCYm5/download

echo Extracting dataset...
tar -xf BUP20.tar.gz --checkpoint=.10000

cd ../..