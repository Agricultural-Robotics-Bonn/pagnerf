echo Creating datasets path...

mkdir -p datasets
cd datasets

echo Downloading BUP_20 Mask2Former detections...
wget -O BUP20_m2f.tar.gz -c https://uni-bonn.sciebo.de/s/r0jbAeQZLCCtiys/download

echo Extracting BUP_20 Mask2Former detections...
tar -xf BUP20_m2f.tar.gz --checkpoint=.10000

cd ..
