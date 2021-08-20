mkdir -p ./damd_multiwoz/data/embeddings
wget -O ./damd_multiwoz/data/embeddings/glove.6B.zip https://nlp.stanford.edu/data/glove.6B.zip
unzip ./damd_multiwoz/data/embeddings/glove.6B.zip -d ./damd_multiwoz/data/embeddings
echo "400000 100" | cat - ./damd_multiwoz/data/embeddings/glove.6B.100d.txt > ./damd_multiwoz/data/embeddings/glove.6B.100d.w2v.txt
rm -rf ./damd_multiwoz/data/embeddings/glove.6B.50d.txt ./damd_multiwoz/data/embeddings/glove.6B.100d.txt ./damd_multiwoz/data/embeddings/glove.6B.200d.txt ./damd_multiwoz/data/embeddings/glove.6B.300d.txt ./damd_multiwoz/data/embeddings/glove.6B.zip
mkdir -p ./damd_multiwoz/data/multi-woz-oppe/reward

python -m spacy download en_core_web_sm

./setup.sh