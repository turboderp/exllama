
echo "-------------------------------------------------------------------------------------------------------------"
echo "-- New"
echo "-------------------------------------------------------------------------------------------------------------"

python test_benchmark_inference.py \
-t /mnt/str/models/llama-7b-4bit-128g/tokenizer.model \
-c /mnt/str/models/llama-7b-4bit-128g/config.json \
-m /mnt/str/models/llama-7b-4bit-128g/llama-7b-4bit-128g.safetensors \
-p

echo "-----------------------------------------"

python test_benchmark_inference.py \
-t /mnt/str/models/llama-13b-4bit-128g/tokenizer.model \
-c /mnt/str/models/llama-13b-4bit-128g/config.json \
-m /mnt/str/models/llama-13b-4bit-128g/llama-13b-4bit-128g.safetensors \
-p

echo "-----------------------------------------"

python test_benchmark_inference.py \
-t /mnt/str/models/llama-30b-4bit-128g/tokenizer.model \
-c /mnt/str/models/llama-30b-4bit-128g/config.json \
-m /mnt/str/models/llama-30b-4bit-128g/llama-30b-4bit-128g.safetensors \
-p

echo "-----------------------------------------"

python test_benchmark_inference.py \
-t /mnt/str/models/llama-30b-4bit-128g-act/tokenizer.model \
-c /mnt/str/models/llama-30b-4bit-128g-act/config.json \
-m /mnt/str/models/llama-30b-4bit-128g-act/llama-30b-4bit-128g.safetensors \
-p

echo "-----------------------------------------"

python test_benchmark_inference.py \
-t /mnt/str/models/llama-30b-4bit-32g-act-ts/tokenizer.model \
-c /mnt/str/models/llama-30b-4bit-32g-act-ts/config.json \
-m /mnt/str/models/llama-30b-4bit-32g-act-ts/llama-30b-4bit-32g.safetensors \
-l 1550 \
-p

echo "-----------------------------------------"

python test_benchmark_inference.py \
-t /mnt/str/models/koala-13B-4bit-128g-act/tokenizer.model \
-c /mnt/str/models/koala-13B-4bit-128g-act/config.json \
-m /mnt/str/models/koala-13B-4bit-128g-act/koala-13B-4bit-128g.safetensors \
-p
