echo "-------------------------------------------------------------------------------------------------------------"
echo "-- Reference"
echo "-------------------------------------------------------------------------------------------------------------"

python test_benchmark_inference.py \
-t /mnt/Fast/models/llama-7b-4bit-128g/ \
-c /mnt/Fast/models/llama-7b-4bit-128g/config.json \
-m /mnt/Fast/models/llama-7b-4bit-128g/llama-7b-4bit-128g.safetensors \
-g 128 \
-o \
-p

echo "-----------------------------------------"

python test_benchmark_inference.py \
-t /mnt/Fast/models/llama-13b-4bit-128g/ \
-c /mnt/Fast/models/llama-13b-4bit-128g/config.json \
-m /mnt/Fast/models/llama-13b-4bit-128g/llama-13b-4bit-128g.safetensors \
-g 128 \
-o \
-p

echo "-----------------------------------------"

python test_benchmark_inference.py \
-t /mnt/Fast/models/llama-30b-4bit-128g/ \
-c /mnt/Fast/models/llama-30b-4bit-128g/config.json \
-m /mnt/Fast/models/llama-30b-4bit-128g/llama-30b-4bit-128g.safetensors \
-g 128 \
-l 256 \
-o \
-p

echo "-------------------------------------------------------------------------------------------------------------"
echo "-- Reference, 16 bit"
echo "-------------------------------------------------------------------------------------------------------------"

python test_benchmark_inference.py \
-t /mnt/Fast/models/llama-7b-4bit-128g/ \
-c /mnt/Fast/models/llama-7b-4bit-128g/config.json \
-m /mnt/Fast/models/llama-7b-4bit-128g/llama-7b-4bit-128g.safetensors \
-g 128 \
-o \
-half \
-p

echo "-----------------------------------------"

python test_benchmark_inference.py \
-t /mnt/Fast/models/llama-13b-4bit-128g/ \
-c /mnt/Fast/models/llama-13b-4bit-128g/config.json \
-m /mnt/Fast/models/llama-13b-4bit-128g/llama-13b-4bit-128g.safetensors \
-g 128 \
-o \
-half \
-p

echo "-----------------------------------------"

python test_benchmark_inference.py \
-t /mnt/Fast/models/llama-30b-4bit-128g/ \
-c /mnt/Fast/models/llama-30b-4bit-128g/config.json \
-m /mnt/Fast/models/llama-30b-4bit-128g/llama-30b-4bit-128g.safetensors \
-g 128 \
-l 1024 \
-o \
-half \
-p

echo "-------------------------------------------------------------------------------------------------------------"
echo "-- New"
echo "-------------------------------------------------------------------------------------------------------------"

python test_benchmark_inference.py \
-t /mnt/Fast/models/llama-7b-4bit-128g/ \
-c /mnt/Fast/models/llama-7b-4bit-128g/config.json \
-m /mnt/Fast/models/llama-7b-4bit-128g/llama-7b-4bit-128g.safetensors \
-g 128 \
-p

echo "-----------------------------------------"

python test_benchmark_inference.py \
-t /mnt/Fast/models/llama-13b-4bit-128g/ \
-c /mnt/Fast/models/llama-13b-4bit-128g/config.json \
-m /mnt/Fast/models/llama-13b-4bit-128g/llama-13b-4bit-128g.safetensors \
-g 128 \
-p

echo "-----------------------------------------"

python test_benchmark_inference.py \
-t /mnt/Fast/models/llama-30b-4bit-128g/ \
-c /mnt/Fast/models/llama-30b-4bit-128g/config.json \
-m /mnt/Fast/models/llama-30b-4bit-128g/llama-30b-4bit-128g.safetensors \
-g 128 \
-mm quant_only \
-p

echo "-----------------------------------------"

python test_benchmark_inference.py \
-t /mnt/Fast/models/llama-30b-4bit-128g/ \
-c /mnt/Fast/models/llama-30b-4bit-128g/config.json \
-m /mnt/Fast/models/llama-30b-4bit-128g/llama-30b-4bit-128g.safetensors \
-g 128 \
-l 1700 \
-p
