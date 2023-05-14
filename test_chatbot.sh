
python test_chatbot.py \
-t /mnt/Fast/models/llama-30b-4bit-128g/tokenizer.model \
-c /mnt/Fast/models/llama-30b-4bit-128g/config.json \
-m /mnt/Fast/models/llama-30b-4bit-128g/llama-30b-4bit-128g.safetensors \
-g 128 \
-un "Jeff" \
-p prompt_chatbort.txt

#python test_chatbot.py \
#-t /mnt/Fast/models/bluemoon-4k-13b-4bit-128g/tokenizer.model \
#-c /mnt/Fast/models/bluemoon-4k-13b-4bit-128g/config.json \
#-m /mnt/Fast/models/bluemoon-4k-13b-4bit-128g/bluemoonrp-13b-4k-epoch6-4bit-128g.safetensors \
#-g 128 \
#-p prompt_bluemoon.txt \
#-un "Player" \
#-bn "DM" \
#-bf \
#-topk 30 \
#-topp 0.45 \
#-minp 0.1 \
#-temp 1.4 \
#-repp 1.3 \
#-repps 256 \
#-l 4096