
python test_chatbot.py \
-t /mnt/str/models/llama-30b-4bit-128g-act/tokenizer.model \
-c /mnt/str/models/llama-30b-4bit-128g-act/config.json \
-m /mnt/str/models/llama-30b-4bit-128g-act/llama-30b-4bit-128g.safetensors \
-un "Jeff" \
-beams 8 -beamlen 8 \
-p prompt_chatbort.txt

#python test_chatbot.py \
#-t /mnt/str/models/bluemoon-4k-13b-4bit-128g/tokenizer.model \
#-c /mnt/str/models/bluemoon-4k-13b-4bit-128g/config.json \
#-m /mnt/str/models/bluemoon-4k-13b-4bit-128g/bluemoonrp-13b-4k-epoch6-4bit-128g.safetensors \
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