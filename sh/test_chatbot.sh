
python test_chatbot.py -d /mnt/str/models/wizardlm-30b-uncensored-4bit-act-order/ -un "Maxine" -p prompt_assistant.txt -nnl -temp 1.0 -topp .75

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