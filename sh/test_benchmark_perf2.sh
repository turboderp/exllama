
echo "-------------------------------------------------------------------------------------------------------------"
python test_benchmark_inference.py -p -d /mnt/str/models/llama-65b-4bit-128g-act -gs 17.2,24
echo "-------------------------------------------------------------------------------------------------------------"
python test_benchmark_inference.py -p -d /mnt/str/models/llama-65b-4bit-32g-act -gs 17.2,24
echo "-------------------------------------------------------------------------------------------------------------"
