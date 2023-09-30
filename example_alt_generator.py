from model import ExLlama, ExLlamaCache, ExLlamaConfig
from lora import ExLlamaLora
from tokenizer import ExLlamaTokenizer
from alt_generator import ExLlamaAltGenerator
import argparse
import torch
import sys
import os
import glob
import model_init
import time


config: ExLlamaConfig           # Config for the model, loaded from config.json
model: ExLlama                  # Model, initialized with ExLlamaConfig
cache: ExLlamaCache             # Cache for generation, tied to model
generator: ExLlamaAltGenerator  # Generator
tokenizer: ExLlamaTokenizer     # Tokenizer
lora: ExLlamaLora = None        # (Optional) LoRA, remember to specify in generator settings


# Initialize model

def init_explicit():
    global model, cache, config, generator, tokenizer, lora

    # Directory containing model, tokenizer

    model_directory = "/mnt/str/models/llama-7b-4bit-128g/"

    # Locate files we need within that directory

    tokenizer_path = os.path.join(model_directory, "tokenizer.model")
    model_config_path = os.path.join(model_directory, "config.json")
    st_pattern = os.path.join(model_directory, "*.safetensors")
    model_path = glob.glob(st_pattern)[0]

    # Create config, model, tokenizer and generator

    config = ExLlamaConfig(model_config_path)                   # create config from config.json
    config.model_path = model_path                              # supply path to model weights file

    model = ExLlama(config)                                     # create ExLlama instance and load the weights
    tokenizer = ExLlamaTokenizer(tokenizer_path)                # create tokenizer from tokenizer model file

    cache = ExLlamaCache(model)                                 # create cache for inference
    generator = ExLlamaAltGenerator(model, tokenizer, cache)    # create generator

    # Load LoRA

    lora_dir = None

    if lora_dir is not None:
        lora_config = os.path.join(lora_dir, "adapter_config.json")
        lora = os.path.join(lora_dir, "adapter_model.bin")
        lora = ExLlamaLora(model, lora_config, lora)


# Alternatively, initialize from command line args

def init_args():
    global model, cache, config, generator, tokenizer, lora

    # Global initialization

    torch.set_grad_enabled(False)
    torch.cuda._lazy_init()

    # Parse arguments

    parser = argparse.ArgumentParser(description = "Generator example")

    model_init.add_args(parser)

    parser.add_argument("-lora", "--lora", type = str, help = "Path to LoRA binary to use during benchmark")
    parser.add_argument("-loracfg", "--lora_config", type = str, help = "Path to LoRA config to use during benchmark")
    parser.add_argument("-ld", "--lora_dir", type = str, help = "Path to LoRA config and binary. to use during benchmark")

    args = parser.parse_args()
    model_init.post_parse(args)
    model_init.get_model_files(args)

    print_opts = []
    model_init.print_options(args, print_opts)

    # Paths

    if args.lora_dir is not None:
        args.lora_config = os.path.join(args.lora_dir, "adapter_config.json")
        args.lora = os.path.join(args.lora_dir, "adapter_model.bin")

    # Model globals

    model_init.set_globals(args)

    # Instantiate model and generator

    config = model_init.make_config(args)

    model = ExLlama(config)
    cache = ExLlamaCache(model)
    tokenizer = ExLlamaTokenizer(args.tokenizer)

    model_init.print_stats(model)

    # Load LoRA

    lora = None
    if args.lora:
        print(f" -- LoRA config: {args.lora_config}")
        print(f" -- Loading LoRA: {args.lora}")
        if args.lora_config is None:
            print(f" ## Error: please specify lora path to adapter_config.json")
            sys.exit()
        lora = ExLlamaLora(model, args.lora_config, args.lora)
        if lora.bias_ignored:
            print(f" !! Warning: LoRA zero bias ignored")

    # Generator

    generator = ExLlamaAltGenerator(model, tokenizer, cache)


# Intialize

# init_args()
init_explicit()

# Example one-shot generation

settings = ExLlamaAltGenerator.Settings()
settings.temperature = 0.75
settings.top_p = 0.8

prompt = "A bird in the hand is worth"

stop_conditions = [".", "!", "bush", tokenizer.newline_token_id]

output = generator.generate(prompt = prompt,
                            stop_conditions = stop_conditions,
                            max_new_tokens = 50,
                            gen_settings = settings)

print()
print(prompt + output)
print()


# Example of (implicit) cache reuse

context = """Albert Einstein (/ˈaɪnstaɪn/ EYEN-styne;[4] German: [ˈalbɛʁt ˈʔaɪnʃtaɪn] (listen); 14 March 1879 – 18 April 1955) was a German-born theoretical physicist,[5] widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century.[1][6] His mass–energy equivalence formula E = mc2, which arises from relativity theory, has been called "the world's most famous equation".[7] He received the 1921 Nobel Prize in Physics "for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect",[8] a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science.[9][10] In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time.[11] His intellectual achievements and originality have made Einstein synonymous with genius.[12]
In 1905, a year sometimes described as his annus mirabilis (miracle year), Einstein published four groundbreaking papers.[13] These outlined a theory of the photoelectric effect, explained Brownian motion, introduced his special theory of relativity—a theory which addressed the inability of classical mechanics to account satisfactorily for the behavior of the electromagnetic field—and demonstrated that if the special theory is correct, mass and energy are equivalent to each other. In 1915, he proposed a general theory of relativity that extended his system of mechanics to incorporate gravitation. A cosmological paper that he published the following year laid out the implications of general relativity for the modeling of the structure and evolution of the universe as a whole.[14][15] The middle part of his career also saw him making important contributions to statistical mechanics and quantum theory. Especially notable was his work on the quantum physics of radiation, in which light consists of particles, subsequently called photons.
For much of the last phase of his academic life, Einstein worked on two endeavors that proved ultimately unsuccessful. Firstly, he fought a long rearguard action against quantum theory's introduction of fundamental randomness into science's picture of the world, objecting that "God does not play dice".[16] Secondly, he attempted to devise a unified field theory by generalizing his geometric theory of gravitation to include electromagnetism too. As a result, he became increasingly isolated from the mainstream of modern physics.
Born in the German Empire, Einstein moved to Switzerland in 1895, forsaking his German citizenship (as a subject of the Kingdom of Württemberg)[note 1] the following year. In 1897, at the age of seventeen, he enrolled in the mathematics and physics teaching diploma program at the Swiss Federal polytechnic school in Zürich, graduating in 1900. In 1901, he acquired Swiss citizenship, which he kept for the rest of his life. In 1903, he secured a permanent position at the Swiss Patent Office in Bern. In 1905, he submitted a successful PhD dissertation to the University of Zurich. In 1914, he moved to Berlin in order to join the Prussian Academy of Sciences and the Humboldt University of Berlin. In 1917, he became director of the Kaiser Wilhelm Institute for Physics; he also became a German citizen again, this time as a subject of the Kingdom of Prussia.[note 1] In 1933, while he was visiting the United States, Adolf Hitler came to power in Germany. Alienated by the policies of the newly elected Nazi government,[17] Einstein decided to remain in the US, and was granted American citizenship in 1940.[18] On the eve of World War II, he endorsed a letter to President Franklin D. Roosevelt alerting him to the potential German nuclear weapons program and recommending that the US begin similar research. Einstein supported the Allies but generally viewed the idea of nuclear weapons with great dismay.[19]
Albert Einstein was born in Ulm,[5] in the Kingdom of Württemberg in the German Empire, on 14 March 1879.[20][21] His parents, secular Ashkenazi Jews, were Hermann Einstein, a salesman and engineer, and Pauline Koch. In 1880, the family moved to Munich, where Einstein's father and his uncle Jakob founded Elektrotechnische Fabrik J. Einstein & Cie, a company that manufactured electrical equipment based on direct current.[5]
Albert attended a Catholic elementary school in Munich from the age of five. When he was eight, he was transferred to the Luitpold-Gymnasium (now known as the Albert-Einstein-Gymnasium [de]), where he received advanced primary and then secondary school education.[22]
In 1894, Hermann and Jakob's company tendered for a contract to install electric lighting in Munich, but without success—they lacked the capital that would have been required to update their technology from direct current to the more efficient, alternating current alternative.[23] The failure of their bid forced them to sell their Munich factory and search for new opportunities elsewhere. The Einstein family moved to Italy, first to Milan and a few months later to Pavia, where they settled in Palazzo Cornazzani, a medieval building which, at different times, had been the home of Ugo Foscolo, Contardo Ferrini and Ada Negri.[24] Einstein, then fifteen, stayed behind in Munich in order to finish his schooling. His father wanted him to study electrical engineering, but he was a fractious pupil who found the Gymnasium's regimen and teaching methods far from congenial. He later wrote that the school's policy of strict rote learning was harmful to creativity. At the end of December 1894, a letter from a doctor persuaded the Luitpold's authorities to release him from its care, and he joined his family in Pavia.[25] While in Italy as a teenager, he wrote an essay entitled "On the Investigation of the State of the Ether in a Magnetic Field".[26][27]
Einstein excelled at physics and mathematics from an early age, and soon acquired the mathematical expertise normally only found in a child several years his senior. He began teaching himself algebra, calculus and Euclidean geometry when he was twelve; he made such rapid progress that he discovered an original proof of the Pythagorean theorem before his thirteenth birthday.[28][29][30] A family tutor, Max Talmud, said that only a short time after he had given the twelve year old Einstein a geometry textbook, the boy "had worked through the whole book. He thereupon devoted himself to higher mathematics ... Soon the flight of his mathematical genius was so high I could not follow."[31] Einstein recorded that he had "mastered integral and differential calculus" while still just fourteen.[29] His love of algebra and geometry was so great that at twelve, he was already confident that nature could be understood as a "mathematical structure".[31]
At thirteen, when his range of enthusiasms had broadened to include music and philosophy,[32] Einstein was introduced to Kant's Critique of Pure Reason. Kant became his favorite philosopher; according to his tutor, "At the time he was still a child, only thirteen years old, yet Kant's works, incomprehensible to ordinary mortals, seemed to be clear to him."[31]"""

def timer(func):
    t = time.time()
    ret = func()
    t = time.time() - t
    return ret, t

settings = ExLlamaAltGenerator.Settings()
settings.temperature = 0.95
settings.top_k = 80
settings.typical = 0.8

questions = ["When was Albert Einstein born?",
             "How many groundbreaking papers did Einstein publish in 1905?",
             "Where did Einstein move in 1895?",
             "When did Einstein graduate?"]

stop_conditions = [tokenizer.newline_token_id, tokenizer.eos_token_id]

for question in questions:
    output, t = timer(lambda: generator.generate(context + "\nQ: " + question + "\nA:", stop_conditions, 100, settings))
    print(f"Generated in {t:.3f} seconds: {question} -> {output}")


# Streaming example

settings = ExLlamaAltGenerator.Settings()
settings.temperature = 1.00
settings.top_k = 80
settings.top_p = 0.9
settings.lora = lora

prompt = "Our story begins in the town of Auchtermuchty, where once"

print()
print(prompt, end = "")
sys.stdout.flush()

output = generator.begin_stream(prompt = prompt,
                                stop_conditions = [],
                                max_new_tokens = 1000,
                                gen_settings = settings)

while True:
    chunk, eos = generator.stream()
    print(chunk, end = "")
    sys.stdout.flush()
    if eos: break
