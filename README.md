# ExLlama

A standalone Python/C++/CUDA implementation of Llama for use with 4-bit GPTQ weights, designed to be fast and
memory-efficient on modern GPUs.

Disclaimer: The project is coming along, but it's still a work in progress!

## Hardware requirements

I am developing on an RTX 4090 and an RTX 3090-Ti. Both cards support the CUDA kernels, but there might be
incompatibilities with older cards.

## Dependencies

* Python 3.9 or newer
* `torch` tested on 2.0.1 and 2.1.0 (nightly) with cu118
* `safetensors` 0.3.1
* `sentencepiece`
* `ninja`

Additionally, only for the web UI:

* `flask`
* `waitress`

## Linux/WSL prerequisites

    pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu118

## Windows prerequisites

To run on Windows (without WSL):

1. Install [MSVC 2022](https://visualstudio.microsoft.com/downloads/). You can choose to install the whole `Visual 
Studio 2022` IDE, or alternatively just the `Build Tools for Visual Studio 2022` package (make sure `Desktop
development with C++` is ticked in the installer), it doesn't really matter which.
2. Install the appropriate version of [PyTorch](https://pytorch.org/get-started/locally/), choosing one of the CUDA
versions. I am developing on the nightly build, but the stable version (2.0.1) should also work.
3. Install CUDA Toolkit, ([11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive) and 
[11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive) both seem to work, just make sure to match PyTorch's
Compute Platform version).
4. For best performance, enable Hardware Accelerated GPU Scheduling.

## How to

Install dependencies, clone repo and run benchmark:

    pip install -r requirements.txt

    git clone https://github.com/turboderp/exllama
    cd exllama

    python test_benchmark_inference.py -d <path_to_model_files> -p -ppl

The CUDA extension is loaded at runtime so there's no need to install it separately. It will be compiled on the first
run and cached to `~/.cache/torch_extensions/` which could take a little while. If nothing happens at first, give it
a minute to compile.

Chatbot example:

    python example_chatbot.py -d <path_to_model_files> -un "Jeff" -p prompt_chatbort.txt

## Web UI

I made a simple web UI for it. Like the rest of the project, it's a work in progress. Don't look at the JavaScript,
it was mostly written by ChatGPT and it will haunt your dreams. But it sort of works, and it's kinda fun, especially
multibot mode:

![_screenshot.jpg](doc/_screenshot.jpg)

To run it:

    pip install -r requirements-web.txt

    python webui/app.py -d <path_to_model_files>

Note that sessions are stored in `~/exllama_sessions/`. You can change the location of the sessions storage with `-sd`
if you want.

## Docker
For security benefits and easier deployment, it is also possible to run the web UI in an isolated docker container. Note: the docker image currently only supports NVIDIA GPUs.

### Requirements
- [Docker](https://docs.docker.com/engine/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

It is recommended to run docker in [rootless mode](https://docs.docker.com/engine/security/rootless/).

### Build

The easiest way to build the docker image is using docker compose. First, set the `MODEL_PATH` and `SESSIONS_PATH` variables in the `.env` file to the actual directories on the host. Then run:

```
docker compose build
```

It is also possible to manually build the image:

```
docker build -t exllama-web .
```

NOTE: by default, the service inside the docker container is run by a non-root user. Hence, the ownership of bind-mounted directories (`/data/model` and `/data/exllama_sessions` in the default `docker-compose.yml` file) is changed to this non-root user in the container entrypoint (`entrypoint.sh`). To disable this, set `RUN_UID=0` in the `.env` file if using `docker compose`, or the following command if you manually build the image:

```
docker build -t exllama-web --build-arg RUN_UID=0 .
```

### Run

Using docker compose:

```
docker compose up
```

The web UI can now be accessed on the host at http://localhost:5000.

The configuration can be viewed in `docker-compose.yml` and changed by creating a `docker-compose.override.yml` file.

Run manually: 

```
docker run --gpus all -p 5000:5000 -v <path_to_model_dir>:/data/model/ -v <path_to_session_dir>:/data/exllama_sessions --rm -it exllama-web --host 0.0.0.0:5000
```


## Results so far

### New implementation
| Model    | Size | grpsz | act             | Seq. len.            | VRAM      | Prompt     | Best    | Worst   | Ppl  |
|----------|------|-------|-----------------|----------------------|-----------|------------|---------|---------|------|
| Llama    | 7B   | 128   | no              | 2,048 t              | 5,194 MB  | 13,918 t/s | 173 t/s | 140 t/s | 6.45 |
| Llama    | 13B  | 128   | no              | 2,048 t              | 9,127 MB  | 7,507 t/s  | 102 t/s | 86 t/s  | 5.60 |
| Llama    | 33B  | 128   | no              | 2,048 t              | 20,795 MB | 2,959 t/s  | 47 t/s  | 40 t/s  | 4.60 |
| Llama    | 33B  | 128   | yes             | 2,048 t              | 20,795 MB | 2,784 t/s  | 45 t/s  | 37 t/s  | 4.55 |
| Llama    | 33B  | 32    | yes             | 1,550 t <sup>1</sup> | 21,486 MB | 2,636 t/s  | 41 t/s  | 37 t/s  | 4.52 |
| Koala    | 13B  | 128   | yes             | 2,048 t              | 9,127 MB  | 5,529 t/s  | 93 t/s  | 79 t/s  | 6.73 |
| WizardLM | 33B  | -     | no <sup>2</sup> | 2,048 t              | 20,199 MB | 2,313 t/s  | 47 t/s  | 40 t/s  | 5.75 |

<sup>1</sup> Can not achieve full sequence length without OoM (yet)  
<sup>2</sup> Not quite sure if this is act-order or not. Weights have no group index, at least   

All tests done on stock RTX 4090 / 12900K, running with a desktop environment, with a few other apps also using VRAM.

**"Prompt"** speed is inference over the sequence length listed minus 128 tokens. **"Worst"** is the average speed for
the last 128 tokens of the full context (worst case) and **"Best"** lists the speed for the first 128 tokens in an
empty sequence (best case.)

VRAM usage is as reported by PyTorch and does not include PyTorch's own overhead (CUDA kernels,
internal buffers etc.) This is somewhat unpredictable anyway. Best bet is to just optimize VRAM usage by the model,
probably aiming for 20 GB on a 24 GB GPU to ensure there is room for a desktop environment and all of Torch's
internals.

Perplexity is measured only to verify that the models are working. The dataset used is a particular, small sample from
WikiText, so scores are not necessarily comparable to other Llama benchmarks.

### Dual GPU results

Since many seem to be interested in running 65B models, I can confirm that this works with two 24 GB GPUs. The
following benchmarks are from a 4090 + 3090-Ti with `-gs 17.2,24`:

| Model    | Size | groupsize | act | Seq. len.            | VRAM      | Prompt    | Best   | Worst  | Ppl  |
|----------|------|-----------|-----|----------------------|-----------|-----------|--------|--------|------|
| Llama    | 65B  | 128       | yes | 2,048 t              | 39,804 MB | 1,109 t/s | 20 t/s | 18 t/s | 4.20 |
| Llama    | 65B  | 32        | yes | 2,048 t              | 43,424 MB | 1,037 t/s | 17 t/s | 16 t/s | 4.11 |


### Testing long sequences

The following tests were all done on **33B/65B, 4bit 128g** with various settings, just to test the max sequence length
and get a sense of what can be achieved with different or multiple GPUs right now. Llama goes incoherent generating 
past 2048 tokens anyway, but with some fine-tuning, who knows? Note that these tests were run a while ago and the
speeds are no longer current.

|                        | Size | Seq. len. | VRAM                 | Long seq. | Ind.   | 
|------------------------|------|-----------|----------------------|-----------|--------|
| 4090/24GB              | 33B  | 2,516 t   | 22,145 MB            | 1140 t/s  | 28 t/s |
| 4090/24GB + 3070Ti/8GB | 33B  | 3,932 t   | 22,055 MB + 7,377 MB | 840 t/s   | 22 t/s |
| A6000/48GB (headless)  | 33B  | 9,032 t   | 46,863 MB            | 645 t/s   | 12 t/s |
| A100/80GB (headless)   | 65B  | 9,520 t   | 79,009 MB            | 650 t/s   | 9 t/s  |

## Todo

Moved the todo list [here](doc/TODO.md).  

## Compatibility

I downloaded a whole bunch of GPTQ models to test compatibility. [Here](doc/model_compatibility.md) is the list of models
confirmed to be working right now.

## Recent updates

**2023-06-02**: Web UI is now in a fairly working state. Expect it to be a little scuffed in places. There will be a
rewrite at some point to make the client-side code less seizure-inducing. It has multibot mode, chat rewind and editing
features, sessions, and more. I'm going to build it out with support for instruct prompting and such, in time.

**2023-06-04**: Refactored a whole bunch to move more of the work into the extension, setting up for more tuning
options to come soon and eventually auto tuning. Also optimized a little, for about a 5% speedup.

**2023-06-06**: Some minor optimizations. Also it should now compile the extension more easily and run more seamlessly
on Windows.

**2023-06-09**: Fused most of the self-attention step. More to come. Slight speedup already, but more importantly went
from 69% actual CPU utilization to 37%. This should do a lot to address the bottleneck on CPUs with lower 
single-threaded performance.

**2023-06-10**: Docker support now! And some minor optimizations. Cleaned up the project a bit.

**2023-06-11**: Added some concurrency a couple of places. It's only beneficial on the 4090, on small models where the
cores are somewhat underutilized and the L2 cache can keep up. For the 3090 it's detrimental to performance, so it's
disabled by default. YMMV. Use `-cs` to try it out.

**2023-06-17**: Fixed a nasty bug in the fused attention that was causing slightly incorrect cache states on 13B and
33B models. You definitely want to update.

**2023-06-18**: LoRA support now. Still needs a lot of testing and some optimization, and currently you can't stack
multiple LoRAs during the same inference. There's also no support in the web UI yet.
