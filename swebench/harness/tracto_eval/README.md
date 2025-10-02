# Evaluation at scale using TractoAI

[TractoAI](https://tracto.ai) is a unified compute platform for AI and data workloads.

This folder provides support of TractoAI as a scalable drop-in backend for SWE-bench eval. It can be used to evaluate any SWE-bench-compatible tasks, including SWE-rebench -- the only requirement is pre-built Docker images.

Unlike local Docker backend, TractoAI scales horizontally with the cluster size and is able to evaluate hundreds of patches in parallel.

## How to try

### Clone & install SWE-rebench/SWE-bench-fork

```bash

# clone
git clone git@github.com:SWE-rebench/SWE-bench-fork.git
cd SWE-bench-fork

# create Python environment for swe-rebench, here conda is used
conda create --name swe-rebench python=3.10
conda activate swe-rebench

# install project in edit mode
pip install -e .
```

### Set up TractoAI

Please see a dedicated [section](#tractoai-cluster-setup) below. Make sure that environment variables `YT_PROXY` and `YT_TOKEN` are set.

Run `yt whoami` to test your configuration.

Also, set this env var for convenience later:
```bash
export TRACTO_TENANT=<<tenant>>
```

### Build & push image for SWE-rebench-fork

TractoAI uses Docker containers to run operations in remote environments. TractoAI also comes with a built-in Docker registry, that is located closely to compute nodes and ensures minimal image pull duration and lack of rate limits.

Log in to your tenant's Tracto registry:

```bash
export TRACTO_REGISTRY_URL="cr.$TRACTO_TENANT.trtr.ai/home/$TRACTO_TENANT/registry"
echo $YT_TOKEN | docker login $TRACTO_REGISTRY_URL --username $(yt whoami) --password-stdin
```

Build & push your image for SWE-bench-fork:

```bash
export TRACTO_EVAL_IMAGE="$TRACTO_REGISTRY_URL/swe-bench-fork:$(date -Idate)"

docker build -t $TRACTO_EVAL_IMAGE --push -f docker/Dockerfile --platform linux/amd64 .
```

Once pushed, you can see your image on TractoAI's UI.


### Import SWE instance images into Tracto registry

Many SWE benchmarks ship pre-built Docker images for SWE instances on DockerHub, for example [SWE-rebench](https://hub.docker.com/u/swerebench) and [SWE-bench](https://hub.docker.com/u/swebench).

Although it's possible to run SWE-bench eval on Tracto using images from DockerHub, **it's highly recommended to import the images into Tracto registry first**. This is a one-time-operation that provides better pull performance and eliminates the impact of rate-limits from DockerHub during evals.

We provide a script to run import of images in parallel on top of TractoAI: `swebench/harness/tracto_eval/import_images_to_tracto.py`.

Example: import images for SWE-rebench Leaderboard, 449 images as of September 2025:

```bash
python -m swebench.harness.tracto_eval.import_images_to_tracto \
    --dataset_name nebius/SWE-rebench-leaderboard  \
    --namespace docker.io/swerebench \
    --tracto-namespace $TRACTO_REGISTRY_URL/swerebench
```

Example: import images for SWE-bench Verified, 500 images:

```bash
python -m swebench.harness.tracto_eval.import_images_to_tracto \
    --dataset_name SWE-bench/SWE-bench_Verified  \
    --namespace docker.io/swebench \
    --tracto-namespace $TRACTO_REGISTRY_URL/swebench
```

### Run evaluation using Tracto

Once SWE-bench-fork is built and SWE instance images are imported to Tracto registry, we are good to run evaluations on Tracto.

The same `swebench/harness/run_evaluation.py` can be used, just pass `--tracto yes` to use Tracto as backend.

Example: run evaluation on SWE-rebench Leaderboard:

```bash
TRACTO_EVAL_IMAGE=$TRACTO_EVAL_IMAGE \
TRACTO_EVAL_RUNS_DIR=//home/$TRACTO_TENANT/evals \
python -m swebench.harness.run_evaluation \
    --dataset_name nebius/SWE-rebench-leaderboard \
    --predictions_path gold \
    --cache_level instance \
    --run_id validate-gold \
    --namespace "$TRACTO_REGISTRY_URL/swerebench"
```

## Feedback

Feel free to create a GitHub issue if you have any problems or ideas.

In case of issues with TractoAI, their [Discord](https://discord.gg/KFpaDAFQrP) server can be useful.

# TractoAI cluster setup

Follow https://console.tracto.ai/login to set up your first TractoAI cluster:
1. Log in with your Google account and create your tenant, tenant name will be used in cluster URL later, i.e. `https://<<tenant>>.trtr.ai`
2. Verify your e-mail. Check your mailbox for a "Verify your email address" e-mail from TractoAI team. Once your e-mail is verified, your tenant will automatically proceed from `Waiting for confirmation` to `Running`.
   1. Fresh users get 14 days trial with 10$ in credits -- no payment details required.
   2. To share access to the tenant, just send an invite via "Invites" tab.
3. Open your TractoAI cluster via button `Go to tenant UI` or just enter `https://<<tenant>>/trtr.ai` directly.
4. Issue your credentials:
   1. Click on your avatar in the bottom-left corner of UI.
   2. Select "Manage Tokens".
   3. Click on "Generate".
   4. You'll receive token and configuration snippet.
   5. Copy and paste this snippet into your `~/.zshrc` / `~/.bashrc`

In the end, you should have lines like these in your `~/.zshrc` / `~/.bashrc`:

```bash
export YT_TOKEN='ytct-<<masked>>'
export YT_PROXY='https://<<tenant>>.trtr.ai'
export YT_CONFIG_PATCHES=...
```

Make sure to run `source ~/.zshrc` / `source ~/.bashrc`.

Test your TractoAI setup:
1. Install TractoAI libraries for Python: `pip install ytsaurus-client ytsaurus-yson`.
2. Run `yt whoami` -- this command should display your username.

🎉 Congratulations! Your TractoAI cluster is ready to use.

You can find more details about [tenants](https://docs.tracto.ai/overview/console/#tenants) and [setup](https://docs.tracto.ai/overview/ytsaurus/setup/) in TractoAI docs.
