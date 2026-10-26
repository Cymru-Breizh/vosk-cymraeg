.PHONY: setup build run clean help

# UID/GID are computed here and passed straight into docker compose — no need
# to write them into .env (the shell environment takes precedence over .env
# in compose interpolation, so stale .env entries are harmless).
USER_ID  := $(shell id -u)
GROUP_ID := $(shell id -g)

# `make` with no target prints the help at the bottom of this file rather than
# starting an image build.
.DEFAULT_GOAL := help

##@ Training environment

# Create .env with HF_TOKEN + HF_USERNAME (UID/GID are handled automatically
# by make)
setup: ## Write .env with HF_TOKEN + HF_USERNAME (prompts if unset)
	@if [ -z "$$HF_TOKEN" ]; then \
		read -p "Enter your HuggingFace token (HF_TOKEN): " token && \
		echo "HF_TOKEN=$$token" > .env; \
	else \
		echo "HF_TOKEN=$$HF_TOKEN" > .env; \
	fi
	@if [ -z "$$HF_USERNAME" ]; then \
		read -p "Enter your HuggingFace username (HF_USERNAME): " username && \
		echo "HF_USERNAME=$$username" >> .env; \
	else \
		echo "HF_USERNAME=$$HF_USERNAME" >> .env; \
	fi
	@echo "Created .env with HF_TOKEN and HF_USERNAME (add DEVICE=<gpu-id> etc. by hand if needed)"

# Build the Docker image
build: ## Build the training image (>20 min first time)
	@[ -f .env ] || $(MAKE) setup
	@UID=$(USER_ID) GID=$(GROUP_ID) docker compose build

# Run the training environment
run: ## Start the training environment (interactive shell)
	@[ -f .env ] || $(MAKE) setup
	@UID=$(USER_ID) GID=$(GROUP_ID) docker compose run --remove-orphans training_env

##@ Housekeeping

# Stop and remove all Kaldi containers and images
clean: ## Stop/remove all Kaldi containers and images
	@echo "Stopping and removing Kaldi containers..."
	-docker compose down --remove-orphans
	-docker ps -a --filter "name=kaldi" -q | xargs -r docker stop
	-docker ps -a --filter "name=kaldi" -q | xargs -r docker rm
	@echo "Removing Kaldi images..."
	-docker images --filter "reference=*kaldi*" -q | xargs -r docker rmi -f
	@echo "Cleanup complete."

##@ Help

help: ## Show this help
	@echo ""
	@echo "Kaldi/Vosk training"
	@echo ""
	@echo "Usage: make <target> [VAR=value ...]"
	@awk 'BEGIN {FS = ":.*## "} \
		/^##@/ { printf "\n%s\n", substr($$0, 5) } \
		/^[a-zA-Z0-9_.-]+:.*## / { printf "  %-22s %s\n", $$1, $$2 }' $(firstword $(MAKEFILE_LIST))
	@echo ""
	@echo "Training runs inside the container from 'make run':"
	@echo "  cd /opt/kaldi/egs/cy && bash run.sh 10        (see howto.md)"
	@echo ""
	@echo "Config comes from .env: HF_TOKEN, HF_USERNAME, DEVICE (gpu id)."
	@echo ""
