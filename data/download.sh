#!/usr/bin/env bash
# Download request traces and the SplitwiseSim profiling table (not committed).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p traces profiles
MC=https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release
AZ=https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data
curl -fsSL -o traces/mooncake_arxiv.jsonl        "$MC/arxiv-trace/mooncake_trace.jsonl"
curl -fsSL -o traces/mooncake_conversation.jsonl "$MC/traces/conversation_trace.jsonl"
curl -fsSL -o traces/azure_code_2023.csv         "$AZ/AzureLLMInferenceTrace_code.csv"
curl -fsSL -o traces/azure_conv_2023.csv         "$AZ/AzureLLMInferenceTrace_conv.csv"
curl -fsSL -o profiles/splitwise_perf_model.csv  https://raw.githubusercontent.com/Mutinifni/splitwise-sim/main/data/perf_model.csv
echo "done"
