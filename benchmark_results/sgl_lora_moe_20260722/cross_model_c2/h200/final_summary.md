# C2 cross-model guardrail summary: h200

Passed 276/276 isolated schedule configurations; 0 failed. Pair timing includes the down-rank zero plus one consumer launch; aligned timing includes the zero, masked base-only activation fill, and aligned consumer under CUDA graph replay.

| Model | T | R | Selected | BN | p50 us | aligned/pair |
|---|---:|---:|---|---:|---:|---:|
| glm-5.2 | 1 | 16 | aligned | 16 | 11.600 | 0.433 |
| glm-5.2 | 1 | 32 | aligned | 64 | 11.360 | 0.383 |
| glm-5.2 | 1 | 64 | aligned | 32 | 11.472 | 0.290 |
| glm-5.2 | 32 | 16 | pair | 64 | 35.504 | 1.711 |
| glm-5.2 | 32 | 32 | aligned | 64 | 39.744 | 0.861 |
| glm-5.2 | 32 | 64 | pair | 64 | 52.192 | 1.102 |
| glm-5.2 | 256 | 16 | pair | 64 | 108.720 | 1.651 |
| glm-5.2 | 256 | 32 | aligned | 64 | 120.000 | 0.770 |
| glm-5.2 | 256 | 64 | aligned | 64 | 167.040 | 0.661 |
| kimi-k2.5 | 1 | 16 | aligned | 16 | 11.520 | 0.433 |
| kimi-k2.5 | 1 | 32 | aligned | 32 | 11.296 | 0.380 |
| kimi-k2.5 | 1 | 64 | aligned | 64 | 11.504 | 0.291 |
| kimi-k2.5 | 32 | 16 | pair | 64 | 36.688 | 1.650 |
| kimi-k2.5 | 32 | 32 | aligned | 64 | 39.824 | 0.891 |
| kimi-k2.5 | 32 | 64 | pair | 64 | 49.696 | 1.155 |
| kimi-k2.5 | 256 | 16 | pair | 64 | 109.664 | 2.137 |
| kimi-k2.5 | 256 | 32 | aligned | 64 | 147.712 | 0.932 |
| kimi-k2.5 | 256 | 64 | aligned | 64 | 215.536 | 0.828 |
| nemotron-3-nano | 1 | 16 | aligned | 16 | 10.544 | 0.477 |
| nemotron-3-nano | 1 | 32 | aligned | 64 | 10.576 | 0.397 |
| nemotron-3-nano | 1 | 64 | aligned | 32 | 10.848 | 0.339 |
| nemotron-3-nano | 32 | 16 | aligned | 64 | 24.128 | 0.834 |
| nemotron-3-nano | 32 | 32 | aligned | 64 | 21.264 | 0.592 |
| nemotron-3-nano | 32 | 64 | aligned | 64 | 28.704 | 0.540 |
| nemotron-3-nano | 256 | 16 | aligned | 64 | 64.800 | 0.833 |
| nemotron-3-nano | 256 | 32 | aligned | 64 | 58.688 | 0.634 |
| nemotron-3-nano | 256 | 64 | aligned | 64 | 72.800 | 0.586 |
| nemotron-3-super | 1 | 16 | aligned | 64 | 12.400 | 0.228 |
| nemotron-3-super | 1 | 32 | aligned | 64 | 11.792 | 0.203 |
| nemotron-3-super | 1 | 64 | aligned | 64 | 13.536 | 0.221 |
| nemotron-3-super | 32 | 16 | pair | 64 | 54.224 | 2.068 |
| nemotron-3-super | 32 | 32 | pair | 64 | 64.384 | 1.385 |
| nemotron-3-super | 32 | 64 | pair | 64 | 103.584 | 1.220 |
| nemotron-3-super | 256 | 16 | aligned | 64 | 325.088 | 0.879 |
| nemotron-3-super | 256 | 32 | aligned | 64 | 279.280 | 0.626 |
| nemotron-3-super | 256 | 64 | aligned | 64 | 359.088 | 0.588 |
| odd-provider-padding | 32 | 32 | aligned | 64 | 14.656 | 0.377 |
| qwen3.5-397b-a17b | 1 | 16 | aligned | 16 | 11.408 | 0.764 |
| qwen3.5-397b-a17b | 1 | 32 | aligned | 64 | 11.392 | 0.709 |
| qwen3.5-397b-a17b | 1 | 64 | aligned | 32 | 11.408 | 0.603 |
| qwen3.5-397b-a17b | 32 | 16 | pair | 64 | 17.920 | 2.216 |
| qwen3.5-397b-a17b | 32 | 32 | pair | 64 | 21.568 | 1.264 |
| qwen3.5-397b-a17b | 32 | 64 | pair | 64 | 33.792 | 1.201 |
| qwen3.5-397b-a17b | 256 | 16 | pair | 64 | 72.192 | 2.248 |
| qwen3.5-397b-a17b | 256 | 32 | aligned | 64 | 102.096 | 0.984 |
| qwen3.5-397b-a17b | 256 | 64 | aligned | 64 | 151.472 | 0.849 |
