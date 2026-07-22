# C2 cross-model guardrail summary: gb300

Passed 276/276 isolated schedule configurations; 0 failed. Pair timing includes the down-rank zero plus one consumer launch; aligned timing includes the zero, masked base-only activation fill, and aligned consumer under CUDA graph replay.

| Model | T | R | Selected | BN | p50 us | aligned/pair |
|---|---:|---:|---|---:|---:|---:|
| glm-5.2 | 1 | 16 | aligned | 16 | 10.400 | 0.364 |
| glm-5.2 | 1 | 32 | aligned | 32 | 9.840 | 0.340 |
| glm-5.2 | 1 | 64 | aligned | 32 | 10.432 | 0.268 |
| glm-5.2 | 32 | 16 | pair | 64 | 35.280 | 1.274 |
| glm-5.2 | 32 | 32 | aligned | 64 | 30.656 | 0.750 |
| glm-5.2 | 32 | 64 | pair | 64 | 40.864 | 1.152 |
| glm-5.2 | 256 | 16 | pair | 64 | 92.096 | 1.625 |
| glm-5.2 | 256 | 32 | aligned | 64 | 100.224 | 0.765 |
| glm-5.2 | 256 | 64 | aligned | 64 | 126.880 | 0.602 |
| kimi-k2.5 | 1 | 16 | aligned | 16 | 9.968 | 0.348 |
| kimi-k2.5 | 1 | 32 | aligned | 32 | 9.584 | 0.335 |
| kimi-k2.5 | 1 | 64 | aligned | 64 | 10.816 | 0.278 |
| kimi-k2.5 | 32 | 16 | pair | 64 | 34.784 | 1.293 |
| kimi-k2.5 | 32 | 32 | aligned | 64 | 30.688 | 0.750 |
| kimi-k2.5 | 32 | 64 | pair | 64 | 40.896 | 1.149 |
| kimi-k2.5 | 256 | 16 | pair | 64 | 94.128 | 2.066 |
| kimi-k2.5 | 256 | 32 | aligned | 64 | 122.784 | 0.923 |
| kimi-k2.5 | 256 | 64 | aligned | 64 | 159.680 | 0.743 |
| nemotron-3-nano | 1 | 16 | aligned | 16 | 11.472 | 0.469 |
| nemotron-3-nano | 1 | 32 | aligned | 64 | 10.784 | 0.406 |
| nemotron-3-nano | 1 | 64 | aligned | 64 | 10.672 | 0.326 |
| nemotron-3-nano | 32 | 16 | aligned | 64 | 22.416 | 0.609 |
| nemotron-3-nano | 32 | 32 | aligned | 64 | 18.672 | 0.507 |
| nemotron-3-nano | 32 | 64 | aligned | 64 | 22.464 | 0.499 |
| nemotron-3-nano | 256 | 16 | aligned | 64 | 53.200 | 0.788 |
| nemotron-3-nano | 256 | 32 | aligned | 64 | 47.040 | 0.589 |
| nemotron-3-nano | 256 | 64 | aligned | 64 | 59.344 | 0.569 |
| nemotron-3-super | 1 | 16 | aligned | 64 | 11.776 | 0.222 |
| nemotron-3-super | 1 | 32 | aligned | 64 | 11.296 | 0.205 |
| nemotron-3-super | 1 | 64 | aligned | 64 | 12.752 | 0.215 |
| nemotron-3-super | 32 | 16 | pair | 64 | 48.576 | 2.065 |
| nemotron-3-super | 32 | 32 | pair | 64 | 55.200 | 1.334 |
| nemotron-3-super | 32 | 64 | pair | 64 | 77.760 | 1.290 |
| nemotron-3-super | 256 | 16 | aligned | 64 | 292.800 | 0.941 |
| nemotron-3-super | 256 | 32 | aligned | 64 | 227.232 | 0.606 |
| nemotron-3-super | 256 | 64 | aligned | 64 | 278.448 | 0.555 |
| odd-provider-padding | 32 | 32 | aligned | 64 | 14.048 | 0.404 |
| qwen3.5-397b-a17b | 1 | 16 | aligned | 32 | 9.968 | 0.578 |
| qwen3.5-397b-a17b | 1 | 32 | aligned | 64 | 9.344 | 0.510 |
| qwen3.5-397b-a17b | 1 | 64 | aligned | 64 | 10.192 | 0.500 |
| qwen3.5-397b-a17b | 32 | 16 | pair | 64 | 20.448 | 1.598 |
| qwen3.5-397b-a17b | 32 | 32 | aligned | 64 | 22.400 | 0.996 |
| qwen3.5-397b-a17b | 32 | 64 | pair | 64 | 26.592 | 1.150 |
| qwen3.5-397b-a17b | 256 | 16 | pair | 64 | 61.344 | 2.236 |
| qwen3.5-397b-a17b | 256 | 32 | aligned | 64 | 85.552 | 0.972 |
| qwen3.5-397b-a17b | 256 | 64 | aligned | 64 | 112.512 | 0.801 |
