# Transcritor API

API de transcrição offline, sem token e sem autenticação: recebe um áudio ou vídeo e
devolve o texto com o instante de cada palavra, em JSON, TXT ou legendas SRT/VTT.

O repositório é organizado por **modelo** e, dentro dele, por **onde roda**. Cada pasta
é um projeto completo e independente — código, dependências e documentação próprios;
nenhum importa nada do outro.

```
whisper/
  docker/
    cpu/        WhisperX na CPU, com separação por falante
    gpu-cuda/   WhisperX na placa NVIDIA, com separação por falante
qwen/
  windows/      Qwen3 na placa de vídeo, instalação nativa no Windows
  docker/       Qwen3 na placa de vídeo, em container (Vulkan ou CUDA)
```

| | [whisper/docker/cpu](whisper/docker/cpu/README.md) | [whisper/docker/gpu-cuda](whisper/docker/gpu-cuda/README.md) | [qwen/windows](qwen/windows/README.md) | [qwen/docker](qwen/docker/README.md) |
|---|---|---|---|---|
| Modelo | WhisperX | WhisperX | Qwen3-ASR + ForcedAligner | Qwen3-ASR + ForcedAligner |
| Roda em | container, CPU | container, GPU CUDA | Windows, GPU Vulkan | container, GPU Vulkan ou CUDA |
| Placas | — | NVIDIA | AMD, NVIDIA, Intel | AMD e Intel no Linux; NVIDIA |
| Separação por falante | sim | sim | não | não |
| Tradução para inglês | sim | sim | não | não |
| Erro mediano no início da palavra | ~65 ms | ~65 ms | ~22 ms | ~22 ms |
| Reunião de 88 min | ~36 min | não medido | ~7 min | ~7 min na GPU |
| Instalação | `docker run` | `docker run --gpus all` | `instalar.bat` | `docker run --gpus` / `--device` |

## Qual usar

- **Precisa saber quem falou?** WhisperX — o único que separa falantes: `whisper/docker/gpu-cuda`
  com placa NVIDIA, `whisper/docker/cpu` sem ela.
- **Legendas no Windows**, com qualquer placa: `qwen/windows`.
- **Legendas num servidor** Linux com AMD ou Intel, ou em qualquer máquina com NVIDIA:
  `qwen/docker`. No Windows com placa AMD ou Intel ele funciona, mas só na CPU — o
  Docker Desktop não repassa essas placas —, então aí prefira `qwen/windows`.

As quatro APIs têm as mesmas rotas (`/transcribe`, `/jobs`, `/jobs/{id}/download`,
`/health`) e o mesmo formato de resultado: um cliente troca de uma para outra mudando só
o endereço, desde que não peça `diarization=true` nem `task=translate`, que as versões
Qwen recusam com HTTP 400. Nas versões Qwen o `/health` traz `"engine": "qwen"` (a do
Windows, `"audiocpp"`); a do WhisperX não traz o campo — é assim que um cliente sabe se
pode pedir separação por falante.

## Licença

**GNU General Public License v3.0 ou posterior** (veja [LICENSE](LICENSE)). Uso
interno — rodar e modificar dentro da sua organização — não dispara nenhuma obrigação da
GPL; ela só se aplica ao **distribuir** o software ou um produto que o embuta, caso em
que o código-fonte deve ser fornecido sob os mesmos termos. As licenças das
dependências de cada projeto estão no README dele.
