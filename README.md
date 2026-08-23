# deeptunnel

**DeepSeek → Anthropic API proxy.**  
Run DeepSeek behind the Anthropic SDK, Claude Code, or any tool that speaks the Anthropic Messages API.

[![PyPI](https://img.shields.io/pypi/v/deeptunnel)](https://pypi.org/project/deeptunnel/)
[![Python](https://img.shields.io/pypi/pyversions/deeptunnel)](https://pypi.org/project/deeptunnel/)

---

## Install

```bash
pip install deeptunnel
```

The WASM proof-of-work solver (`sha3_wasm_bg.wasm`, ~26 KB) is downloaded
automatically from GitHub on the first run and cached in `~/.cache/deeptunnel/`.
You can also place the file next to the script manually to skip the download.

---

## Quick start

```bash
# 1. Set your DeepSeek bearer token
#    Get it: browser DevTools → Network → any DeepSeek request → Authorization header
export DEEPSEEK_TOKEN="your-token-here"

# 2. Start the proxy
deeptunnel

# 3. In another shell, point the Anthropic SDK at it
export ANTHROPIC_BASE_URL="http://localhost:8765"
export ANTHROPIC_API_KEY="local-proxy-key"
claude          # or any other Anthropic SDK client
```

---

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model fast` | ✓ | Use the fast DeepSeek model (`model_type="default"`) |
| `--model expert` | | Use the expert model (`model_type=null`) |
| `--search` | ✓ | Enable web search |
| `--no-search` | | Disable web search |
| `--think` | | Enable thinking mode |
| `--port PORT` | `8765` | Port to listen on |

```bash
deeptunnel --model expert --no-search --think --port 9000
```

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEEPSEEK_TOKEN` | *required* | Bearer token(s), comma-separated for rotation |
| `DEEPSEEK_COOKIES` | | Optional cookie string |
| `PROXY_PORT` | `8765` | Port override (CLI `--port` takes precedence) |
| `PROXY_MAX_SESSIONS` | `64` | Max cached DeepSeek sessions |
| `PROXY_MAX_HISTORY_MESSAGES` | `40` | Max history messages per prompt |

### Multiple tokens (auto-rotation)

```bash
export DEEPSEEK_TOKEN="token1,token2,token3"
deeptunnel
```

Tokens rotate automatically when DeepSeek returns a "server busy" /
`generation_timeout` response.

---

## How it works

1. Receives Anthropic-format `/v1/messages` requests.
2. Translates Anthropic tool definitions → XML schema in the prompt.
3. Solves DeepSeek's proof-of-work challenge (via WASM).
4. Forwards the request to `chat.deepseek.com`.
5. Parses DeepSeek's plain-text response for `<tool_use>` blocks.
6. Re-emits proper Anthropic `tool_use` / `text` content blocks as SSE.

---

## Manual WASM placement

If the automatic download fails (firewall, air-gapped environment), download
the file manually and place it in `~/.cache/deeptunnel/`:

```bash
mkdir -p ~/.cache/deeptunnel
curl -L 'https://github.com/IMApurbo/deepseek_scrapper/blob/main/deepseek-cli/sha3_wasm_bg.wasm' \
     -o ~/.cache/deeptunnel/sha3_wasm_bg.wasm
```

---

## License

MIT © [IMApurbo](https://github.com/IMApurbo)
