# Benchmarking on a real phone

The spec's performance target names a **2022 mid-range Android phone**, and no
amount of measuring on a laptop or a CI container settles whether it is met.
`scripts/bench/` is a page that runs the real engine with the real models on
whatever device opens it.

## Do you need GitHub Pages?

**No.** Serving the folder from your machine over the local network is enough,
and it is the fastest route.

The benchmark uses **no microphone** — it generates its audio in the page — and
that is what removes the hosting problem. Microphone capture needs
`getUserMedia` and `AudioWorklet`, both of which require a *secure context*:
HTTPS, or `localhost`. A phone reaching your laptop at
`http://192.168.1.50:4173` is neither. Verified in Chromium at exactly such an
origin:

```
origin: http://phone.lan:4173
isSecureContext:  false
AudioWorklet:     undefined      <- capture would be impossible here
WebAssembly:      available
```

Everything the benchmark actually needs — WebAssembly, a classic Worker, the
models over plain HTTP — works without a secure context, and the full run
completes. So:

| What you want to measure | Needs HTTPS? | How |
| --- | --- | --- |
| Engine cost per window | no | serve over the LAN, below |
| Anything using the microphone | **yes** | GitHub Pages, a tunnel, or a dev certificate |

If you later want to measure live capture on a phone, that is when you need
HTTPS — and then GitHub Pages, `cloudflared tunnel`, `ngrok` or a locally
trusted certificate all work. The built folder is fully relative
(`base: './'`), so it can be dropped into any of them unchanged.

## Running it over the local network

```bash
pnpm eval:fetch --models     # once: YAMNet task files + MediaPipe WASM assets
pnpm bench                   # builds, then serves on 0.0.0.0:4173
```

Vite prints a `Network:` address. Open that on the phone, on the same Wi-Fi, and
press **Run benchmark**. It takes about 40 s and downloads roughly 17 MB of
models first, so use Wi-Fi rather than mobile data.

If the phone cannot reach the address, it is almost always the laptop's firewall
rather than the page.

## What it measures

Three scenarios, each four passes over 30 s of audio, the first discarded as
warm-up, median of the rest — the same methodology as `docs/eval-results.md`:

| Scenario | What it tells you |
| --- | --- |
| Classifier + embedder | the headline number, against the 30 ms target |
| Classifier only | the cost without the embedder, which is over half the total |
| Guards on, silence | what `createEngine({ guards })` saves by skipping the embedder |

It also records the model load time and the device's user agent, core count,
reported memory and screen, so a result is attributable to hardware.

## Reporting a result

Press **Copy results** and paste the JSON into the issue or pull request. It is
self-describing: it carries the target, the framing constants, the methodology
and the device, so a number never arrives without the context needed to read it.

## Reading the result

The target is 30 ms per window. The engine produces one window every
`HOP_SECONDS` (0.4875 s), so the real-time budget is far larger than 30 ms — the
target exists so that a phone can keep up while doing other work, with the
screen on, without draining the battery. A device at 45 ms is still keeping up
with the stream; it is just spending more of the phone than intended.

If a device comes in over budget, the first thing to look at is the embedder: it
is over half the per-window cost, and `createEngine({ guards })` already avoids
it on windows that do not matter.
