# Model catalog

The catalog lists the models the router may pick, with prices, cache multipliers, efforts and flags. It is a TOML file loaded into `ModelProfile` records (`docs/SCHEMA.md`). The default catalog ships inside the package at `src/steproute/catalogs/default.toml` (package data, so an installed wheel has it).

```python
from steproute.core.catalog import load_catalog, load_default_catalog

catalog = load_default_catalog()  # or load_catalog("my.toml")
for profile in catalog:  # tier order, weakest first
    ...
catalog.get("claude-haiku-5-5")
catalog.pool()  # non-holdout models
catalog.local_models()
```

Sources for every price, id and limit in the default catalog are in `docs/INTEGRATION-NOTES.md`, section 7 (checked 2026-11-13). They are not repeated here.

## File format

A catalog is a list of `[[model]]` tables. Nothing else is allowed at the top level.

```toml
[[model]]
id = "claude-haiku-5-5"
provider = "anthropic"
price_in = 0.10          # USD per million input tokens
price_out = 0.50
context_window = 1000000
tier = 4
cache_write_mult = 1.25
cache_read_mult = 0.1
efforts = ["low", "medium", "high", "xhigh", "max"]
effort_output_mult = { low = 1.0, medium = 1.0, high = 1.0, xhigh = 1.0, max = 1.0 }
effort_map = { low = "low", medium = "medium" }
verified = true

[[model.price_tiers]]
above_tokens = 100000
price_in = 0.50
price_out = 2.50

[model.meta]
default_effort = "medium"
```

Fields are those of `ModelProfile` (see `docs/SCHEMA.md` for types and defaults). A catalog is hand-written config, so loading is strict, unlike trace records: any unknown key fails. Validation raises `CatalogError`, whose message and `.model` / `.field` attributes name the offending model and field. It rejects:

- unknown keys (at the top level, in a model, or in a price tier) and duplicate ids;
- efforts outside `EFFORTS` (`none, low, medium, high, xhigh, max`), an empty or duplicated `efforts` list;
- `effort_output_mult` or `effort_map` keys that are not in the model's `efforts`;
- a partly filled `effort_output_mult`: give every listed effort a multiplier, or leave the table empty (which means 1.0 for all, unmeasured);
- negative prices, multipliers or `effort_output_mult` values, a non-positive `context_window`, a negative `tier`;
- price tiers whose `above_tokens` are not positive and strictly increasing.

"Round-trips" means `Catalog.from_dict(catalog.to_dict()) == catalog`, and the same for a catalog freshly parsed from TOML. There is no TOML writer.

## Field notes

- **Prices and multipliers.** `cache_write_mult` and `cache_read_mult` are multiples of `price_in`, per model. The Anthropic entries use the 5-minute write (1.25); the 1-hour write is in `meta.cache_write_1h_mult`. DeepSeek has no write premium (1.0); its read multiplier is hit price over miss price. Ollama is free (0 prices, 1.0 multipliers).
- **DeepSeek uses peak prices** in `price_in` / `price_out` (conservative for spend caps). Off-peak prices are in `meta.price_in_offpeak`, `meta.price_out_offpeak`, `meta.cache_hit_price_peak` and `meta.cache_hit_price_offpeak`. When the peak windows are applied is data too: `meta.peak_hours` (next section).
- **`price_tiers`.** Base prices apply up to and including the first `above_tokens`. A request whose prompt is longer pays that tier's rate on the whole request. The prompt length counts all input tokens including cache reads and writes. Exactly 100,000 tokens is still the lower tier. Use `ModelProfile.prices_for(prompt_tokens)`.
- **`tier`.** Cascade strength order across the whole pool: higher is stronger, ties allowed. Issue 5 uses it to decide whether an escalation is strictly stronger. **The default values are an editable assumption, not a measurement**: `qwen3.5:9b` 1; `gpt-oss:20b` 2; `qwen3-coder:30b` and `devstral:24b` 3; `claude-haiku-5-5` and `deepseek-flash` 4; `claude-sonnet-5-5` and `deepseek-v4-pro` 5; `claude-opus-5-5` 6; `claude-fable-5-2` 7.
- **`efforts` and `effort_map`.** `efforts` are the shared levels the model supports. `effort_map` gives the value sent as `reasoning_effort` for each, in the form the endpoint the proxy calls accepts (for example `"max"` for DeepSeek). For Ollama that endpoint is the OpenAI-compatible `/v1`, not the native API, so the values are `reasoning_effort` names, not native `think` booleans. A supported effort missing from the map means "send no reasoning parameter". Anthropic omits `none` because thinking cannot be switched off uniformly across its models. DeepSeek omits `medium` and `xhigh` because it aliases them to `high`; listing them would duplicate a configuration. `qwen3.5:9b` can only turn thinking on or off: `none` sends `"none"` (Ollama sets `think = false`) and `medium` sends `"medium"` (Ollama turns thinking on). This is verified from Ollama's source and docs at v0.40.16, not by a run; `meta.effort_map_evidence` lists the files. The native `"true"`/`"false"` are not valid on `/v1`: a model with thinking metadata resolves an unknown name to its default (thinking on), and one without rejects it; `gpt-oss:20b` has `low, medium, high`; `qwen3-coder:30b` and `devstral:24b` have no reasoning control (`medium` only, empty map).
- **`effort_change_keeps_cache`.** Whether changing effort between two requests keeps the provider's prompt cache. The cost model treats the cache owner as the pair (model, effort) unless this is true. It is `false` everywhere (conservative): Anthropic only keeps the cache with the per-message-effort beta, and a gpt-oss level change rewrites the system header.
- **`cache_min_tokens`.** The shortest prompt the provider caches. Below it nothing is read or written and no write premium is charged. 512 for the four Anthropic models (notes 7a). DeepSeek documents no minimum and Ollama has none, so both keep the default 0.
- **Latency (`latency_base_ms`, `latency_ms_per_input_token`, `latency_ms_per_output_token`, `latency_switch_ms`).** Optional estimates used by the cost model for expected latency (`docs/COST-MODEL.md`). The first three are set together or not at all; a partial set is a `CatalogError`. `latency_switch_ms` is added when the step's model is not the session's current model, which is how a local model reload is expressed. **No default-catalog model sets them: nothing has been measured**, so the cost model reports no latency for the default catalog.
- **Local compute price.** Ollama models have `price_in = price_out = 0`. To charge for local compute, set non-zero prices on the local profile; nothing else changes. Also set `cache_read_mult` near 0 then, since a KV-cache hit skips prefill.
- **`effort_output_mult`.** **Unmeasured placeholders: 1.0 for every listed effort.** Nobody has measured these; they are to be replaced from the pilot replay (issue 18). `verified` does not cover them.
- **`tools`.** `false` excludes the model from agent tasks. Set it after `steproute catalog check` fails for the model.
- **`holdout`.** Reserved for the cold-start experiment (`devstral:24b`). `Catalog.pool()` leaves holdout models out; they must never be used for training or normal routing.
- **`context_window`.** For Ollama this is the operational cap of 32768 (decision D2), not the native limit, which is in `meta.native_context`.

### `meta` keys used by the default catalog

| Key | Meaning |
|---|---|
| `cache_write_1h_mult` | Anthropic 1-hour cache write as a multiple of `price_in`. |
| `default_effort` | The provider's default effort, where known. |
| `native_context` | The model's own context limit in tokens. |
| `size_gb` | Download size on disk (Ollama). |
| `price_in_offpeak`, `price_out_offpeak`, `cache_hit_price_peak`, `cache_hit_price_offpeak`, `offpeak_note` | DeepSeek time-of-day pricing (USD per million tokens). |
| `peak_hours` | The hours when `price_in` / `price_out` apply; off-peak prices apply at every other hour. A table; see "Peak hours" below. |
| `open_cache_questions` | Cache facts the provider docs leave undetermined. |
| `reasoning_control` | How the model's reasoning is controlled (Ollama). |
| `effort_map_evidence` | Where the `effort_map` values were checked (source read, not run). |
| `source`, `checked` | Where the numbers came from and the date checked. |

### Peak hours

A model with a `[model.meta.peak_hours]` table is billed at its catalog prices inside the windows and at `meta.price_in_offpeak` / `meta.price_out_offpeak` at all other hours (the cache-hit ratio off-peak is `cache_hit_price_offpeak / price_in_offpeak`). A model without the table has one price at every hour (`flat`). `src/steproute/core/offpeak.py` reads the table and answers: is this instant peak or off-peak, when does the current off-peak stretch end, and when does the next one start. The catalog loader rejects a malformed table or missing off-peak prices.

```toml
[model.meta.peak_hours]
timezone = "UTC"                       # the only value accepted
windows = [
  { days = ["mon", "tue", "wed", "thu", "fri"], start = "01:00", end = "04:00" },
  { days = ["mon", "tue", "wed", "thu", "fri"], start = "06:00", end = "10:00" },
]
source = "https://api-docs.deepseek.com/quick_start/pricing"   # required
checked = "2026-11-13"                                         # required
note = "..."                                                   # optional
```

`start` is included and `end` is not; a window cannot cross midnight (write two). The default catalog gives both DeepSeek models these hours (checked 13 Nov 2026):

| Days (UTC) | Peak | Off-peak |
|---|---|---|
| Monday to Friday | 01:00-04:00 and 06:00-10:00 | 00:00-01:00, 04:00-06:00, 10:00-24:00 |
| Saturday, Sunday | none | all day |

Off-peak is half the peak price for cache misses, cache hits and output alike. What is not covered:

- **Chinese public holidays** are off-peak all day at DeepSeek. We do not keep that calendar and cannot verify it, so they count as ordinary weekdays: no discount is ever assumed on such a day.
- **A call that crosses a boundary.** DeepSeek's page does not say how it is billed. steproute prices a call off-peak only if both its start and its end are off-peak; anything else is priced at peak.
- None of this has been checked against a real DeepSeek bill.

## `verified`

`verified = true` means the price, cache multipliers, context window and effort levels were checked against provider docs. It does **not** cover `tier` (assumed), `effort_output_mult` (placeholders), latency estimates, `effort_change_keeps_cache`, or cache behaviour the provider does not document. For DeepSeek, the ids and prices are checked, but cache scope per model and the effect of toggling thinking are undetermined (`meta.open_cache_questions`).

Loading a catalog in which any model has `verified = false` emits exactly one `UnverifiedCatalogWarning` (a `UserWarning`) listing those ids. A fully verified catalog emits none. The default catalog is fully verified, so it does not warn.

## Commands

```
steproute catalog show [--catalog PATH]
```

Lists id, provider, tier, efforts and flags (`local`, `holdout`, `no-tools`, `verified` / `unverified`), in tier order.

```
steproute catalog check [--catalog PATH] [--base-url URL] [--model ID ...] [--timeout SECONDS]
```

The tool-call smoke test. For each model it sends one chat-completions request with one simple tool to an OpenAI-compatible endpoint and passes only if `choices[0].message.tool_calls` holds a call with a function name and JSON-parsable arguments. A tool call written as plain text in `content` fails. It prints one line per model; for a failure it says to set `tools = false` for that model. Exit code 0 if all pass, 1 if any fail, 3 if the server is unreachable (with a distinct message), 2 for usage or catalog errors.

The default target is local Ollama at `http://localhost:11434/v1` and the default selection is the catalog's local models. Cloud models are refused by this command; they are checked through the proxy (issue 12).

### Running it once Ollama is installed

Ollama is not installed on every machine, and the default context is unreliable (see the notes, section 7c). Set it explicitly and pull the tags first:

```
OLLAMA_CONTEXT_LENGTH=32768 ollama serve
ollama pull qwen3.5:9b gpt-oss:20b qwen3-coder:30b devstral:24b
steproute catalog check
```

Load one model at a time (`OLLAMA_MAX_LOADED_MODELS=1`) if memory is tight. The test suite has one real-Ollama test, skipped unless `STEPROUTE_TEST_OLLAMA=1` is set (`STEPROUTE_TEST_OLLAMA_MODEL` picks the model); it never runs in normal CI.
