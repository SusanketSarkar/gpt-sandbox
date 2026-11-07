# plugins/

Every `*.py` file here is imported automatically (files starting with `_` are helpers, importable but not auto-run).
Register components with `@register(kind, name)` and refer to them by name from configs.

    gpt new attention my_attn      # scaffolds plugins/my_attn.py as a working copy of `mha`
    gpt new insert my_gate         # hook into the residual stream, no model edits
    gpt list                       # everything registered

When you train, the plugin files are snapshotted into `runs/<name>/plugins/`, so `eval`/`query` on that
run always use the exact code it was trained with, even if you edit the file later.
