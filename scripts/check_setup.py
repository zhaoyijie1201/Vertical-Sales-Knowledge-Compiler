"""Verify the environment: settings, API key, and optionally one real model call.

    python scripts/check_setup.py              settings and key, no model call, no cost
    python scripts/check_setup.py --call       also one small call to the model under test
    python scripts/check_setup.py --models anthropic   list model slugs containing a word

The key is never printed.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request

from vskc.cli import fail, setup_console
from vskc.config import api_key, key_file, load_settings


def get_json(url: str, key: str = None):
    req = urllib.request.Request(url)
    if key:
        req.add_header("Authorization", "Bearer " + key)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main(argv=None) -> int:
    setup_console()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--call", action="store_true", help="make one real call to the model under test")
    p.add_argument("--models", metavar="WORD", help="list model slugs that contain WORD")
    args = p.parse_args(argv)

    st = load_settings()
    print("base url            %s" % st.base_url)
    print("model under test    %s" % (st.model_under_test or "(not set)"))
    print("generation, dev     %s" % (st.model_gen_dev or "(not set)"))
    print("generation, heldout %s" % (st.model_gen_heldout or "(not set)"))
    print("response format     %s" % st.response_format)
    print("top_k               %d" % st.top_k)
    print("retrieval           %s" % st.retrieval)
    print("prices per Mtok     in %s, out %s, checked on %s"
          % (st.price_in_per_mtok, st.price_out_per_mtok, st.price_checked_on or "n/a"))

    try:
        key = api_key()
    except RuntimeError as e:
        return fail(str(e))
    print("api key             found (%d characters), not shown" % len(key))
    print("key file            %s (%s)" % (key_file(), "exists" if key_file().is_file() else "absent"))

    try:
        info = get_json(st.base_url.rstrip("/") + "/key", key).get("data", {})
        print("key check           accepted by the server")
        print("  usage so far      %s" % info.get("usage"))
        print("  limit             %s" % info.get("limit"))
        print("  limit remaining   %s" % info.get("limit_remaining"))
    except urllib.error.HTTPError as e:
        return fail("the server rejected the key: HTTP %d" % e.code)
    except Exception as e:
        return fail("could not reach the server: %s" % type(e).__name__)

    if args.models:
        data = get_json(st.base_url.rstrip("/") + "/models").get("data", [])
        word = args.models.lower()
        hits = [m for m in data if word in m.get("id", "").lower()]
        print("\n%d models contain %r" % (len(hits), args.models))
        for m in sorted(hits, key=lambda m: m["id"]):
            pr = m.get("pricing", {})
            params = m.get("supported_parameters", [])
            try:
                pin = "%.2f" % (float(pr.get("prompt", 0)) * 1e6)
                pout = "%.2f" % (float(pr.get("completion", 0)) * 1e6)
            except (TypeError, ValueError):
                pin = pout = "?"
            print("  %-48s in %7s  out %7s per Mtok  structured_outputs=%s"
                  % (m["id"], pin, pout, "structured_outputs" in params))

    if args.call:
        if not st.model_under_test:
            return fail("VSKC_MODEL_UNDER_TEST is not set in .env")
        from openai import OpenAI

        client = OpenAI(base_url=st.base_url, api_key=key)
        resp = client.chat.completions.create(
            model=st.model_under_test, max_tokens=200,
            messages=[{"role": "user", "content": "Reply with the single word: ready"}],
        )
        print("\nmodel call          ok")
        print("  reply             %r" % (resp.choices[0].message.content or "").strip()[:80])
        if resp.usage:
            print("  tokens            in %d, out %d" % (resp.usage.prompt_tokens, resp.usage.completion_tokens))
    return 0


if __name__ == "__main__":
    sys.exit(main())
