"""Static publishing contracts; these tests never run the simulation or call the network."""
import json
import re
import shutil
import struct
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
ORIGIN = "https://fruitfly.crimconsortium.com"
UPDATED = "2026-09-12T12:35:00-04:00"


class Page(HTMLParser):
    def __init__(self, path):
        super().__init__()
        self.elements = []
        self.captures = []
        self.active = None
        self.feed(path.read_text())

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.elements.append((tag, attrs))
        if tag in ("title", "script", "h1"):
            self.active = [tag, attrs, ""]

    def handle_data(self, data):
        if self.active is not None:
            self.active[2] += data

    def handle_endtag(self, tag):
        if self.active is not None and tag == self.active[0]:
            self.captures.append(self.active)
            self.active = None

    def attrs(self, tag):
        return [attrs for element, attrs in self.elements if element == tag]

    def meta(self, key):
        values = [
            attrs["content"] for attrs in self.attrs("meta")
            if attrs.get("name", attrs.get("property")) == key
        ]
        assert len(values) == 1, f"Expected one {key} tag, found {len(values)}"
        return values[0]

    def captured(self, tag):
        return [(attrs, text) for element, attrs, text in self.captures if element == tag]


def workflow(filename):
    # BaseLoader preserves GitHub's "on" key instead of interpreting it as YAML 1.1 True.
    return yaml.load((ROOT / ".github/workflows" / filename).read_text(), Loader=yaml.BaseLoader)


def test_homepage_core_metadata():
    page = Page(SITE / "index.html")
    title = page.captured("title")[0][1]
    description = page.meta("description")
    assert 0 < len(title) < 60
    assert 140 <= len(description) <= 160
    assert page.meta("author") == "CrimConsortium"
    assert page.meta("robots") == "index, follow, max-image-preview:large"
    assert page.meta("theme-color") == "#000000"
    assert page.attrs("html")[0]["lang"] == "en"
    assert len(page.captured("h1")) == 1
    assert [a["href"] for a in page.attrs("link") if a.get("rel") == "canonical"] == [ORIGIN + "/"]
    assert page.meta("og:title") == page.meta("twitter:title") == title
    assert page.meta("og:description") == page.meta("twitter:description") == description
    assert page.meta("og:type") == "website"
    assert page.meta("og:url") == ORIGIN + "/"
    assert page.meta("og:site_name") == "fruitfly — CrimConsortium"
    assert page.meta("og:locale") == "en_US"
    assert page.meta("og:updated_time") == UPDATED


def test_social_media_targets_and_dimensions():
    page = Page(SITE / "index.html")
    assert page.meta("og:video") == ORIGIN + "/fly.mp4"
    assert page.meta("og:video:secure_url") == ORIGIN + "/fly.mp4"
    assert page.meta("og:video:type") == "video/mp4"
    assert page.meta("og:video:width") == "1280"
    assert page.meta("og:video:height") == "720"
    assert page.meta("og:image") == page.meta("og:image:secure_url") == ORIGIN + "/fly.gif"
    assert page.meta("og:image:type") == "image/gif"
    gif_header = (SITE / "fly.gif").read_bytes()[:10]
    width, height = struct.unpack("<HH", gif_header[6:10])
    assert (int(page.meta("og:image:width")), int(page.meta("og:image:height"))) == (width, height)
    assert page.meta("og:image:alt")
    assert page.meta("twitter:card") == "player"
    assert page.meta("twitter:player") == ORIGIN + "/player.html"
    assert page.meta("twitter:player:width") == "1280"
    assert page.meta("twitter:player:height") == "720"
    assert page.meta("twitter:image") == ORIGIN + "/fly.gif"
    assert page.meta("twitter:image:alt") == page.meta("og:image:alt")
    names = {attrs.get("name") for attrs in page.attrs("meta")}
    assert "twitter:site" not in names
    assert "twitter:creator" not in names


def test_structured_data_uses_only_known_brand_and_related_links():
    page = Page(SITE / "index.html")
    scripts = [text for attrs, text in page.captured("script") if attrs.get("type") == "application/ld+json"]
    assert len(scripts) == 1
    data = json.loads(scripts[0])
    assert data["@context"] == "https://schema.org"
    graph = {entry["@type"]: entry for entry in data["@graph"]}
    assert set(graph) == {"Organization", "WebSite", "WebPage"}
    organization = graph["Organization"]
    assert organization["name"] == "CrimConsortium"
    assert organization["url"] == "https://crimconsortium.com"
    assert "contactPoint" not in organization
    assert "sameAs" not in organization  # Related projects are not organization identities.
    assert graph["WebSite"]["publisher"]["@id"] == organization["@id"]
    assert "author" not in graph["WebPage"]  # Homepage attribution is the site's publisher.
    assert graph["WebPage"]["isPartOf"]["@id"] == graph["WebSite"]["@id"]
    assert graph["WebPage"]["description"] == graph["WebSite"]["description"] == page.meta("description")
    related = graph["WebPage"]["relatedLink"]
    assert set(related) == {
        "https://github.com/crimconsortium/fruitfly",
        "https://crimconsortium.com",
    }
    assert set(related).issubset({attrs.get("href") for attrs in page.attrs("a")})


def test_player_is_accessible_native_video_without_autoplay():
    page = Page(SITE / "player.html")
    assert page.attrs("html")[0]["lang"] == "en"
    assert len(page.captured("h1")) == 1
    assert 0 < len(page.captured("title")[0][1]) < 60
    assert 140 <= len(page.meta("description")) <= 160
    assert page.meta("author") == "CrimConsortium"
    assert page.meta("robots") == "noindex, follow"
    assert [a["href"] for a in page.attrs("link") if a.get("rel") == "canonical"] == [ORIGIN + "/player.html"]
    video, = page.attrs("video")
    assert "controls" in video and "playsinline" in video
    assert "autoplay" not in video
    assert video["preload"] == "metadata"
    assert (video["width"], video["height"]) == ("1280", "720")
    ids = {attrs.get("id") for _, attrs in page.elements}
    assert video["aria-labelledby"] in ids
    assert video["aria-describedby"] in ids
    assert page.attrs("source") == [{"src": "fly.mp4", "type": "video/mp4"}]
    assert not page.attrs("script")  # Playback uses native controls, including without JavaScript.
    assert any(attrs.get("href") == "./" for attrs in page.attrs("a"))
    assert any(attrs.get("href") == "fly.mp4" for attrs in page.attrs("a"))


def test_domain_indexing_and_site_assets():
    assert (SITE / "CNAME").read_text() == "fruitfly.crimconsortium.com\n"
    robots = (SITE / "robots.txt").read_text()
    assert "User-agent: *\nAllow: /" in robots
    assert f"Sitemap: {ORIGIN}/sitemap.xml" in robots
    sitemap = ElementTree.parse(SITE / "sitemap.xml")
    ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    # The embed is noindex, but must remain fetchable by social players.
    assert [el.text for el in sitemap.findall("s:url/s:loc", ns)] == [ORIGIN + "/"]
    assert [el.text for el in sitemap.findall("s:url/s:lastmod", ns)] == [UPDATED]
    assert ElementTree.parse(SITE / "favicon.svg").getroot().tag.endswith("svg")
    for filename in ("index.html", "player.html", "stats.json", "fly.mp4", "fly.gif",
                     "fly-loop.mp4", "fly-loop.jpg", "favicon.svg"):
        assert (SITE / filename).stat().st_size > 0


def test_pages_triggers_permissions_and_success_guard():
    pages = workflow("pages.yml")
    assert set(pages["on"]) == {"push", "workflow_dispatch", "workflow_run"}
    assert pages["on"]["push"] == {
        "branches": ["main"],
        "paths": ["site/**", ".github/workflows/pages.yml"],
    }
    assert pages["on"]["workflow_run"] == {"workflows": ["go"], "types": ["completed"]}
    assert pages["permissions"] == {"contents": "read", "pages": "write", "id-token": "write"}
    assert pages["concurrency"] == {"group": "pages", "cancel-in-progress": "false"}
    guard = " ".join(pages["jobs"]["deploy"]["if"].split())
    assert guard == (
        "(github.event_name != 'workflow_run' && github.ref == 'refs/heads/main') || "
        "(github.event_name == 'workflow_run' && "
        "github.event.workflow_run.conclusion == 'success' && "
        "github.event.workflow_run.head_branch == 'main' && "
        "github.event.workflow_run.head_repository.full_name == github.repository)"
    )


def test_pages_uses_latest_main_and_only_complete_site_files():
    job = workflow("pages.yml")["jobs"]["deploy"]
    steps = job["steps"]
    actions = {step["uses"]: step for step in steps if "uses" in step}
    checkout = actions["actions/checkout@v4"]["with"]
    assert checkout["ref"] == "main"  # Not go's pre-render head_sha.
    assert checkout["persist-credentials"] == "false"
    assert "actions/configure-pages@v5" in actions
    assert actions["actions/upload-pages-artifact@v3"]["with"]["path"] == "site/"
    assert actions["actions/deploy-pages@v4"]["id"] == "deployment"
    assert job["environment"] == {
        "name": "github-pages",
        "url": "${{ steps.deployment.outputs.page_url }}",
    }
    check = next(step for step in steps if step.get("name") == "Require the complete static site")
    for filename in ("index.html", "player.html", "stats.json", "fly.mp4", "fly.gif",
                     "fly-loop.mp4", "CNAME"):
        assert filename in check["run"]
    assert steps.index(check) < steps.index(actions["actions/upload-pages-artifact@v3"])
    assert not any("src/" in step.get("run", "") for step in steps)

    go = workflow("go.yml")
    assert go["name"] == "go"
    go_steps = go["jobs"]["everything"]["steps"]
    checkpoint = next(step for step in go_steps if step.get("name", "").startswith("Save the crawl checkpoint"))
    assert "git add data/citations/" in checkpoint["run"]
    assert "site/" not in checkpoint["run"]
    render = next(step for step in go_steps if step.get("name") == "Render the video")
    final = next(step for step in go_steps if step.get("name") == "Commit everything")
    assert go_steps.index(final) > go_steps.index(render)
    assert "git add data/ site/ RESULTS.md" in final["run"]


def test_committed_stats_contains_the_original_display_fields():
    stats = json.loads((SITE / "stats.json").read_text())
    for section, fields in {"result": ("reachable", "walls_hit", "n_steps"), "engine": ("neurons",)}.items():
        for field in fields:
            value = stats[section][field]
            assert type(value) is int and value >= 0


NODE_RUNNER = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const escape = value => String(value).replace(/[&<>"]/g, c => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'
}[c]));
function element() {
  return {
    html: '', text: '',
    set innerHTML(value) { this.html = value; this.text = ''; },
    set textContent(value) { this.text = value; this.html = ''; },
    append(...nodes) {
      this.html += nodes.map(n => n.tag ? `<${n.tag}>` : escape(n.text)).join('');
    }
  };
}
const elements = {stats: element(), 'control-numbers': element(), 'run-story': element()};
const calls = [], errors = [];
const context = {
  document: {
    getElementById: id => elements[id],
    createElement: tag => ({tag}),
    createTextNode: text => ({text})
  },
  console: {error: (...args) => errors.push(args.map(String).join(' '))},
  fetch: async (...args) => {
    calls.push(args);
    if (input.mode === 'network') throw new Error('Offline');
    return {
      ok: input.mode !== 'http',
      status: input.mode === 'http' ? 503 : 200,
      json: async () => {
        if (input.mode === 'invalid-json') throw new SyntaxError('Invalid JSON');
        return input.stats;
      }
    };
  }
};
(async () => {
  await vm.runInNewContext(input.script, context);
  process.stdout.write(JSON.stringify({elements, calls, errors}));
})().catch(error => { console.error(error); process.exitCode = 1; });
"""


def run_stats(stats, mode="success"):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is optional; needed only to execute the browser statistics script")
    page = Page(SITE / "index.html")
    script, = [text for attrs, text in page.captured("script") if attrs.get("id") == "run"]
    result = subprocess.run(
        [node, "-e", NODE_RUNNER],
        input=json.dumps({"script": script, "stats": stats, "mode": mode}),
        text=True, capture_output=True, check=True, timeout=10,
    )
    return json.loads(result.stdout)


def test_stats_requests_fresh_data_and_preserves_displayed_metrics():
    stats = json.loads((SITE / "stats.json").read_text())
    result = run_stats(stats)
    assert result["calls"] == [["stats.json", {"cache": "no-store"}]]
    assert not result["errors"]
    cards = re.findall(r"<b>([^<]*)</b><span>([^<]*)</span>", result["elements"]["stats"]["html"])
    # Every counter is thousands-separated, so 5,248 never appears next to 20,000 as "5248".
    assert cards == [
        (f'{stats["result"]["reachable"]:,}', "papers read"),
        (f'{stats["result"]["walls_hit"]:,}', "paywalls hit"),
        (f'{stats["engine"]["neurons"]:,}', "neurons simulated"),
    ]
    # "moves" duplicated "papers read", so it is deliberately absent.
    assert "moves" not in result["elements"]["stats"]["html"]
    story = result["elements"]["run-story"]["html"]
    assert stats["seed"]["title"] in story
    assert f'{stats["result"]["reachable"]:,} papers' in story
    assert f'{stats["result"]["walls_hit"]:,} paywalls' in story
    control = result["elements"]["control-numbers"]["html"]
    assert f'(mean of {stats["calibration"]["n_shuffle_replicates"]}, sd ' in control
    assert stats["calibration"]["verdict"] in control


@pytest.mark.parametrize("mode,stats", [
    ("network", {}), ("http", {}), ("invalid-json", {}), ("success", None), ("success", []),
])
def test_stats_failures_are_visible_and_never_fake_zero(mode, stats):
    result = run_stats(stats, mode)
    assert result["errors"]
    assert result["elements"]["stats"]["text"] == "Run statistics are unavailable. Please reload to try again."
    assert result["elements"]["stats"]["html"] == ""
    assert result["elements"]["control-numbers"]["text"] == "Calibration statistics are unavailable for this request."


def test_missing_counts_are_not_reported_as_zero_but_real_zero_is_preserved():
    result = run_stats({"result": {"reachable": 0, "walls_hit": None, "n_steps": 0}})
    html = result["elements"]["stats"]["html"]
    assert "<b>0</b><span>papers read</span>" in html
    assert "<b>n/a</b><span>paywalls hit</span>" in html
    assert "<b>n/a</b><span>neurons simulated</span>" in html
    assert result["elements"]["control-numbers"]["text"] == "No calibration was recorded for this run."


@pytest.mark.parametrize("extra", [{}, {"selectivity_shuffled_mean": None}])
def test_legacy_single_shuffle_calibration_remains_supported(extra):
    result = run_stats({"calibration": {
        "selectivity_real": 0.41, "selectivity_shuffled": 0.67, "chance_level": 0.125, **extra,
    }})
    html = result["elements"]["control-numbers"]["html"]
    assert "Real connectome: 41%" in html
    assert "Shuffled control: 67%" in html
    assert "Chance: 13%." in html
    assert "mean of" not in html


def test_the_control_reads_like_the_rest_of_the_page():
    body = (SITE / "index.html").read_text()
    assert "<h2>Did the brain matter?</h2>" in body
    assert 'class="null"' not in body and ".null {" not in body


def test_calibration_verdict_is_plain_text_not_html():
    result = run_stats({"calibration": {"selectivity_real": 0.41, "verdict": "<b>Keep this literal</b>"}})
    assert "&lt;b&gt;Keep this literal&lt;/b&gt;" in result["elements"]["control-numbers"]["html"]


def test_homepage_leads_with_the_illustration_then_the_measured_run():
    page = Page(SITE / "index.html")
    videos = [attrs for attrs in page.attrs("video")]
    assert [v["src"] for v in videos] == ["fly-loop.mp4", "fly.mp4"]
    loop, run = videos
    # The looping illustration is the hook; it must not be mistaken for the simulation.
    assert loop["poster"] == "fly-loop.jpg"
    assert "loop" in loop and "muted" in loop
    assert "autoplay" in loop and "autoplay" not in run
    for video in videos:
        assert (video["width"], video["height"]) == ("1280", "720")
    body = (SITE / "index.html").read_text()
    assert "not the simulation" in body
    # The run's numbers sit between the illustration and the measured render.
    assert body.index('id="stats"') > body.index('src="fly-loop.mp4"')
    assert body.index('id="stats"') < body.index('src="fly.mp4"')


def test_the_field_wide_numbers_on_the_page_match_the_measured_traces():
    import csv
    rows = list(csv.DictReader((ROOT / "data/traces/summary.csv").open()))
    body = (SITE / "index.html").read_text()
    blocked = sum(row["stuck_reason"] == "seed_paywalled" for row in rows)
    read = sum(int(row["reachable"]) for row in rows)
    walls = sum(int(row["walls_hit"]) for row in rows)
    truncated = sum(row["truncated"] == "True" for row in rows)
    assert f"{blocked} of the {len(rows)} randomly drawn papers were paywalled" in body
    assert f"{read:,} papers" in body
    assert f"{walls:,} paywalls" in body
    assert f"{truncated} of those {len(rows) - blocked} hit our 800-paper cap" in body


def test_background_credits_the_connectome_and_the_access_evidence():
    body = (SITE / "index.html").read_text()
    # Both empirical claims on the page carry a source a reader can open.
    assert "cell.com/cell/fulltext/S0092-8674(26)00942-6" in body  # MaleCNS in Cell
    assert "lesscrime.info/files/open_access_to_criminology_postprint.pdf" in body  # Ashby, JCJE
    assert "male-cns.janelia.org" in body and "openalex.org" in body


def test_page_claims_no_crimrxiv_affiliation():
    """No CrimRxiv branding or links; the feedback mailbox is the one allowed mention."""
    body = (SITE / "index.html").read_text()
    allowed = 'Feedback, comments, reviews or anything else are welcome at'
    assert allowed in body
    assert '<a href="mailto:consortium@crimrxiv.com">consortium@crimrxiv.com</a>' in body
    stripped = body.replace("mailto:consortium@crimrxiv.com", "").replace(
        "consortium@crimrxiv.com", "")
    assert "rimrxiv" not in stripped.lower()


def test_the_page_carries_the_family_theme_toggle():
    """The other CrimConsortium tools ship this exact control, so this one does too."""
    body = (SITE / "index.html").read_text()
    assert 'class="icon-btn" type="button" data-theme-toggle' in body
    assert 'aria-label="Toggle color scheme"' in body

    theme, = [text for attrs, text in Page(SITE / "index.html").captured("script") if attrs.get("id") == "theme"]
    # Follows the system scheme on load, flips on click, persists nothing.
    assert "(prefers-color-scheme: dark)" in theme
    assert "localStorage" not in theme and "cookie" not in theme
    assert "setAttribute('data-theme', t)" in theme

    for scheme in ("light", "dark"):
        assert f'[data-theme="{scheme}"]' in body
    # Both schemes define every colour the stylesheet consumes.
    used = set(re.findall(r"var\((--[a-z-]+)\)", body))
    for scheme in ('[data-theme="light"]', '[data-theme="dark"]'):
        block = body.split(scheme, 1)[1].split("}", 1)[0]
        assert used <= set(re.findall(r"(--[a-z-]+):", block)), scheme


def test_prose_carries_no_bold_and_the_notes_resolve():
    """Scott asked for no bold type in the copy; headings carry the hierarchy instead."""
    body = (SITE / "index.html").read_text()
    assert "<strong>" not in body and "<b " not in body
    script, = [t for a, t in Page(SITE / "index.html").captured("script") if a.get("id") == "run"]
    assert "<strong>" not in script

    # Every endnote marker points at a note that exists, and every note points back.
    markers = set(re.findall(r'<sup><a id="(r\d+)" href="#(n\d+)">', body))
    assert markers
    for ref, note in markers:
        assert f'<li id="{note}">' in body
        assert f'href="#{ref}"' in body


def test_the_wordmark_links_home():
    body = (SITE / "index.html").read_text()
    assert '<a class="brand" href="https://crimconsortium.com">CRIMCONSORTIUM</a>' in body
    assert "Created by" in body and "Perplexity" in body
