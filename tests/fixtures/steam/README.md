# Steam test fixtures

## `appdetails_portal_2_english.json`

* **Sanitized test fixture** — it is *not* a verbatim dump of the live
  response.
* **Source URL**:
  `https://store.steampowered.com/api/appdetails?appids=620&l=english`
* **App ID**: `620` (Portal 2)
* **Locale**: `english` (`l=english`)
* **Retrieved**: 2026-11-02, with exactly one read-only HTTP GET
  request. No retry, no redirect, no second App ID.
* **HTTP result**: `200`, `application/json`, 28,475 bytes of body.

### Fields retained

Only the fields the description pipeline needs:

| Field | Why it is kept |
| --- | --- |
| `type` | Payload classification. |
| `name` | Product name shown to the translator. |
| `detailed_description` | Primary description source — **HTML kept intact**. |
| `about_the_game` | Supplementary/fallback source — **HTML kept intact**. |
| `short_description` | Supporting context — **HTML entities kept intact**. |
| `release_date` | Deterministic release date, parsed from Steam JSON. |
| `platforms` | Deterministic Windows/macOS/Linux availability flags. |
| `pc_requirements` | Deterministic Windows requirements — **HTML kept intact**. |

### Fields removed

`price_overview`, `package_groups`, `recommendations`, `screenshots`,
`movies`, `metacritic`, `achievements`, `content_descriptors`,
`developers`, `publishers`, `categories`, `genres`, `supported_languages`,
`website`, `legal_notice`, `demos`, `controller_support`, and every other
unrelated metadata field. No pricing, package IDs, user-generated content
or screenshots were retained.

### Fidelity notes

The HTML in `detailed_description`, `about_the_game`, `short_description`
and `pc_requirements.minimum` is preserved **byte-for-byte as returned by
Steam**, including Steam's own tab indentation and `<br>` tags, so the
block parser is exercised against the real markup. The description text
itself was not rewritten, re-wrapped or re-indented.

`pc_requirements` only carries `minimum` because Portal 2's English
payload has no `recommended` block; the mapping keeps a missing tier
as unknown rather than inferring one.

The raw live response was not retained anywhere on disk.
