from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests
from lxml import etree, html
from langdetect import DetectorFactory, detect_langs


SITE_HOST = "www.getnet.net"
SUPPORTED_LOCALES = ("en", "mx")
SITEMAP_URLS = {
    "en": "https://www.getnet.net/en/sitemap.xml",
    "mx": "https://www.getnet.net/mx/sitemap.xml",
}
ROBOTS_URL = "https://www.getnet.net/robots.txt"
USER_AGENT = "GetnetKnowledgeMiner/1.0"
REQUEST_TIMEOUT_SECONDS = 20
MAX_REDIRECTS = 5
REQUEST_DELAY_SECONDS = 0.25
DATA_DIR = Path(__file__).parent / "data"
RELATED_CONTENT_HEADINGS = {
    "latest posts",
    "related articles",
    "related posts",
    "you may also like",
    "últimos artículos",
    "artículos relacionados",
    "también te puede interesar",
}

HEADING_TAGS = {f"h{level}" for level in range(1, 7)}
TEXT_TAGS = {"p", "li", "blockquote", "dt", "dd"}
EXCLUDED_TAGS = {
    "aside",
    "footer",
    "form",
    "header",
    "nav",
    "script",
    "style",
    "svg",
    "noscript",
}
SITEMAP_NAMESPACE = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

LOGGER = logging.getLogger(__name__)
DetectorFactory.seed = 0


class MiningError(RuntimeError):
    """Raised when a source cannot be mined safely or completely."""


def _validate_locale(locale: str) -> str:
    if locale not in SUPPORTED_LOCALES:
        raise ValueError(f"Unsupported locale {locale!r}; expected one of {SUPPORTED_LOCALES}")
    return locale


def default_output(locale: str) -> Path:
    return DATA_DIR / f"getnet_{_validate_locale(locale)}_sections.jsonl"


def allowed_page_url(url: str, locale: str = "en") -> str | None:
    """Return a normalized same-origin page URL for the requested locale."""
    _validate_locale(locale)
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != SITE_HOST
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
        or parsed.query
        or parsed.fragment
        or not (
            parsed.path == f"/{locale}"
            or parsed.path.startswith(f"/{locale}/")
        )
    ):
        return None

    path = parsed.path.rstrip("/") or "/"
    return urlunsplit(("https", SITE_HOST, path, "", ""))


def _get_same_origin(
    session: requests.Session,
    url: str,
    *,
    accepted_types: tuple[str, ...] | None = None,
    allow_missing: bool = False,
    locale: str = "en",
) -> requests.Response:
    _validate_locale(locale)
    current_url = url
    for _ in range(MAX_REDIRECTS + 1):
        if current_url != ROBOTS_URL and allowed_page_url(current_url, locale) is None:
            raise MiningError(f"Refusing URL outside the Getnet {locale} section: {current_url}")

        try:
            response = session.get(
                current_url,
                headers={"User-Agent": USER_AGENT},
                timeout=REQUEST_TIMEOUT_SECONDS,
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            raise MiningError(f"Request failed for {current_url}: {exc}") from exc

        if response.is_redirect:
            location = response.headers.get("Location")
            if not location:
                raise MiningError(f"Redirect from {current_url} has no Location header")
            current_url = urljoin(current_url, location)
            if current_url != ROBOTS_URL and allowed_page_url(current_url, locale) is None:
                raise MiningError(f"Refusing redirect outside the Getnet {locale} section: {current_url}")
            continue

        if allow_missing and response.status_code in (404, 410):
            return response
        if response.status_code != requests.codes.ok:
            raise MiningError(f"Unexpected HTTP {response.status_code} for {current_url}")

        if accepted_types:
            content_type = response.headers.get("Content-Type", "").lower()
            if not any(media_type in content_type for media_type in accepted_types):
                raise MiningError(
                    f"Unexpected content type {content_type!r} for {current_url}"
                )
        return response

    raise MiningError(f"Too many redirects while requesting {url}")


def load_allowed_urls(session: requests.Session, locale: str = "en") -> list[str]:
    _validate_locale(locale)
    robots_response = _get_same_origin(
        session, ROBOTS_URL, accepted_types=("text/plain",), locale=locale
    )
    sitemap_response = _get_same_origin(
        session,
        SITEMAP_URLS[locale],
        accepted_types=("application/xml", "text/xml"),
        locale=locale,
    )

    sitemap_root = etree.fromstring(sitemap_response.content)
    if etree.QName(sitemap_root).localname != "urlset":
        raise MiningError(f"Expected the {locale} sitemap to contain a URL set")

    urls: list[str] = []
    seen: set[str] = set()
    for location in sitemap_root.xpath("//sm:url/sm:loc/text()", namespaces=SITEMAP_NAMESPACE):
        normalized = allowed_page_url(location.strip(), locale)
        if normalized and normalized not in seen:
            seen.add(normalized)
            urls.append(normalized)

    if not urls:
        raise MiningError(f"The {locale} sitemap did not contain any in-scope URLs")

    disallowed_paths: list[str] = []
    current_user_agent = None
    for raw_line in robots_response.text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        directive, value = (part.strip() for part in line.split(":", 1))
        directive = directive.lower()
        if directive == "user-agent":
            current_user_agent = value
        elif directive == "disallow" and current_user_agent in ("*", USER_AGENT) and value:
            disallowed_paths.append(value)

    blocked = [
        url for url in urls
        if any(urlsplit(url).path.startswith(path) for path in disallowed_paths)
    ]
    if blocked:
        raise MiningError(f"Sitemap includes URLs disallowed by robots.txt: {blocked}")
    return urls


def classify_url(url: str) -> str:
    path = urlsplit(url).path
    if path.rstrip("/") in ("/en", "/mx"):
        return "home"
    if "/our-solutions/" in path or "/nuestras-soluciones/" in path:
        return "solution"
    if "/your-business/" in path or "/tu-negocio/" in path:
        return "business"
    if "/resources/blog/" in path or "/recursos/blog/" in path:
        return "article"
    if "/resources/client-stories/" in path:
        return "client_story"
    if "/resources/" in path or "/recursos/" in path:
        return "resource"
    if "/legal/" in path:
        return "legal"
    if "/corporate/" in path:
        return "corporate"
    if path.startswith("/en/developers"):
        return "developer"
    if path.startswith(("/en/partners", "/mx/partners")):
        return "partner"
    if path.startswith(("/en/contact", "/mx/contacto")):
        return "contact"
    return "general"


def _clean_text(value: str) -> str:
    return " ".join(value.split())


def _detect_language(sections: list[dict[str, str]], fallback: str = "en") -> str:
    page_text = _clean_text(" ".join(section["text"] for section in sections))
    if len(page_text) < 100:
        return fallback

    detections = detect_langs(page_text)
    if detections and detections[0].prob >= 0.9:
        return detections[0].lang
    return fallback


def _extract_sections(
    page: html.HtmlElement,
    url: str,
    extracted_at: str,
    language: str = "en",
) -> list[dict[str, str]]:
    for node in page.xpath("//script|//style|//noscript|//svg|//nav|//header|//footer|//aside|//form"):
        parent = node.getparent()
        if parent is not None:
            parent.remove(node)

    content_roots = page.xpath("//main")
    if not content_roots:
        content_roots = page.xpath("//article")
    if not content_roots:
        content_roots = page.xpath("//body")
    root = content_roots[0] if content_roots else page
    page_title = _clean_text(" ".join(root.xpath("ancestor-or-self::html/head/title//text()")))
    if not page_title:
        page_title = _clean_text(" ".join(page.xpath("//title//text()")))

    sections: list[dict[str, str]] = []
    seen_section_ids: set[str] = set()
    current_heading: str | None = None
    heading_path: list[tuple[int, str]] = []
    current_text: list[str] = []
    skipped_nodes: set[html.HtmlElement] = set()

    def append_section(heading: str, body: str) -> None:
        if not heading and not body:
            return
        text = "\n".join(part for part in (heading, body) if part)
        content_id = hashlib.sha256(f"{url}\n{heading}\n{text}".encode("utf-8")).hexdigest()
        if content_id in seen_section_ids:
            return
        seen_section_ids.add(content_id)
        sections.append(
            {
                "id": content_id,
                "url": url,
                "page_title": page_title,
                "section_heading": heading or "Overview",
                "category": classify_url(url),
                "language": language,
                "extracted_at": extracted_at,
                "text": text,
            }
        )

    def flush() -> None:
        body = "\n".join(current_text)
        heading = current_heading or "Overview"
        if current_heading is None and not body:
            return
        append_section(heading, body)

    for node in root.iterdescendants():
        if node in skipped_nodes:
            continue
        tag = node.tag.lower() if isinstance(node.tag, str) else ""
        if tag in EXCLUDED_TAGS:
            continue
        if "carousel-item" in (node.get("class") or "").split():
            flush()
            title_nodes = node.xpath(
                './/*[contains(concat(" ", normalize-space(@class), " "), '
                '" imageandtextbanner-title ")]'
            )
            card_title = _clean_text(title_nodes[0].text_content()) if title_nodes else ""
            if card_title:
                heading = " - ".join(
                    part for part in (current_heading or page_title, card_title) if part
                )
                card_text: list[str] = []
                for selector in (
                    './/*[contains(concat(" ", normalize-space(@class), " "), '
                    '" imageandtextbanner-pretitle ")]',
                    './/*[contains(concat(" ", normalize-space(@class), " "), '
                    '" imageandtextbanner-subtitle ")]',
                ):
                    card_text.extend(
                        text
                        for text in (
                            _clean_text(element.text_content())
                            for element in node.xpath(selector)
                        )
                        if text
                    )
                descriptions = node.xpath(
                    './/*[contains(concat(" ", normalize-space(@class), " "), '
                    '" imageandtextbanner-description ")]'
                )
                for description in descriptions:
                    card_text.extend(
                        text
                        for text in (
                            _clean_text(element.text_content())
                            for element in description.xpath(
                                ".//p|.//li|.//blockquote|.//dt|.//dd|.//td|.//th"
                            )
                        )
                        if text
                    )
                append_section(heading, "\n".join(card_text))
            skipped_nodes.update(node.iterdescendants())
            continue
        if tag in HEADING_TAGS:
            heading_text = _clean_text(node.text_content())
            if (
                classify_url(url) == "article"
                and heading_text.casefold() in RELATED_CONTENT_HEADINGS
            ):
                flush()
                break
            flush()
            level = int(tag[1])
            if level == 1:
                heading_path.clear()
                current_heading = " - ".join(
                    part for part in (page_title, heading_text) if part
                )
            else:
                while heading_path and heading_path[-1][0] >= level:
                    heading_path.pop()
                heading_path.append((level, heading_text))
                current_heading = " - ".join(
                    part
                    for part in (
                        page_title,
                        *(heading for _, heading in heading_path),
                    )
                    if part
                )
            current_text = []
        elif tag == "tr":
            cells = node.xpath("./th|./td")
            if cells and all(cell.tag.lower() == "th" for cell in cells):
                skipped_nodes.update(node.iterdescendants())
                continue

            tables = node.xpath("ancestor::table[1]")
            header_cells = []
            if tables:
                header_rows = [
                    row
                    for row in tables[0].xpath(".//tr")
                    if (row_cells := row.xpath("./th|./td"))
                    and all(cell.tag.lower() == "th" for cell in row_cells)
                ]
                if header_rows:
                    header_cells = header_rows[0].xpath("./th|./td")

            if cells and len(header_cells) == len(cells):
                flush()
                base_heading = current_heading or page_title or "Overview"
                subject = _clean_text(cells[0].text_content())
                for index, cell in enumerate(cells):
                    value = _clean_text(cell.text_content())
                    if not value:
                        continue
                    column = _clean_text(header_cells[index].text_content())
                    heading = " - ".join(
                        part for part in (base_heading, subject, column) if part
                    )
                    append_section(heading, value)
            else:
                row_text = _clean_text(" ".join(cell.text_content() for cell in cells))
                if row_text:
                    current_text.append(row_text)
            skipped_nodes.update(node.iterdescendants())
        elif tag in TEXT_TAGS:
            text = _clean_text(node.text_content())
            if text:
                current_text.append(text)

    flush()

    if not sections:
        fallback_text = _clean_text(root.text_content())
        if fallback_text:
            content_id = hashlib.sha256(f"{url}\n{fallback_text}".encode("utf-8")).hexdigest()
            sections.append(
                {
                    "id": content_id,
                    "url": url,
                    "page_title": page_title,
                    "section_heading": "Page content",
                    "category": classify_url(url),
                    "language": language,
                    "extracted_at": extracted_at,
                    "text": fallback_text,
                }
            )
    return sections


def _homepage_links(page: html.HtmlElement, url: str, locale: str) -> list[str]:
    links: list[str] = []
    seen: set[str] = set()
    for href in page.xpath("//a[@href]/@href"):
        normalized = allowed_page_url(urljoin(url, href), locale)
        if normalized and normalized not in seen:
            seen.add(normalized)
            links.append(normalized)
    return links


def mine(output_path: Path | None = None, locale: str = "en") -> int:
    _validate_locale(locale)
    output_path = output_path or default_output(locale)
    session = requests.Session()
    urls = load_allowed_urls(session, locale)
    seen_urls = set(urls)
    extracted_at = datetime.now(timezone.utc).isoformat()
    all_sections: list[dict[str, str]] = []
    included_pages = 0

    for index, url in enumerate(urls, start=1):
        if index > 1:
            time.sleep(REQUEST_DELAY_SECONDS)
        response = _get_same_origin(
            session,
            url,
            accepted_types=("text/html",),
            allow_missing=True,
            locale=locale,
        )
        if response.status_code in (404, 410):
            LOGGER.warning("Skipping missing sitemap page %s: HTTP %s", url, response.status_code)
            continue
        final_url = allowed_page_url(response.url, locale)
        if final_url is None:
            raise MiningError(f"Page resolved outside the Getnet {locale} section: {response.url}")
        page = html.fromstring(response.content, base_url=final_url)
        if locale == "mx" and urlsplit(final_url).path.rstrip("/") == "/mx":
            for linked_url in _homepage_links(page, final_url, locale):
                if linked_url not in seen_urls:
                    seen_urls.add(linked_url)
                    urls.append(linked_url)

        language = "en" if locale == "en" else "es"
        page_sections = _extract_sections(page, final_url, extracted_at, language)
        if not page_sections:
            raise MiningError(f"No readable sections were extracted from {final_url}")
        detected_language = _detect_language(page_sections, language)
        if detected_language != language:
            LOGGER.warning(
                "Skipping page with unexpected language (%s, expected %s): %s",
                detected_language,
                language,
                final_url,
            )
            continue
        all_sections.extend(page_sections)
        included_pages += 1
        LOGGER.info("[%d/%d] %s: %d sections", index, len(urls), final_url, len(page_sections))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as output_file:
            for section in all_sections:
                output_file.write(json.dumps(section, ensure_ascii=False) + "\n")
        temporary_path.replace(output_path)
    except OSError as exc:
        temporary_path.unlink(missing_ok=True)
        raise MiningError(f"Could not write corpus to {output_path}: {exc}") from exc

    LOGGER.info(
        "Saved %d sections from %d %s pages (%d sitemap and homepage URLs checked) to %s",
        len(all_sections),
        included_pages,
        locale,
        len(urls),
        output_path,
    )
    return len(all_sections)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine a Getnet locale into section records.")
    parser.add_argument(
        "--locale",
        choices=SUPPORTED_LOCALES,
        default="en",
        help="Website locale to mine (default: en)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSONL output path (defaults to getnet_<locale>_sections.jsonl)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        mine(args.output, args.locale)
    except MiningError as exc:
        parser.exit(1, f"Mining failed: {exc}\n")


if __name__ == "__main__":
    main()
