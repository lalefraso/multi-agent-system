import unittest

from lxml import html

from getnet_ingestion.miner import (
    _detect_language,
    _extract_sections,
    allowed_page_url,
    classify_url,
)


class AllowedPageUrlTests(unittest.TestCase):
    def test_accepts_only_https_getnet_english_pages(self):
        self.assertEqual(
            allowed_page_url("https://www.getnet.net/en/our-solutions/online-payments"),
            "https://www.getnet.net/en/our-solutions/online-payments",
        )
        self.assertEqual(
            allowed_page_url("https://www.getnet.net/en/"),
            "https://www.getnet.net/en",
        )

    def test_rejects_external_languages_and_query_urls(self):
        rejected = (
            "https://example.com/en/page",
            "https://www.getnet.net/pt/page",
            "https://www.getnet.net/enough/page",
            "http://www.getnet.net/en/page",
            "https://www.getnet.net.evil.example/en/page",
            "https://www.getnet.net/en/page?campaign=1",
        )
        for url in rejected:
            with self.subTest(url=url):
                self.assertIsNone(allowed_page_url(url))


class SectionExtractionTests(unittest.TestCase):
    def test_extracts_whole_heading_sections_without_external_links(self):
        page = html.fromstring(
            """
            <html><head><title>Online Payments</title></head>
            <body>
              <nav><p>Navigation text</p></nav>
              <main>
                <h1>Digital payments</h1>
                <p>Introductory content.</p>
                <h2>Get Link &amp; Pay</h2>
                <p>Create links to sell anywhere.</p>
                <ul><li>Sell using WhatsApp.</li><li>Accept debit or credit.</li></ul>
                <a href="https://outside.example/info">External link</a>
              </main>
              <footer><p>Footer text</p></footer>
            </body></html>
            """
        )
        sections = _extract_sections(page, "https://www.getnet.net/en/our-solutions/online-payments", "now")

        self.assertEqual(
            [section["section_heading"] for section in sections],
            ["Online Payments - Digital payments", "Online Payments - Get Link & Pay"],
        )
        self.assertEqual(
            sections[1]["text"].splitlines()[0],
            "Online Payments - Get Link & Pay",
        )
        self.assertIn("Create links to sell anywhere.", sections[1]["text"])
        self.assertIn("Sell using WhatsApp.", sections[1]["text"])
        self.assertEqual(sections[1]["text"].count("Get Link & Pay"), 1)
        self.assertNotIn("Page:", sections[1]["text"])
        self.assertNotIn("Section:", sections[1]["text"])
        self.assertNotIn("Navigation text", " ".join(section["text"] for section in sections))
        self.assertNotIn("Footer text", " ".join(section["text"] for section in sections))
        self.assertEqual(sections[1]["category"], "solution")

    def test_removes_duplicate_sections_and_article_recommendations(self):
        page = html.fromstring(
            """
            <html><head><title>Sample Article</title></head><body><main>
              <h1>Article title</h1>
              <p>Article introduction with enough words to identify this page as English content.</p>
              <h2>Section</h2><p>Same content.</p>
              <h2>Section</h2><p>Same content.</p>
              <h2>Latest Posts</h2>
              <h3>Recommended article title</h3><p>Recommended article teaser.</p>
            </main></body></html>
            """
        )
        sections = _extract_sections(
            page,
            "https://www.getnet.net/en/resources/blog/sample-article",
            "now",
        )

        self.assertEqual(
            [section["section_heading"] for section in sections],
            [
                "Sample Article - Article title",
                "Sample Article - Section",
            ],
        )
        self.assertEqual(len({section["id"] for section in sections}), len(sections))

    def test_includes_parent_and_subsection_headings(self):
        page = html.fromstring(
            """
            <html><head><title>Agentic Commerce</title></head><body><main>
              <h1>Turn AI interactions into Commerce</h1>
              <h2>Our Agentic Commerce Suite</h2>
              <p>Suite introduction.</p>
              <h3>MCP AI Toolkit</h3>
              <p>Orchestrate secure agentic payments.</p>
              <h3>Shopping Agent</h3>
              <p>Your AI-powered sales assistant.</p>
            </main></body></html>
            """
        )

        sections = _extract_sections(
            page,
            "https://www.getnet.net/en/our-solutions/agentic-commerce",
            "now",
        )

        self.assertEqual(
            [section["section_heading"] for section in sections],
            [
                "Agentic Commerce - Turn AI interactions into Commerce",
                "Agentic Commerce - Our Agentic Commerce Suite",
                "Agentic Commerce - Our Agentic Commerce Suite - MCP AI Toolkit",
                "Agentic Commerce - Our Agentic Commerce Suite - Shopping Agent",
            ],
        )

    def test_extracts_carousel_cards_as_semantic_subsections(self):
        page = html.fromstring(
            """
            <html><head><title>Agentic Commerce</title></head><body><main>
              <h2>Our Agentic Commerce Suite</h2>
              <p>One orchestration layer. Four complementary solutions.</p>
              <div class="carousel-item">
                <div class="imageandtextbanner-title">MCP AI Tool Kit</div>
                <div class="imageandtextbanner-subtitle">Orchestrate secure payments</div>
                <div class="imageandtextbanner-description">
                  <p>Connect merchants, agents and payment networks.</p>
                  <ul><li>Authenticate agents</li><li>Track transactions</li></ul>
                </div>
              </div>
            </main></body></html>
            """
        )

        sections = _extract_sections(
            page,
            "https://www.getnet.net/en/our-solutions/agentic-commerce",
            "now",
        )

        card = next(section for section in sections if "MCP AI Tool Kit" in section["section_heading"])
        self.assertEqual(
            card["section_heading"],
            "Agentic Commerce - Our Agentic Commerce Suite - MCP AI Tool Kit",
        )
        self.assertIn("Orchestrate secure payments", card["text"])
        self.assertIn("Connect merchants, agents and payment networks.", card["text"])
        self.assertIn("Authenticate agents", card["text"])
        self.assertNotIn("MCP AI Tool Kit", sections[0]["text"])

    def test_extracts_table_rows_by_subject_and_column(self):
        page = html.fromstring(
            """
            <html><head><title>Privacy policy</title></head><body><main>
              <h2>Data processing</h2>
              <table>
                <tr><th>Data Subject</th><th>Type of data and origin</th><th>Purpose</th><th>Legal basis</th></tr>
                <tr>
                  <th>Getnet Portal Users</th>
                  <td>Account and profile information.</td>
                  <td>Create and manage user accounts.</td>
                  <td>Performance of a contract.</td>
                </tr>
              </table>
            </main></body></html>
            """
        )

        sections = _extract_sections(
            page,
            "https://www.getnet.net/en/legal/privacy-policy",
            "now",
        )

        self.assertEqual(
            [section["section_heading"] for section in sections],
            [
                "Privacy policy - Data processing",
                "Privacy policy - Data processing - Getnet Portal Users - Data Subject",
                "Privacy policy - Data processing - Getnet Portal Users - Type of data and origin",
                "Privacy policy - Data processing - Getnet Portal Users - Purpose",
                "Privacy policy - Data processing - Getnet Portal Users - Legal basis",
            ],
        )
        self.assertEqual(
            [section["text"].splitlines()[-1] for section in sections],
            [
                "Privacy policy - Data processing",
                "Getnet Portal Users",
                "Account and profile information.",
                "Create and manage user accounts.",
                "Performance of a contract.",
            ],
        )

    def test_detects_non_english_content_inside_english_sitemap(self):
        sections = [{"text": " ".join(["Relatório de transparência salarial elaborado pelo Ministério do Trabalho e Emprego."] * 8)}]
        self.assertEqual(_detect_language(sections), "pt")

    def test_uses_english_fallback_for_short_page_text(self):
        self.assertEqual(_detect_language([{"text": "Getnet payments"}]), "en")

    def test_classifies_articles_and_legal_content_separately(self):
        self.assertEqual(
            classify_url("https://www.getnet.net/en/resources/blog/payment-trends"),
            "article",
        )
        self.assertEqual(
            classify_url("https://www.getnet.net/en/legal/privacy-policy"),
            "legal",
        )


if __name__ == "__main__":
    unittest.main()
