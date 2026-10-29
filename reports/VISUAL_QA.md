# PDF verification

AI internal visual inspection, not human approval. All seven pages were rendered with Poppler at 100 DPI and inspected. No clipping, overlap, missing glyphs or unreadable tables/charts were observed. Page numbers and margins are consistent.

PDF SHA-256: `fb4a0253f227a944363bf84609c95479d5f64d85911d0236d5c5435f0227cf88`.

Pypdf independently confirmed seven pages and presence of the exact aggregate score, denominator, asset bytes and freeze commit. Input hashes and the command are in build_record.json. Logs are analysis.stdout.log / analysis.stderr.log / analysis.exit.txt and build.stdout.log / build.stderr.log / build.exit.txt.
