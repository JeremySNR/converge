# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-09-02

### Added

- `fromGemini` accepts camelCase input as emitted by the `@google/genai` JS SDK (`systemInstruction`, `inlineData` / `mimeType`, `fileData` / `fileUri`, `functionCall`, `functionResponse`) alongside the existing snake_case REST spellings. Both casings may be mixed within one payload.
- `toGemini(messages, { casing: 'snake' | 'camel' })` option. The default remains `'snake'` for backwards compatibility; `'camel'` emits the SDK shape.
- `GeminiCasing` and `ToGeminiOptions` types are exported. `GeminiPayload` now declares both `system_instruction` and `systemInstruction`.
- Gemini 2.x function call `id` is preserved: `fromGemini` reads `functionCall.id` / `functionResponse.id` (falling back to the function name), and `toGemini` writes `id` on both parts whenever the canonical message carries one.
- GitHub Actions CI workflow running typecheck, tests and a build check on push and pull request.
- `package-lock.json` is now committed.
- `repository`, `homepage`, `bugs`, `author`, `engines` and `sideEffects` fields in `package.json`.

### Fixed

- camelCase Gemini payloads were silently flattened: a three-turn tool conversation from the JS SDK came back as a single user text message.
- Two parallel Gemini calls to the same function collapsed to identical tool call ids after a round-trip, because the function name was being used as the id. The bogus `_\d+` suffix-stripping regex on function names has been removed.

### Changed

- README documents both Gemini casings, the `casing` option and id handling. The "Known lossy conversions" table now covers Anthropic `thinking` / `redacted_thinking` blocks and the OpenAI Responses API format.

## [0.1.1] - 2026-04-06

### Fixed

- README rewritten: stale `ToolUsePart` type removed from the canonical format section and the import path corrected to `@jeremysnr/converge`.

### Changed

- `package.json` reformatted to standard npm layout.

## [0.1.0] - 2026-04-06

### Added

- Initial release as `@jeremysnr/converge` (the unscoped `converge` name was already taken).
- `fromOpenAI` / `toOpenAI`, `fromAnthropic` / `toAnthropic`, `fromGemini` / `toGemini` conversions through a shared canonical `Message[]` format.
- Text, image (base64 and URL), tool call and tool result support across all three providers.
- 35 tests covering per-provider conversion, cross-provider round-trips and edge cases.

[Unreleased]: https://github.com/JeremySNR/converge/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/JeremySNR/converge/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/JeremySNR/converge/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/JeremySNR/converge/releases/tag/v0.1.0
