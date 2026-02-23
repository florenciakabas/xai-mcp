# ADR-003: stdio Transport for PoC, Streamable HTTP for Production

## Status: Accepted
## Date: 2025-02-20

## Context

The MCP server needs a transport layer to communicate with VS Code Copilot.
Two options are available in FastMCP out of the box: stdio (standard input/output)
and Streamable HTTP (network socket). The PoC must run locally with no
infrastructure; production must be accessible over the network.

## Decision

Use **stdio transport** for the PoC (local development). Switch to
**Streamable HTTP transport** for production. FastMCP supports both with a
single flag change — no application code changes required.

**PoC (current):**
```python
mcp.run(transport="stdio")
```

**Production:**
```python
mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
```

The `.vscode/mcp.json` configuration points to the local stdio process.
In production, it would point to the remote HTTP URL instead:
```json
{ "url": "https://your-databricks-app.azuredatabricks.net/mcp" }
```

## Consequences

- ✅ Zero infrastructure needed for PoC — runs on any developer laptop
- ✅ No code changes required to switch transports — only config changes
- ✅ Same tool contracts, same narratives, same tests in both environments
- ✅ FastMCP's `streamable-http` mode supports concurrent requests (PoC stdio does not)
- ⚠️ stdio transport is single-client only — not suitable for shared team use
- ⚠️ HTTP transport requires authentication layer (OAuth via Databricks or company SSO)

## Alternatives Considered

- **SSE (Server-Sent Events)**: Older MCP transport, being phased out. Rejected.
- **WebSocket**: More complex, no advantage over Streamable HTTP for this use case. Rejected.
