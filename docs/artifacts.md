# Weight artifacts

The [artifact catalog](../configs/release-artifacts.json) describes the historical packed linear weights and their hashes. Load the associated native checkpoint before applying a selected-layer artifact. The artifact covers selected linear matrices. Native heads and other untouched parameters come from the pinned checkpoint.

[Upstream licenses](../licenses/README.md) document the applicable terms and attribution. Use `scripts/fetch_artifacts.py --help` to inspect the catalog and download a listed artifact. The main W4A4 benchmark uses the compute path described in [Method](method.md).
