# Model verification, 22 October 2026

- Artifact SHA-256: `92ac184550783b012fc0de66b4a2c025661bf9d0779eab64c3d30d797d3380e7`.
- Synthetic raw-input profiles checked against R: 39.
- Maximum exported-profile probability error in R: 0.
- Tree-threshold boundary profiles checked in R: 91.
- Maximum boundary probability error in R: 0.
- Development rows checked for equivalent feature encoding: 1,867.
- Python `verify_model.py` loads and verifies every exported synthetic profile.
- Tests cover tree traversal, threshold boundaries, encoding, class order, invalid inputs, and display-label handling.

The actual tree contents are from the newly exported model, with `nodesize=25` and `maxnodes=30`. Display names were reviewed against the new exported group means and do not affect predictions. Streamlit rendering was not executed locally because its runtime dependencies are unavailable in this execution environment.
