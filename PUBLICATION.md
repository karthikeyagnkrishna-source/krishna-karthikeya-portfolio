# Public source release

At the owner's request, the repository now publishes the complete UGP source collection found in the audited local project folders, historical notebooks, source archives, and saved server snapshots. This includes **981 source/configuration/document files**, **137 notebook variants**, and their code-cell exports. See the [full index](ugp/source-history/README.md) and [manifest](ugp/source-history/SOURCE_MANIFEST.json) for exact coverage.

CS781 code remains excluded. Credentials, private keys, personal contact details, machine-specific SSH/path values, raw datasets, learned weights, and cached feature data are excluded or redacted. Installed environments and upstream dependency checkouts are excluded from the source collection; obtain them from their original owners under their licenses. No blanket license or sole-authorship claim is applied to shared inputs.

Original and modified terrain-model handoffs are now public as source snapshots. Their provenance remains explicit: supplied design, local implementation, and historical causal-pilot revisions. Availability of code does not establish completed training or a validated physical model.

Notebook outputs, attachments, metadata and execution state are removed. Original executed-cell and error-output counts are retained as metadata in the manifest. Source logic is preserved apart from credential removal and replacement of machine-specific paths with `/path/to/ugp/...`. These placeholders require configuration, and historical dependencies may require their original Python/CUDA environment.

The RTX host was unreachable during this pass. This release makes no claim to include code that exists only on that inaccessible server. The full source collection has not been retrained or evaluated by this publication step.
