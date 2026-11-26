# Release checklist

- Publish weight archives and add download links with extraction paths.
- Record and verify the PyTorch/CUDA environment on Windows and Linux.
- Run all downstream evaluations from the packaged directory.
- Document dataset sources and versions.
- Supply historical distillation and ablation training configurations.
- Match final paper results to checkpoints and selection protocols.
- Exclude caches, private paths, and temporary results from the release.

Pretraining data is outside the current packaging scope. Third-party licenses remain in their original directories.
