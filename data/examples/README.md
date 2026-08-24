# Bundled RigNet examples

Each model contains the files needed to inspect the source shape and run the
PhysSkin checkpoint:

```text
00000001/<model_id>/models/
├── model_normalized.obj
└── samples/
    ├── internal_filled.npz  # key: points, float32 (N, 3)
    └── latents.npz          # key: latents, float32 (1, 256, 768)
```

The examples are intended for inference and visualization.
