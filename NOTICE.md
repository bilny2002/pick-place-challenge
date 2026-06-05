# Third-party assets

This project fetches assets at runtime; none are vendored in this repository.

## Robot models — MuJoCo Menagerie (via `robot_descriptions`)
Franka Emika Panda and Robotiq 2F-85 MJCFs come from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie),
licensed under the Apache License 2.0 (with per-model licenses therein).

## Ball mesh & room backdrop — Poly Haven
The ball mesh (`baseball_01`) and the indoor HDRI used for the skybox come from
[Poly Haven](https://polyhaven.com), released under **CC0** (public domain).

## Bowl & other objects — Google Scanned Objects (via `mujoco_scanned_objects`)
The pickable objects are fetched from
[kevinzakka/mujoco_scanned_objects](https://github.com/kevinzakka/mujoco_scanned_objects).
- The 3D assets (OBJ meshes, PNG textures) are from Google's
  [Scanned Objects](https://blog.research.google/2022/06/scanned-objects-by-google-research.html)
  dataset, licensed **CC-BY-4.0** (© Google LLC).
- The MJCF wrappers are licensed MIT (© Kevin Zakka).
