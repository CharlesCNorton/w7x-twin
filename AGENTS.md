# Agent guide

Read to understand the repository: `README.md`, `todo.md`, `docs/physics.md`,
the code under `src/w7x_twin/`, `tests/`, `tools/`, `run.sh` and
`pyproject.toml`.

Do not read the following wholesale; they are data, not explanation.

- `results/**/*.json`: computed records. Open a single record only when a task
  needs its numbers. `results/magnetics/w7x_field.json` is a 20 MB
  interpolation table and `results/magnetics/w7x_geometry.json` a 1.4 MB
  geometry export; never open either.
- `src/w7x_twin/records/*.json`: digitised measurement inputs. Read only the
  values a task needs.
- `artifact/w7x_twin3d.html`: 22 MB generated page; never open it. The source
  is `artifact/twin3d.template.html`; read it only when changing the viewer.
- `docs/*.png`, `docs/*.jpg`: rendered figures.
