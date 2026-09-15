# Pokémon corpus avatars

Drop two images here and the app picks them up automatically — no code change needed:

- `assistant.png` — used for the assistant's chat bubble when the Pokémon corpus is selected
- `user.png` — used for your own chat bubble

Any of `.png` / `.jpg` / `.jpeg` / `.webp` works (checked in that order if more than one exists for
the same name). Square images read best as chat avatars — something in the 200×200–512×512 range is
plenty; there's no reason for these to be large files.

If neither file is present, the app falls back to plain emoji avatars, so this folder can stay
empty.

## Use fan art / your own art only

Not official sprites or artwork pulled from the games, Bulbapedia, Serebii, or similar — see
`docs/architecture.md`'s note on this and the in-app disclaimer for why. This corpus is already a
non-affiliated fan project by nature (it's a RAG demo *about* Pokémon, not a Pokémon product), and
using someone else's copyrighted art assets isn't something a "not affiliated" disclaimer fixes —
it only addresses trademark/endorsement confusion, not copyright. If the art isn't yours, credit the
artist by adding a line to this file.
