# Pokémon corpus avatars

Drop two images here and the app picks them up automatically, no code change needed:

- `assistant.png`: used for the assistant's chat bubble when the Pokémon corpus is selected
- `user.png`: used for your own chat bubble

Any of `.png` / `.jpg` / `.jpeg` / `.webp` works (checked in that order if more than one exists for
the same name). Square images read best as chat avatars, something in the 200×200-512×512 range is
plenty; there's no reason for these to be large files.

If neither file is present, the app falls back to plain emoji avatars, so this folder can stay
empty.

## On the current images

`assistant.jpg` / `user.png` are official game sprites (a Professor overworld sprite and a pixel-art
crop of the Trainer Red sprite), not fan art. The search for suitable fan art didn't turn up
anything usable in time. That's a conscious trade-off, made with the actual distinction in mind: the
in-app "not affiliated" disclaimer covers trademark/endorsement confusion, but it doesn't grant
copyright permission to use someone else's art. Accepted here as low risk for a small,
non-commercial, low-traffic portfolio demo, not a claim that it's risk-free. If this ever needs to
be tightened up (wider distribution, anything commercial), swap these for genuine fan art or
original art first.
