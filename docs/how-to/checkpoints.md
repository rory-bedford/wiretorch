# Save checkpoints and resume training

Give the trainer an output directory and a checkpoint interval:

```python
trainer = wt.SNNTrainer(..., checkpoint_interval=48)
trainer.train(output_dir=run_dir)
```

Every `checkpoint_interval` chunks it writes `run_dir/checkpoints/checkpoint_latest.pt`,
and `checkpoint_best.pt` when the mean loss since the previous checkpoint is the lowest
so far. A checkpoint contains the model's parameters and simulation state, the
optimiser, scaler and scheduler, the random number generators, the state carried by
the losses (such as the van Rossum filter), the recent loss history, and any gradients
accumulated since the last update. Files are written atomically, so if a job is killed
while saving, the previous checkpoint is still intact.

To resume, build the trainer with the same arguments, with a dataloader that starts at
the beginning of the data, and call `resume` with the checkpoint file or the run
directory:

```python
trainer = wt.SNNTrainer(...)            # same arguments
chunk = trainer.resume(run_dir)         # the latest checkpoint
trainer.train(output_dir=run_dir)
```

The trainer skips the data that was already used, so the resumed run continues exactly
as the original would have, even from a checkpoint taken between weight updates or
during burn-in.

Without the trainer, `model.get_checkpoint_state()` and
`model.load_checkpoint_state(state)` save and restore the simulation state, and the
model's `state_dict` its parameters.
