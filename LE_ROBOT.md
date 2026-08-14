# Fine-tuning on `le_robot_rlds`

The local dataset at `modified_libero_rlds/le_robot_rlds` is registered with:

- one primary RGB image (`image`)
- two wrist RGB images (`wrist_image`, `wrist_image_2`)
- a 28-dimensional proprioceptive state (`state`)
- a 28-dimensional absolute joint-position action (`action`)
- 25 actions per predicted chunk

The 25-step chunk is a one-second horizon when data is recorded at 25 Hz. Change
`NUM_ACTIONS_CHUNK` in `prismatic/vla/constants.py` if the dataset control rate
or desired execution horizon differs.

Launch LoRA fine-tuning from the repository root with:

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir ./modified_libero_rlds \
  --dataset_name le_robot_rlds \
  --run_root_dir ./runs \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 3 \
  --use_proprio True \
  --batch_size 1 \
  --shuffle_buffer_size 1000 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 50000 \
  --max_steps 100005 \
  --save_freq 10000 \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity YOUR_WANDB_ENTITY \
  --wandb_project YOUR_WANDB_PROJECT \
  --run_id_note le_robot--parallel_dec--25_acts--28d_joint_actions--3_images--proprio
```

Keep `le_robot_rlds` in the command line: platform dimensions are selected before
the training configuration is parsed. Increase `--nproc-per-node` and
`--batch_size` only as GPU memory permits. The 28-dimensional continuous action
head is substantially larger than the original 7-dimensional LIBERO head.

Do not use the upstream `100000`-sample shuffle-buffer default for this dataset.
Each buffered sample contains three decoded images, so that setting can exhaust
host RAM before the first training step. Start with `1000`; reduce it to `256`
if the operating system still kills the process while the buffer is filling.

If `action` stores deltas rather than absolute joint targets, change the custom
dataset masks in `prismatic/vla/datasets/rlds/oxe/materialize.py`; the current
registration deliberately treats all 28 action dimensions as absolute.
