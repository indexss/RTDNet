import wandb

project="real-my-eve-transformer"
entity="csbarista"
config={
        "learning_rate": 0.002,
        "architecture": "transformer",
        "dataset": "EVE dataset",
        "epochs": 10,
    }

def init_wandb():
    run = wandb.init(
    project=project,
    entity=entity,
    config=config
    )
    return run