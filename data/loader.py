from torch.utils.data import DataLoader

def create_loader(dataset, config, is_train=True):
    """Paper's data loading defaults from Appendix A.1"""
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        shuffle=is_train,
        persistent_workers=True,
        drop_last=is_train  # Paper recommends for stability
    )