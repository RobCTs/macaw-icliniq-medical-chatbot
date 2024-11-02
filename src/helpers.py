# src/helpers.py

from torch.utils.tensorboard import SummaryWriter

def create_summary_writer(log_dir):
    """Create a TensorBoard SummaryWriter."""
    writer = SummaryWriter(log_dir)
    return writer

def log_training_progress(writer, epoch, train_loss, val_loss):
    """Log training and validation loss to TensorBoard."""
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/validation', val_loss, epoch)
