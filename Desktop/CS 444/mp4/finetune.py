import numpy as np
from torch import nn
import torch
from vision_transformer import vit_b_32, ViT_B_32_Weights
from tqdm import tqdm
import numpy as np

def get_encoder(name):
    if name == 'vit_b_32':
        torch.hub.set_dir("model")
        model = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
        # 修改 ViT 模型，使其支持 prompts
        def forward(self, x, prompts=None):
            x = self._process_input(x)
            n = x.shape[0]

            batch_class_token = self.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)

            x = self.encoder(x, prompts)

            x = x[:, 0]
            return x

        # 将新的 forward 方法绑定到模型上
        model.forward = forward.__get__(model, model.__class__)
    return model

class ViTLinear(nn.Module):
    def __init__(self, n_classes, encoder_name):
        super(ViTLinear, self).__init__()

        self.vit_b = get_encoder(encoder_name)

        # 冻结 ViT 模型的前几个层
        num_layers_to_freeze = 6  # 冻结前 6 层
        for name, param in self.vit_b.named_parameters():
            if 'encoder.layers' in name:
                layer_num = int(name.split('encoder_layer_')[1].split('.')[0])
                if layer_num < num_layers_to_freeze:
                    param.requires_grad = False
            else:
                param.requires_grad = True  # 其他参数设为可训练

        # Reinitialize the head with a new layer
        self.vit_b.heads = nn.Identity()
        self.linear = nn.Linear(768, n_classes)

    def forward(self, x):
        out = self.vit_b(x)
        y = self.linear(out)
        return y
    

def test(test_loader, model, device):
    model.eval()
    total_loss, correct, n = 0., 0., 0

    for x, y in tqdm(test_loader):
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        correct += (y_hat.argmax(dim=1) == y).float().mean().item()
        loss = nn.CrossEntropyLoss()(y_hat, y)
        total_loss += loss.item()
        n += 1
    accuracy = correct / n
    loss = total_loss / n
    return loss, accuracy

def inference(test_loader, model, device, result_path):
    """Generate predicted labels for the test set."""
    model.eval()

    predictions = []
    with torch.no_grad():
        for x, _ in tqdm(test_loader):
            x = x.to(device)
            y_hat = model(x)
            pred = y_hat.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
    
    with open(result_path, "w") as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    print(f"Predictions saved to {result_path}")

class Trainer():
    def __init__(self, model, train_loader, val_loader, writer,
                 optimizer, lr, wd, momentum, 
                 scheduler, epochs, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.epochs = epochs
        self.device = device
        self.writer = writer

        self.model.to(self.device)

        # 只优化需要训练的参数
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        # 优化器设置
        if optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                             lr=lr, weight_decay=wd,
                                             momentum=momentum)
        elif optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), 
                                               lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        # 学习率调度器设置
        if scheduler == 'multi_step':
            self.lr_schedule = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[60, 80], gamma=0.1)
        elif scheduler == 'cosine':
            self.lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs)
        elif scheduler == 'warmup_cosine':
            total_steps = len(train_loader) * epochs
            warmup_steps = int(0.1 * total_steps)  # 预热占总训练步骤的 10%
            self.lr_schedule = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[
                    torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1e-7 / lr, total_iters=warmup_steps),
                    torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_steps - warmup_steps)
                ],
                milestones=[warmup_steps]
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler}")

    def train_epoch(self):
        self.model.train()
        total_loss, correct, n = 0., 0., 0

        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            y_hat = self.model(x)
            loss = nn.CrossEntropyLoss()(y_hat, y)
            loss.backward()
            self.optimizer.step()
            self.lr_schedule.step()  # 在每个优化步骤后更新学习率调度器

            total_loss += loss.item()
            correct += (y_hat.argmax(dim=1) == y).float().mean().item()
            n += 1

        return total_loss / n, correct / n
    
    def val_epoch(self):
        self.model.eval()
        total_loss, correct, n = 0., 0., 0

        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            y_hat = self.model(x)
            correct += (y_hat.argmax(dim=1) == y).float().mean().item()
            loss = nn.CrossEntropyLoss()(y_hat, y)
            total_loss += loss.item()
            n += 1
        accuracy = correct / n
        loss = total_loss / n
        return loss, accuracy

    def train(self, model_file_name, best_val_acc=-np.inf):
        best_epoch = np.NaN
        pbar = tqdm(range(self.epochs))
        for epoch in pbar:
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.val_epoch()
            self.writer.add_scalar('lr', self.lr_schedule.get_last_lr(), epoch)
            self.writer.add_scalar('val_acc', val_acc, epoch)
            self.writer.add_scalar('val_loss', val_loss, epoch)
            self.writer.add_scalar('train_acc', train_acc, epoch)
            self.writer.add_scalar('train_loss', train_loss, epoch)
            pbar.set_description("val acc: {:.4f}, train acc: {:.4f}".format(val_acc, train_acc), refresh=True)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(self.model.state_dict(), model_file_name)
            self.lr_schedule.step()
        
        return best_val_acc, best_epoch
    

class VPTDeep(nn.Module):
    def __init__(self, n_classes, encoder_name, prompt_len=10, num_layers=12, hidden_dim=768):
        super(VPTDeep, self).__init__()

        # 加载预训练的 ViT 模型
        self.vit_b = get_encoder(encoder_name)
        self.vit_b.heads = nn.Identity()  # 移除原始的分类头

        # 确保提示参数和分类头的参数可训练

        # 冻结 ViT 主干网络的参数
        for param in self.vit_b.parameters():
            param.requires_grad = False

        # 初始化可学习的提示参数，形状为 (1, num_layers, prompt_len, hidden_dim)
        self.prompt_len = prompt_len
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.prompts = nn.Parameter(torch.zeros(1, num_layers, prompt_len, hidden_dim))

        # 使用均匀分布初始化提示参数，避免使用 math 模块
        v = (6. / (2 * self.hidden_dim)) ** 0.5  # 计算得到 v ≈ 0.0625
        nn.init.uniform_(self.prompts, -v, v)

        # 新的分类头
        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        # 获取批量大小
        batch_size = x.size(0)

        # 将提示参数扩展到批量大小，形状为 (batch_size, num_layers, prompt_len, hidden_dim)
        prompts = self.prompts.expand(batch_size, -1, -1, -1)

        # 将提示参数传递给 ViT 模型
        outputs = self.vit_b(x, prompts)

        # 输出分类结果
        logits = self.head(outputs)
        return logits