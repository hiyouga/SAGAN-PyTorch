import os
import time
import torch
import random
import argparse
from tqdm import tqdm
from scorer import Scorer
from data_utils import load_data
from sagan_trainer import SAGAN_Trainer
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter


class Instructor:

    def __init__(self, args):
        self.args = args
        print('=> creating model')
        self.trainer = SAGAN_Trainer(args)
        print('=> initializing tensorboard')
        self.writer = SummaryWriter()
        print('=> creating scorer')
        self.scorer = Scorer(device=args.device, resize=True)
        self._print_args()

    def _print_args(self):
        print('TRAINING ARGUMENTS:')
        for arg in vars(self.args):
            print(f">>> {arg}: {getattr(self.args, arg)}")

    def train_sagan(self):
        dataloader = load_data(im_size=self.args.im_size,
                               batch_size=self.args.batch_size,
                               workers=self.args.num_workers,
                               dataset=self.args.dataset,
                               data_path=os.path.join(self.args.data_dir, self.args.dataset))
        data_iter = iter(dataloader)
        model_save_step = int(self.args.model_save_step * len(dataloader))
        fixed_z = torch.randn(self.args.batch_size, self.args.z_dim).to(self.args.device)
        real_images, _ = next(data_iter)
        real_images = (real_images * 0.5 + 0.5).clamp(0, 1)
        self.writer.add_images('real', real_images, 0)
        save_image(real_images, os.path.join(self.args.sample_dir, self.args.timestamp, 'real.png'))
        all_preds = list()
        for inputs, _ in tqdm(dataloader):
            inputs = inputs.to(self.args.device) * 0.5 + 0.5
            all_preds.append(self.scorer.get_preds(inputs))
        all_preds = torch.cat(all_preds, dim=0)
        score, _ = self.scorer.compute_score(all_preds, splits=10)
        print(f"real inception score: {score:.4f}")
        for step in range(self.args.total_step):
            ''' train discriminator'''
            self.trainer.D.train()
            self.trainer.G.train()
            try:
                real_images, _ = next(data_iter)
            except:
                data_iter = iter(dataloader)
                real_images, _ = next(data_iter)
            real_images = real_images.to(self.args.device)
            d_loss_real, d_loss_fake, g_loss_fake = self.trainer.train(real_images)
            ''' print info '''
            if (step + 1) % self.args.log_step == 0:
                print(f"step: {step + 1}/{self.args.total_step}, g_loss_fake: {g_loss_fake:.4f}")
                print(f"d_loss_real: {d_loss_real:.4f}, d_loss_fake: {d_loss_fake:.4f}")
                self.writer.add_scalar('Loss/D_real', d_loss_real, step + 1)
                self.writer.add_scalar('Loss/D_fake', d_loss_fake, step + 1)
                self.writer.add_scalar('Loss/G_fake', g_loss_fake, step + 1)
                self.writer.add_scalar('Gamma/G_attn1', self.trainer.G.attn1.gamma.mean().item(), step + 1)
                self.writer.add_scalar('Gamma/D_attn1', self.trainer.D.attn1.gamma.mean().item(), step + 1)
            ''' eval generator '''
            if (step + 1) % self.args.eval_step == 0:
                self.trainer.G.eval()
                all_preds = list()
                for i in tqdm(range(100)):
                    z = torch.randn(self.args.batch_size, self.args.z_dim).to(self.args.device)
                    inputs = self.trainer.G(z) * 0.5 + 0.5
                    all_preds.append(self.scorer.get_preds(inputs))
                all_preds = torch.cat(all_preds, dim=0)
                score, _ = self.scorer.compute_score(all_preds, splits=10)
                print(f"fake inception score: {score:.4f}")
                self.writer.add_scalar('Score/IS_fake', score, step + 1)
            ''' sample image '''
            if (step + 1) % self.args.sample_step == 0:
                self.trainer.G.eval()
                fake_images = self.trainer.G(fixed_z)
                fake_images = (fake_images * 0.5 + 0.5).clamp(0, 1)
                self.writer.add_images('fake', fake_images, step + 1)
                save_image(fake_images, os.path.join(self.args.sample_dir, self.args.timestamp, f"fake_{step + 1}.png"))
            ''' save model '''
            if (step + 1) % model_save_step == 0:
                torch.save(self.trainer.G.state_dict(), os.path.join(self.args.save_dir, self.args.timestamp, f"{step + 1}_G.pt"))
                torch.save(self.trainer.D.state_dict(), os.path.join(self.args.save_dir, self.args.timestamp, f"{step + 1}_D.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ''' dataset '''
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10'])
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--sample_dir', type=str, default='sample')
    parser.add_argument('--save_dir', type=str, default='saves')
    parser.add_argument('--num_workers', type=int, default=16)
    ''' model '''
    parser.add_argument('--adv_loss', type=str, default='wgan-gp', choices=['hinge', 'wgan-gp'])
    parser.add_argument('--im_size', type=int, default=32)
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    parser.add_argument('--lambda_gp', type=float, default=10)
    ''' optimization '''
    parser.add_argument('--total_step', type=int, default=1000000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--g_lr', type=float, default=0.0001)
    parser.add_argument('--d_lr', type=float, default=0.0004)
    parser.add_argument('--beta1', type=float, default=0.0)
    parser.add_argument('--beta2', type=float, default=0.9)
    ''' environment'''
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda'])
    parser.add_argument('--parallel', default=False, action='store_true')
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=100)
    parser.add_argument('--eval_step', type=int, default=200)
    parser.add_argument('--model_save_step', type=int, default=10)
    parser.add_argument('--timestamp', type=str, default=None)
    args = parser.parse_args()
    args.timestamp = args.timestamp if args.timestamp else str(int(time.time())) + format(random.randint(0, 999), '03')
    args.device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    for dir_name in [args.data_dir, args.sample_dir, args.save_dir]:
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
    os.mkdir(os.path.join(args.sample_dir, args.timestamp))
    os.mkdir(os.path.join(args.save_dir, args.timestamp))
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    ins = Instructor(args)
    ins.train_sagan()
