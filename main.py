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
        self._print_args()

    def _print_args(self):
        print('TRAINING ARGUMENTS:')
        for arg in vars(self.args):
            print(f">>> {arg}: {getattr(self.args, arg)}")

    def train_sagan(self):
        print('=> creating model...')
        trainer = SAGAN_Trainer(args)
        writer = SummaryWriter()
        print('=> creating scorer...')
        scorer = Scorer(device=args.device, resize=True)
        print('=> loading data...')
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
        writer.add_images('real', real_images, 0)
        save_image(real_images, os.path.join(self.args.sample_dir, self.args.timestamp, 'real.png'))
        all_preds = list()
        for inputs, _ in tqdm(dataloader):
            inputs = inputs.to(self.args.device) * 0.5 + 0.5
            all_preds.append(scorer.get_preds(inputs))
        score, _ = scorer.compute_score(torch.cat(all_preds, dim=0), splits=10)
        print(f"real inception score: {score:.4f}")
        best_score = 0
        for step in range(self.args.total_step):
            ''' train sagan model '''
            trainer.D.train()
            trainer.G.train()
            try:
                real_images, _ = next(data_iter)
            except:
                data_iter = iter(dataloader)
                real_images, _ = next(data_iter)
            real_images = real_images.to(self.args.device)
            d_loss_real, d_loss_fake, g_loss_fake = trainer.train(real_images)
            ''' print info '''
            if (step + 1) % self.args.log_step == 0:
                print(f"step: {step + 1}/{self.args.total_step}, g_loss_fake: {g_loss_fake:.4f}")
                writer.add_scalar('Loss/D_real', d_loss_real, step + 1)
                writer.add_scalar('Loss/D_fake', d_loss_fake, step + 1)
                writer.add_scalar('Loss/G_fake', g_loss_fake, step + 1)
                writer.add_scalar('Score/G_attn1', trainer.G.attn1.gamma.mean().item(), step + 1)
                writer.add_scalar('Score/D_attn1', trainer.D.attn1.gamma.mean().item(), step + 1)
            ''' compute inception score '''
            if (step + 1) % self.args.eval_step == 0:
                trainer.G.eval()
                all_preds = list()
                for i in tqdm(range(self.args.sample_num)):
                    z = torch.randn(self.args.batch_size, self.args.z_dim).to(self.args.device)
                    inputs = trainer.G(z) * 0.5 + 0.5
                    all_preds.append(scorer.get_preds(inputs))
                score, _ = scorer.compute_score(torch.cat(all_preds, dim=0), splits=10)
                best_score = score if score > best_score else best_score
                print(f"fake inception score: {score:.4f}")
                writer.add_scalar('Score/IS_fake', score, step + 1)
            ''' sample image '''
            if (step + 1) % self.args.sample_step == 0:
                trainer.G.eval()
                fake_images = trainer.G(fixed_z)
                fake_images = (fake_images * 0.5 + 0.5).clamp(0, 1)
                writer.add_images('fake', fake_images, step + 1)
                save_image(fake_images, os.path.join(self.args.sample_dir, self.args.timestamp, f"fake_{step + 1}.png"))
            ''' save model '''
            if (step + 1) % model_save_step == 0:
                torch.save(trainer.G.state_dict(), os.path.join(self.args.save_dir, self.args.timestamp, f"{step + 1}_G.pt"))
                torch.save(trainer.D.state_dict(), os.path.join(self.args.save_dir, self.args.timestamp, f"{step + 1}_D.pt"))
        writer.close()
        print(f"best inception score: {best_score:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ''' dataset '''
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10'])
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--sample_dir', type=str, default='sample')
    parser.add_argument('--save_dir', type=str, default='saves')
    parser.add_argument('--num_workers', type=int, default=16)
    ''' model '''
    parser.add_argument('--im_size', type=int, default=32)
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    ''' optimization '''
    parser.add_argument('--total_step', type=int, default=200000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--g_lr', type=float, default=0.0001)
    parser.add_argument('--d_lr', type=float, default=0.0004)
    parser.add_argument('--beta1', type=float, default=0.0)
    parser.add_argument('--beta2', type=float, default=0.9)
    parser.add_argument('--lambda_gp', type=float, default=10)
    parser.add_argument('--adv_loss', type=str, default='wgan-gp', choices=['hinge', 'wgan-gp'])
    ''' environment'''
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda'])
    parser.add_argument('--parallel', default=False, action='store_true')
    parser.add_argument('--log_step', type=int, default=20)
    parser.add_argument('--sample_step', type=int, default=200)
    parser.add_argument('--eval_step', type=int, default=500)
    parser.add_argument('--model_save_step', type=int, default=10)
    parser.add_argument('--sample_num', type=int, default=100)
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
