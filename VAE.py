import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image


def loss_function(recon_x, x, mu, logvar):
    """
    :param recon_x: generated image
    :param x: original image
    :param mu: latent mean of z
    :param logvar: latent log variance of z
    """
    BCE_loss = nn.BCELoss(reduction='sum')
    reconstruction_loss = BCE_loss(recon_x, x)
    KL_divergence = -0.5 * torch.sum(1+logvar-torch.exp(logvar)-mu**2)
    print(reconstruction_loss, KL_divergence)

    return reconstruction_loss + KL_divergence


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc2_mean = nn.Linear(400, 20)
        self.fc2_logvar = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

###########Encoder Part############
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2_mean(h1), self.fc2_logvar(h1)

    def reparametrization(self, mu, logvar):
        # sigma = 0.5*exp(log(sigma^2))= 0.5*exp(log(var))
        std = 0.5 * torch.exp(logvar)
        # N(mu, std^2) = N(0, 1) * std + mu
        z = torch.randn(std.size()) * std + mu
        return z


#############Decoder Part#############
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrization(mu, logvar)
        return self.decode(z), mu, logvar


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

vae = VAE()
optimizer = torch.optim.Adam(vae.parameters(), lr=0.0003)

# Training
def train(epoch):
    vae.train()
    all_loss = 0.
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to('cpu'), targets.to('cpu')
        real_imgs = torch.flatten(inputs, start_dim=1)

        # Train Discriminator
        gen_imgs, mu, logvar = vae(real_imgs)
        loss = loss_function(gen_imgs, real_imgs, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_loss += loss.item()
        print('Epoch {}, loss: {:.6f}'.format(epoch, all_loss/(batch_idx+1)))
        # Save generated images for every epoch
    fake_images = gen_imgs.view(-1, 1, 28, 28)
    save_image(fake_images, 'img/fake_images-{}.png'.format(epoch + 1))



for epoch in range(40):
    train(epoch)

torch.save(vae.state_dict(), './vae.pth')