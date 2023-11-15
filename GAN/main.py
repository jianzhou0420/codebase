import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

# Training Hyperparameters
device=torch.device('cuda')
num_epochs = 100
batch_size = 64



# Define the Generator network
class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, img_shape),
            nn.Tanh()  # Tanh activation for images in the range [-1, 1]
        )
        

    def forward(self, z): # z means latent variables
        img = self.fc(z)
        return img

# Define the Discriminator network
class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(img_shape, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Sigmoid activation for binary classification
        )

    def forward(self, img):
        validity = self.fc(img)
        return validity

# Hyperparameters
latent_dim = 100
img_shape = 28 * 28  # Assuming MNIST-like images

# Create the generator and discriminator
generator = Generator(latent_dim, img_shape).to(device)
discriminator = Discriminator(img_shape).to(device)

# Loss and optimizers

optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)



# Load a dataset (e.g., MNIST)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

for epoch in tqdm(range(num_epochs)):
    for i, (real_imgs, real_labels) in enumerate(dataloader):
        real_imgs=real_imgs.to(device)
        real_labels=real_labels.to(device)
        
        # Adversarial ground truths
        batch_size = real_imgs.size(0)
        valid = torch.ones((batch_size, 1),device=device)
        fake = torch.zeros((batch_size, 1),device=device)

        # Generate a batch of random noise
        z = torch.randn((batch_size, latent_dim),device=device)

        # Generate fake images
        fake_imgs = generator(z)

        # Train the discriminator：尽可能分辨真伪
        optimizer_D.zero_grad()
        
        # BCELoss(p,y) = -[y * log(p) + (1 - y) * log(1 - p)], 
        # when y=1, BCELoss=-log(p),
        # when y=0, BCELoss=-log(1-p).
        # BCELoss(D(x),1)+BCELoss(D(G(z)),0)=-(log(D(x))+log(1-D(G(z))))
        # Here, our objective function is log D ( x^(i)) + log ( 1−D ( G ( z^(i) )))
        real_loss = F.binary_cross_entropy(discriminator(real_imgs.view(real_imgs.size(0), -1)), valid)
        fake_loss = F.binary_cross_entropy(discriminator(fake_imgs.detach().view(fake_imgs.size(0), -1)), fake)
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        # Train the generator ：尽可能骗过判别器
        optimizer_G.zero_grad()
        # 此时y=1，BCELoss=-log(D(G(z)),此时，D越把G(z)当作真的，BCELoss的绝对值越小。
        # 值得注意的是，文中的优化公式是min log(1-D(G(z)))。其实我们不需要管优化公式是什么，
        # 是先有为了让D(G(z))越等于1越好，才构建了min(log(1-D(G(z))))这个优化公式。
        # 不用min(log(1-D(G(z))))，用这里的-log(D(G(z)))也是可以的。
        # 只要目的是让D(G(z))越等于1 就行了。
        
        
        # 总结一下：在设计loss公式的时候，我们先明确让哪个参数趋近于什么数值，然后再构建loss公式。
        # 像BCE这个公式，原本意思是交叉熵，这里用得到交叉熵吗？与交叉熵有半毛钱关系？没有！仅仅为了让D(G(z))越等于1越好，才用了BCE这个公式。
        g_loss = F.binary_cross_entropy(discriminator(fake_imgs.view(fake_imgs.size(0), -1)), valid) # 
        g_loss.backward()
        optimizer_G.step()

        # Print progress
        if i%900==0:
            print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")
            torch.save(generator, 'generator1.pth')

# Save the generator model

