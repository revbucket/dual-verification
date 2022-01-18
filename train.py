
""" File that holds simple training loop stuff
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import utilities as utils

DEFAULT_DATASET_DIR = '~/datasets'

# ==========================================
# =           Parameters Object            =
# ==========================================

class TrainParameters(utils.ParameterObject):
    """ Holds parameters to run training.
    Training loop should just take (this, neural net)
    and spit out a trained network
    """

    def __init__(self, trainset: torch.utils.data.DataLoader,
                 valset: torch.utils.data.DataLoader,
                 num_epochs: int,
                 optimizer=None,
                 loss_function=None,
                 weight_reg=None,
                 test_after_epoch: int=1,
                 adv_attack=None,
                 use_cuda: int=1):
        """
        ARGS:
            trainset : torch.utils.data.DataLoader object holding the
                       training data
            valset : torch.utils.data.DataLoader object holding the
                     validation dataset
            num_epochs: how many epochs to run for
            optimizer: optimizer CLASS (not initialized) that takes in
                       net.parameters() to initialize.
                       Defaults to ADAM
            loss_function: function that takes in (yPred, y) and outputs a
                           scalar value. Defaults to CrossEntropy
            test_after_epoch: int that dictates how often we run tests and print
                              val loss (0 means never test)
            use_cuda: boolean that checks whether we train using GPU acceleration
                      (checks if we CAN, first)
        """

        ### Initialize default
        if optimizer is None:
            optimizer = optim.Adam  # still need params to be init here
        if loss_function is None:
            loss_function = nn.CrossEntropyLoss()
        if not torch.cuda.is_available():
            use_cuda = False

        kwargify = {'trainset': trainset,
                    'valset': valset,
                    'num_epochs': num_epochs,
                    'optimizer': optimizer,
                    'loss_function': loss_function,
                    'weight_reg': weight_reg,
                    'adv_attack': adv_attack,
                    'test_after_epoch': test_after_epoch,
                    'use_cuda': use_cuda,}

        super(TrainParameters, self).__init__(**kwargify)

    def cuda(self):
        # code to make trainset/valset cuda
        self.use_cuda = True

    def cpu(self):
        self.use_cuda = False


    def devicify(self, *tensors):
        if self.use_cuda:
            return [_.cuda() for _ in tensors]
        else:
            return [_.cpu() for _ in tensors]


# ==============================================================
# =           Main Training Loop                               =
# ==============================================================


def training_loop(network, train_params):
    """ Main training loop. Takes in only the network and params. """

    # Setup cuda stuff
    if train_params.use_cuda:
        network.cuda()
        train_params.cuda()
    else:
        network.cpu()
        train_params.cpu()

    # initialize optimizer
    optimizer = train_params.optimizer(network.parameters())


    # Build weight regularizations
    if train_params.weight_reg is not None:
        def weight_reg(wr = train_params.weight_reg, model=network):
            els = []
            if 'l1' in wr:
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                els.apppend(wr['l1'] * l1_norm)
            if 'l2' in wr:
                l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                els.append(wr['l2'] * l2_norm)
            return sum(els)
    else:
        weight_reg = lambda : 0.0


    # Build adversarial attacks
    if train_params.adv_attack is not None:
        train_params.adv_attack.attach_network(network)
        attack = lambda x, y: train_params.adv_attack.attack(x, y)
    else:
        # Just return the x input
        attack = lambda x, y: x


    # Do training loop
    for epoch_num in range(1, train_params.num_epochs + 1):
        for batch_num, (examples, labels) in enumerate(train_params.trainset):
            examples, labels = train_params.devicify(examples, labels)
            batch_size = examples.shape[0]
            #examples, labels = examples[:batch_size // 10], labels[:batch_size//10]
            optimizer.zero_grad()
            examples = attack(examples, labels)
            labels_pred = network(examples)
            loss_val = train_params.loss_function(labels_pred, labels) + weight_reg()


            loss_val.backward()
            optimizer.step()

        if (train_params.test_after_epoch > 0 and
            epoch_num % train_params.test_after_epoch == 0):
            test_str = test_validation(network, train_params)
            print('Epoch %02d | Loss : %.04f | Accuracy : %.02f | Adv Acc: %.02f' %
                   ((epoch_num,) + test_str))

    return network


def test_validation(network, train_params):
    """ Collects the loss and top1 accuracy of the network over the valset
        Returns (avg_loss, avg_accuracy *100)
    """

    total_loss = 0
    total_acc = 0
    total_count = 0
    adv_acc = 0

    for examples, labels in train_params.valset:
        with torch.no_grad():
            examples, labels = train_params.devicify(examples, labels)

            count = labels.numel()
            ypred = network(examples)

            total_loss += train_params.loss_function(ypred, labels) * count
            total_acc += (ypred.max(1)[1] == labels).sum().item()
            total_count += count

        if train_params.adv_attack is not None:
            atk_examples = train_params.adv_attack.attack(examples, labels)
            adv_acc += (network(atk_examples).max(dim=1)[1]==labels).sum().item()
    return (total_loss / total_count,
            total_acc / total_count * 100.0,
            adv_acc / total_count * 100.0)


# =============================================================
# =           ADVERSARIAL ATTACK STUFF                        =
# =============================================================

class PGD:
    def __init__(self, network, norm, eps, num_iter=10, rand_init=True,
                 iter_eps=None, lb=None, ub=None):
        """ Stores values for making an attack batch using PGD """
        self.network = network

        assert norm in [2, float('inf')]
        self.norm = norm
        self.eps = eps
        self.num_iter = num_iter
        self.rand_init = rand_init
        self.lb = lb
        self.ub = ub
        if self.num_iter == 1:
            self.iter_eps = self.eps
        else:
            self.iter_eps = iter_eps or (2 * self.eps / self.num_iter)


    def attach_network(self, network):
        self.network = network


    def _project(self, x, eta):
        """ Modies eta so the (x+eta) lives in the intersection of
            {the eps-ball around x, the lb-ub box}
        """

        if self.norm == float('inf'):
            eta.data = (torch.clamp(x + eta, x - self.eps, x + self.eps) - x).data
        if self.norm == 2:
            eta_norm = eta.norm(dim=1, keepdim=True)
            eta_bignorm = (eta_norm > self.eps).squeeze()
            eta[eta_bignorm].data.div_(eta_norm[eta_bignorm]).mul_(self.eps)
            #eta.data[eta_bignorm].div_(eta_norm[eta_bignorm] * self.eps)

        if self.lb is not None: # assume ub is also not None
            eta.data = (torch.clamp(x + eta, self.lb, self.ub) - x).data
        return eta


    def attack(self, x, y):
        """ Generates the pgd (untargeted) attack for minibatch (x, y)
            Returns the adversarial example (not the noise)
        """
        shape = x.shape
        if hasattr(self.network[0], 'input_shape'):
            x = x.view((-1,) + self.network[0].input_shape)
        else:
            x = x.view(shape[0], -1)
        machine_eps = torch.ones(x.shape[0], dtype=x.dtype) * 1e-12

        eta = torch.zeros_like(x, requires_grad=True)
        if self.rand_init:
            eta.data.uniform_(-self.eps, self.eps)
        eta = self._project(x, eta)

        loss_fxn = nn.CrossEntropyLoss()
        for iter_num in range(self.num_iter):
            # Compute gradients
            eta.grad = None
            loss_val = loss_fxn(self.network(x + eta), y)
            eta_grad = autograd.grad(outputs=loss_val, inputs=eta)[0]

            # Modify gradients based on norm
            if self.norm == float('inf'):
                eta_grad = torch.sign(eta_grad)
            else:
                grad_norm = torch.max(machine_eps, eta_grad.norm(dim=1)).view(-1, 1)
                eta_grad.div_(grad_norm)

            # Project onto feasible set
            eta = self._project(x, eta + eta_grad * self.iter_eps)

        return (x + eta).view(shape)


    def eval_attack(self, x, y):
        """ Runs an attack and evaluates the accuracy of the adv examples
            Returns: {'clean_acc': clean accuracy,
                      'adv_acc': adversarial accuracy,
                      'adv_ex': actual adversarial examples }
        """

        acc = lambda x, y: (self.network(x).max(dim=1)[1] == y).float().mean()
        adv_ex = self.attack(x, y)

        clean_acc = acc(x, y)
        adv_acc = acc(adv_ex, y)
        return {'clean_acc': clean_acc,
                'adv_acc': adv_acc,
                'adv_ex': adv_ex}





# =============================================================
# =           MNIST DATA LOADERS                              =
# =============================================================


def load_mnist_data(train_or_val, batch_size=128, shuffle=True,
                    use_cuda=True, dataset_dir=DEFAULT_DATASET_DIR,
                    extra_transforms=None):
    assert train_or_val in ['train', 'val']
    use_cuda = torch.cuda.is_available() and use_cuda

    constructor = {'batch_size': batch_size,
                   'shuffle': shuffle,
                   'num_workers': 4,
                   'pin_memory': use_cuda}

    transform_chain = transforms.ToTensor()
    if extra_transforms is not None:
        if not isinstance(extra_transforms, list):
            extra_transforms = [extra_transforms]
        transform_chain = transforms.Compose([transform_chain] + extra_transforms)

    MNIST_dataset = datasets.MNIST(root=dataset_dir,
                                   train=(train_or_val == 'train'),
                                   download=True,
                                   transform=transform_chain)


    return torch.utils.data.DataLoader(MNIST_dataset, **constructor)





# ================================================================
# =           CIFAR DATA LOADERS                                 =
# ================================================================

def load_cifar_data(train_or_val, batch_size=128, shuffle=True,
                    use_cuda=True, dataset_dir=DEFAULT_DATASET_DIR,
                    extra_transforms=None):
    assert train_or_val in ['train', 'val']
    use_cuda = torch.cuda.is_available() and use_cuda

    constructor = {'batch_size': batch_size,
                   'shuffle': shuffle,
                   'num_workers': 4,
                   'pin_memory': use_cuda}
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    transform_chain = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, 4),
                        transforms.ToTensor(),
                        normalize,])



    CIFAR_dataset = datasets.CIFAR10(root=dataset_dir,
                                     train=(train_or_val == 'train'),
                                     download=True,
                                     transform=transform_chain)
    return torch.utils.data.DataLoader(CIFAR_dataset, **constructor)

