import argparse, os
import cv2
import datasets
import random
from datasets import load_dataset
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision import transforms
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
from imwatermark import WatermarkEncoder

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

# student_validate_only = False

# if student_validate_only:
torch.set_grad_enabled(False)
# else:
#     torch.set_grad_enabled(True)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, student_ckpt, device=torch.device("cuda"), verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    if device == torch.device("cuda"):
        model.cuda()
    elif device == torch.device("cpu"):
        model.cpu()
        model.cond_stage_model.device = "cpu"
    else:
        raise ValueError(f"Incorrect device name. Received: {device}")
    model.eval()

    unet_config = config.model.params.unet_config
    unet_model = instantiate_from_config(unet_config)
    # if student_validate_only:
    #     sd = torch.load(student_ckpt, map_location="cpu")
    a, b = unet_model.load_state_dict(sd, strict=False)
    if len(a) > 0 and verbose:
        print("missing keys:")
        print(a)
    if len(b) > 0 and verbose:
        print("unexpected keys:")
        print(b)
    if device == torch.device("cuda"):
        unet_model.cuda()
    elif device == torch.device("cpu"):
        unet_model.cpu()

    return model, unet_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a professional photograph of an astronaut riding a triceratops",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm",
        action='store_true',
        help="use DPM (2) sampler",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=3,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=9.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file, separated by newlines",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v2-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="repeat each prompt in file this often",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device on which Stable Diffusion will be run",
        choices=["cpu", "cuda"],
        default="cpu"
    )
    parser.add_argument(
        "--torchscript",
        action='store_true',
        help="Use TorchScript",
    )
    parser.add_argument(
        "--ipex",
        action='store_true',
        help="Use Intel® Extension for PyTorch*",
    )
    parser.add_argument(
        "--bf16",
        action='store_true',
        help="Use bfloat16",
    )
    opt = parser.parse_args()
    return opt


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img

def compute_snr(model, timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = model.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        alpha = sqrt_alphas_cumprod[timesteps].float()
        
        sigma = sqrt_one_minus_alphas_cumprod[timesteps].float()

        # Compute SNR.
        snr = (alpha / sigma) ** 2 + 1
        return snr


def add_noise(
        model,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        alphas_cumprod = model.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        # timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

def one_batch(
        is_train,
        batch,
        model, 
        batch_size, 
        sampler,
        input_shape,
        unet_model,
        device,
        mse,
        optimizer):
    prompts = batch["input_texts"]
    # print(prompts)
    distribution = model.encode_first_stage(batch["pixel_values"].to("cuda"))
    x_input = model.get_first_stage_encoding(distribution)
    index = random.randint(0, 1)
    noise = torch.randn_like(x_input)
    noisy_img = add_noise(model, x_input, noise, index)
    snr = compute_snr(model, index).to("cpu").item()
    uc = None
    if opt.scale != 1.0:
        uc = model.get_learned_conditioning(batch_size * [""])
    c = model.get_learned_conditioning(prompts)
    # model.eval()
    # with torch.no_grad():
    #     outs = sampler.ddim_sampling_distill(c, input_shape,
    #                                         callback=None,
    #                                         img_callback=None,
    #                                         quantize_denoised=False,
    #                                         mask=None, x0=None,
    #                                         ddim_use_original_steps=False,
    #                                         noise_dropout=0,
    #                                         temperature=1,
    #                                         score_corrector=None,
    #                                         corrector_kwargs=None,
    #                                         x_T=noisy_img,
    #                                         index = index,
    #                                         log_every_t=100,
    #                                         unconditional_guidance_scale=opt.scale,
    #                                         unconditional_conditioning=uc,
    #                                         dynamic_threshold=None,
    #                                         ucg_schedule=None
    #                                         )
    t_in = torch.full((batch_size,), index, device=device, dtype=torch.long)

    unet_model.eval()

    student_out = unet_model(noisy_img, t_in, c)

    x_in = torch.randn(2, 4, 64, 64, dtype=torch.float32).to("cuda")
    t_in = torch.zeros(2, dtype=torch.float32).to("cuda")
    c_in = torch.randn(2, 77, 768, dtype=torch.float32).to("cuda")
    # torch_out = sd_model(x_in, t_in, c_in)
    # self.diffusion_model.to("cpu")
    torch.onnx.export(unet_model,               # model being run
        (x_in, t_in, c_in),                         # model input (or a tuple for multiple inputs)
        "./sd_webui_unet_1024.onnx",   # where to save the model (can be a file or file-like object)
        export_params=True,        # store the trained parameter weights inside the model file
        opset_version=16,          # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        keep_initializers_as_inputs=True,
        input_names = ['x_in', "t_in", "c_in"],   # the model's input names
        output_names = ['latent'], # the model's output names
        dynamic_axes={'x_in' : {0 : 'bs'},    # variable length axes
                    't_in' : {0 : 'bs'},
                    'c_in' : {0 : 'bs'},
                    'latent' : {0 : 'bs'}})
    

    # square_loss = mse(student_out, outs)

    return 1


def main(opt):

    DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
    }
    datasets.utils.logging.set_verbosity_warning()


    dataset_train = load_dataset(
        "lambdalabs/pokemon-blip-captions",
        None,
        split = 'train[:20%]',
        cache_dir="/data/stable-diffusion-all/stable-diffusion-21/cache",
    )

    dataset_test = load_dataset(
        "lambdalabs/pokemon-blip-captions",
        None,
        split = 'train[80%:]',
        cache_dir="/data/stable-diffusion-all/stable-diffusion-21/cache",
    )

    column_names = dataset_train.column_names

    image_column = "image"
    if image_column not in column_names:
        raise ValueError(
            f"--image_column' value 'image' needs to be one of: {', '.join(column_names)}"
        )
    

    caption_column = "text"
    if caption_column not in column_names:
        raise ValueError(
            f"--caption_column' value 'text' needs to be one of: {', '.join(column_names)}"
        )

    resolution = 512
    center_crop = True
    random_flip = True
    train_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    
    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        return examples
    
    train_dataset = dataset_train.with_transform(preprocess_train)
    test_dataset = dataset_test.with_transform(preprocess_train)
    
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_texts = [example["text"] for example in examples]
        return {"pixel_values": pixel_values, "input_texts": input_texts}
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=opt.n_samples,
        num_workers=0,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn= collate_fn,
        batch_size = opt.n_samples,
        num_workers=0,
    )

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    device = torch.device("cuda") if opt.device == "cuda" else torch.device("cpu")
    # unet_model = load_student_model(config, device=torch.device("cuda"))
    model, unet_model = load_model_from_config(config, f"{opt.ckpt}", "./student_model_"+str(0),  device) 
    
    if opt.plms:
        sampler = PLMSSampler(model, device=device)
    elif opt.dpm:
        sampler = DPMSolverSampler(model, device=device)
    else:
        sampler = DDIMSampler(model, device=device)

    mse = torch.nn.MSELoss(reduce=sum)
    optimizer_cls = torch.optim.Adam
    optimizer = optimizer_cls(
        unet_model.parameters(),
        lr=1e-4
    )

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "SDV2"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = [p for p in data for i in range(opt.repeat)]
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    sampler.make_schedule(ddim_num_steps=opt.steps, ddim_eta=opt.ddim_eta, verbose=False)
    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
    C, H, W = shape
    size = (batch_size, C, H, W)
    print(f'Data shape for DDIM sampling is {size}, eta {opt.ddim_eta}')

    epoch = 1
    precision_scope = autocast if opt.precision=="autocast" or opt.bf16 else nullcontext
    count = 0
    with precision_scope(opt.device), model.ema_scope():
        #train
        for n in range(epoch):
            # if not student_validate_only:
            #     for batch in train_dataloader:
            #         count = count + 1
            #         square_loss = one_batch(
            #                         True,
            #                         batch,
            #                         model, 
            #                         batch_size, 
            #                         sampler,
            #                         size,
            #                         unet_model,
            #                         device,
            #                         mse,
            #                         optimizer)
                    
            #         if count%10 == 0:
            #             print("current loss: ", square_loss)
                
            #     torch.save(unet_model.state_dict(), "./student_model_"+str(n))
            
            # validate
            loss_list = []
            for batch in test_dataloader:
                count = count + 1
                square_loss = one_batch(
                                False,
                                batch,
                                model, 
                                batch_size, 
                                sampler,
                                size,
                                unet_model,
                                device,
                                mse,
                                optimizer)
                loss_list.append(square_loss)
                break
            
            validate_loss_mean = sum(loss_list) / len(loss_list)
            print("validate loss mean is: ", validate_loss_mean)
            

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    opt = parse_args()
    main(opt)