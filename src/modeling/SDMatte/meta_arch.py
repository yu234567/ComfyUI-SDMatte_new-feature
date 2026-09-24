import os
import torch
from torchvision import transforms

import folder_paths
import comfy

# Get the models directory from ComfyUI
MODEL_DIR = os.path.join(folder_paths.models_dir, "SDMatte")

# Register the SDMatte folder path with ComfyUI
folder_paths.add_model_folder_path("SDMatte", MODEL_DIR)

# 可自动下载的远程模型
MODEL_URLS = {
    "SDMatte.safetensors": "https://huggingface.co/1038lab/SDMatte/resolve/main/SDMatte.safetensors",
    "SDMatte_plus.safetensors": "https://huggingface.co/1038lab/SDMatte/resolve/main/SDMatte_plus.safetensors",
}


def _scan_local_models():
    """扫描 models/SDMatte/ 下所有模型文件"""
    exts = (".pth", ".pt", ".safetensors", ".bin")
    names = []
    if os.path.isdir(MODEL_DIR):
        try:
            for f in sorted(os.listdir(MODEL_DIR)):
                if f.lower().endswith(exts):
                    names.append(f)
        except OSError:
            pass
    return names


def _all_ckpt_choices():
    """合并本地扫描 + 远程可选，去重（本地优先）"""
    remote = list(MODEL_URLS.keys())
    local = _scan_local_models()
    merged = []
    seen = set()
    for name in local + remote:
        if name not in seen:
            seen.add(name)
            merged.append(name)
    return merged


# 模块级模型缓存：key = (pretrained_repo, ckpt_path, use_fp16)，value = SDMatte model
_SDMATTE_CACHE = {}


def _release_cache_by_ckpt(ckpt_path, device=None):
    keys_to_remove = [k for k in _SDMATTE_CACHE.keys() if k[1] == ckpt_path]
    for k in keys_to_remove:
        try:
            _SDMATTE_CACHE[k].to("cpu")
        except Exception:
            pass
        try:
            del _SDMATTE_CACHE[k]
        except Exception:
            pass
    if keys_to_remove and device is not None and device.type == 'cuda':
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass
    return len(keys_to_remove)


def _release_all_cache(device=None):
    n = len(_SDMATTE_CACHE)
    for k in list(_SDMATTE_CACHE.keys()):
        try:
            _SDMATTE_CACHE[k].to("cpu")
        except Exception:
            pass
        try:
            del _SDMATTE_CACHE[k]
        except Exception:
            pass
    if n and device is not None and device.type == 'cuda':
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass
    return n


def download_model(model_name, models_dir=MODEL_DIR, model_urls=MODEL_URLS):
    # 1) 已注册的 SDMatte 路径里找
    all_search_paths = folder_paths.get_folder_paths("SDMatte") or []
    for search_path in all_search_paths:
        check_path = os.path.join(search_path, model_name)
        if os.path.isfile(check_path):
            try:
                if os.path.getsize(check_path) > 0:
                    print(f"[SDMatte] Found model at: {check_path}")
                    return check_path
            except OSError:
                pass

    # 2) 本地 models/SDMatte/ 里找
    local_path = os.path.join(models_dir, model_name)
    if os.path.isfile(local_path) and os.path.getsize(local_path) > 0:
        print(f"[SDMatte] Using local model: {local_path}")
        return local_path

    # 3) 本地没有，且没有下载 URL → 报错
    url = model_urls.get(model_name)
    if not url:
        raise ValueError(
            f"[SDMatte] 模型 '{model_name}' 本地不存在，且未配置下载 URL。\n"
            f"请把模型文件放到：{local_path}"
        )

    # 4) 有 URL → 下载
    target_path = os.path.join(models_dir, model_name)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    if os.path.isfile(target_path):
        try:
            if os.path.getsize(target_path) > 0:
                return target_path
        except OSError:
            pass

    print(f"[SDMatte] Model '{model_name}' not found. Downloading to {target_path}...")

    tmp_path = target_path + ".tmp"

    try:
        try:
            import requests
            try:
                from tqdm import tqdm
            except Exception:
                tqdm = None

            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0) or 0)

                with open(tmp_path, 'wb') as f:
                    bar = None
                    if tqdm and total_size > 0:
                        bar = tqdm(desc=model_name, total=total_size, unit='iB', unit_scale=True, unit_divisor=1024)

                    for chunk in response.iter_content(chunk_size=1024*1024):
                        if chunk:
                            f.write(chunk)
                            if bar:
                                bar.update(len(chunk))

                    if bar:
                        bar.close()

            if total_size > 0:
                try:
                    if os.path.getsize(tmp_path) != total_size:
                        raise IOError(f"[SDMatte] Incomplete download: {os.path.getsize(tmp_path)} != {total_size}")
                except OSError:
                    raise

        except (ImportError, ModuleNotFoundError):
            import urllib.request
            urllib.request.urlretrieve(url, tmp_path)

        if os.path.isfile(target_path) and os.path.getsize(target_path) > 0:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            return target_path

        os.replace(tmp_path, target_path)
        print(f"[SDMatte] Download complete: {target_path}")
        return target_path

    except KeyboardInterrupt:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise

SDMatteCore = None


def _resize_norm_image_bchw(image_bchw: torch.Tensor, size_hw=(1024, 1024)) -> torch.Tensor:
    resize = transforms.Resize(size_hw, antialias=True)
    norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    x = resize(image_bchw)
    x = norm(x)
    return x


def _resize_mask_b1hw(mask_b1hw: torch.Tensor, size_hw=(1024, 1024)) -> torch.Tensor:
    if mask_b1hw.dim() == 3:
        mask_b1hw = mask_b1hw.unsqueeze(1)
    elif mask_b1hw.dim() == 2:
        mask_b1hw = mask_b1hw.unsqueeze(0).unsqueeze(0)

    resize = transforms.Resize(size_hw)
    out = resize(mask_b1hw)
    return out


class SDMatteApply:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (_all_ckpt_choices(), ),
                "image": ("IMAGE", {"tooltip": "需要进行抠图的输入图像"}),
                "trimap": ("MASK", {"tooltip": "三值图掩码：白色=前景，黑色=背景，灰色=未知区域"}),
                "inference_size": ([512, 640, 768, 896, 1024], {
                    "default": 1024, 
                    "tooltip": "推理分辨率，越高质量越好但速度越慢。推荐1024(最高质量)或768(平衡性能)"
                }),
                "is_transparent": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": "输入图像是否包含透明通道。如果原图有透明背景请启用"
                }),
                "output_mode": (["alpha_only", "matted_rgba", "matted_rgb"], {
                    "default": "alpha_only",
                    "tooltip": "输出模式：alpha_only=只输出遮罩；matted_rgba=透明背景抠图；matted_rgb=黑色背景抠图(推荐，避免干扰)"
                }),
                "mask_refine": ("BOOLEAN", {
                    "default": True, 
                    "tooltip": "启用遮罩优化，使用trimap约束过滤不需要的区域，减少背景干扰"
                }),
                "trimap_constraint": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "trimap约束阈值(0.0-1.0)。越低→越信任trimap，背景越干净；越高→模型越自由"
                }),
                "keep_model_loaded": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "开启后模型常驻显存，不自动卸载，第二次运行更快；关闭则每次运行后释放显存"
                }),
                "use_fp16": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "开启则模型参数转 fp16 并关闭 autocast（省显存、更快，精度略降）；关闭则 fp32 参数 + fp16 autocast（最稳）"
                }),
            },
            "optional": {
                "force_cpu": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("alpha_mask", "matted_image")
    FUNCTION = "apply_matte"
    CATEGORY = "Matting/SDMatte"

    def apply_matte(self, ckpt_name, image, trimap, inference_size, is_transparent, output_mode,
                    mask_refine, trimap_constraint, keep_model_loaded=False, use_fp16=False, force_cpu=False):
        device = comfy.model_management.get_torch_device()
        if force_cpu:
            device = torch.device('cpu')

        global SDMatteCore
        if SDMatteCore is None:
            from .src.modeling.SDMatte.meta_arch import SDMatte as SDMatteCore

        diffusers_paths = folder_paths.get_folder_paths("diffusers") or []
        pretrained_repo = None
        for path in diffusers_paths:
            candidate_path = os.path.join(path, "stable-diffusion-2-1-base")
            if os.path.isdir(candidate_path):
                pretrained_repo = candidate_path
                break

        if pretrained_repo is None:
            raise FileNotFoundError("Stable Diffusion 2.1 base model not found in diffusers directory. Please download it from https://huggingface.co/stabilityai/stable-diffusion-2-1")

        ckpt_path = download_model(ckpt_name)
        cache_key = (pretrained_repo, ckpt_path, bool(use_fp16))

        if not keep_model_loaded:
            removed = _release_cache_by_ckpt(ckpt_path, device=device)
            if removed:
                print(f"[SDMatte] 清理 {removed} 个残留缓存")

        sdmatte_model = None
        if keep_model_loaded and cache_key in _SDMATTE_CACHE:
            sdmatte_model = _SDMATTE_CACHE[cache_key]
            print(f"[SDMatte] Reusing cached model: {ckpt_path} (fp16={use_fp16})")

        if sdmatte_model is None:
            sdmatte_model = SDMatteCore(
                pretrained_model_name_or_path=pretrained_repo,
                load_weight=False,
                use_aux_input=True,
                aux_input="trimap",
                aux_input_list=["point_mask", "bbox_mask", "mask", "trimap"],
                attn_mask_aux_input=["point_mask", "bbox_mask", "mask", "trimap"],
                use_encoder_hidden_states=True,
                use_attention_mask=True,
                add_noise=False,
            )

            if ckpt_path.endswith(".safetensors"):
                from safetensors import safe_open
                state_root = {}
                with safe_open(ckpt_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_root[key] = f.get_tensor(key)
            else:
                state_root = torch.load(ckpt_path, map_location="cpu", weights_only=False)

            candidate_keys = [
                'state_dict','model_state_dict','params','weights',
                'ema','model_ema','ema_state_dict','net','module','model','unet'
            ]
            state_dict = None
            if isinstance(state_root, dict):
                for k in candidate_keys:
                    inner = state_root.get(k)
                    if isinstance(inner, dict):
                        state_dict = inner
                        break
            if state_dict is None:
                state_dict = state_root

            sdmatte_model.load_state_dict(state_dict, strict=False)
            sdmatte_model.eval()

            if use_fp16:
                try:
                    sdmatte_model = sdmatte_model.half()
                    print("[SDMatte] use_fp16=True: 模型参数已转 fp16")
                except Exception as e:
                    print(f"[SDMatte] fp16 转换失败，回退 fp32: {e}")

            if keep_model_loaded:
                _SDMATTE_CACHE[cache_key] = sdmatte_model
                print(f"[SDMatte] Model loaded and cached: {ckpt_path} (fp16={use_fp16})")
            else:
                print(f"[SDMatte] Model loaded (not cached): {ckpt_path} (fp16={use_fp16})")

        sdmatte_model.to(device)

        if device.type == 'cuda':
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            try:
                unet = getattr(sdmatte_model, 'unet', None)
                if unet is not None and hasattr(unet, 'set_attn_processor'):
                    from diffusers.models.attention_processor import SlicedAttnProcessor
                    unet.set_attn_processor(SlicedAttnProcessor(slice_size=1))
            except Exception:
                pass

        B, H, W, C = image.shape
        orig_h, orig_w = H, W

        print(f"[SDMatte] image batch={B}, trimap shape={trimap.shape if trimap is not None else None}")

        model_dtype = next(sdmatte_model.parameters()).dtype
        img_bchw = image.permute(0, 3, 1, 2).contiguous().to(device)
        img_in = _resize_norm_image_bchw(img_bchw, (int(inference_size), int(inference_size))).to(model_dtype)

        is_trans = torch.tensor([1 if is_transparent else 0] * B, device=device)
        data = {"image": img_in, "is_trans": is_trans, "caption": [""] * B}

        def to_b1hw(x):
            return _resize_mask_b1hw(x.contiguous().to(device), (int(inference_size), int(inference_size)))

        tri = to_b1hw(trimap) * 2 - 1
        tri = tri.to(model_dtype)

        B_t = tri.shape[0]
        if B_t == 1 and B > 1:
            tri = tri.expand(B, -1, -1, -1).contiguous()
            print(f"[SDMatte] trimap 广播: 1 -> {B}")
        elif B_t != B:
            raise ValueError(
                f"[SDMatte] batch 不匹配: image batch={B}, trimap batch={B_t}。"
                f"多图请配 1 张 trimap（广播），或相同数量。"
            )

        data["trimap"] = tri
        data["trimap_coords"] = torch.tensor([[0,0,1,1]]*B, dtype=tri.dtype, device=device)

        with torch.no_grad():
            if device.type == 'cuda' and not use_fp16:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    pred_alpha = sdmatte_model(data)
            else:
                pred_alpha = sdmatte_model(data)

        pred_alpha = pred_alpha.float()

        out = transforms.Resize((orig_h, orig_w))(pred_alpha)
        out = out.squeeze(1).clamp(0, 1).detach().cpu()

        if mask_refine:
            trimap_cpu = trimap.cpu()

            if trimap_cpu.dim() == 4:
                trimap_cpu = trimap_cpu.squeeze(1)
            elif trimap_cpu.dim() == 2:
                trimap_cpu = trimap_cpu.unsqueeze(0)

            if trimap_cpu.shape[0] == 1 and out.shape[0] > 1:
                trimap_cpu = trimap_cpu.expand(out.shape[0], -1, -1).contiguous()
            elif trimap_cpu.shape[0] != out.shape[0]:
                raise ValueError(
                    f"[SDMatte] mask_refine batch 不匹配: out={out.shape}, trimap={trimap_cpu.shape}"
                )

            foreground_regions = trimap_cpu > trimap_constraint
            background_regions = trimap_cpu < (1.0 - trimap_constraint)
            unknown_regions = ~(foreground_regions | background_regions)

            refined_alpha = out.clone()
            refined_alpha[background_regions] = 0.0
            refined_alpha[foreground_regions] = torch.clamp(refined_alpha[foreground_regions] * 1.2, 0, 1)

            alpha_threshold = 0.3
            low_confidence = (refined_alpha < alpha_threshold) & unknown_regions
            refined_alpha[low_confidence] = 0.0

            out = refined_alpha

        alpha_expanded = out.unsqueeze(-1)

        if output_mode == "alpha_only":
            matted_image = torch.zeros_like(image.cpu())
        elif output_mode == "matted_rgba":
            matted_image = torch.cat([
                image.cpu(),
                alpha_expanded.expand(-1, -1, -1, 1)
            ], dim=-1)
        elif output_mode == "matted_rgb":
            trimap_cpu = trimap.cpu()

            if trimap_cpu.dim() == 4:
                trimap_cpu = trimap_cpu.squeeze(1)
            elif trimap_cpu.dim() == 2:
                trimap_cpu = trimap_cpu.unsqueeze(0)

            if trimap_cpu.shape[0] == 1 and image.shape[0] > 1:
                trimap_cpu = trimap_cpu.expand(image.shape[0], -1, -1).contiguous()
            elif trimap_cpu.shape[0] != image.shape[0]:
                raise ValueError(
                    f"[SDMatte] matted_rgb batch 不匹配: image={image.shape}, trimap={trimap_cpu.shape}"
                )

            trimap_expanded = trimap_cpu.unsqueeze(-1)
            foreground_mask = (trimap_expanded > 0.2) & (alpha_expanded > 0.1)
            matted_image = image.cpu() * foreground_mask.float()
        else:
            matted_image = image.cpu() * alpha_expanded

        if not keep_model_loaded:
            try:
                sdmatte_model.to("cpu")
            except Exception:
                pass
            try:
                del sdmatte_model
            except Exception:
                pass

            removed = _release_cache_by_ckpt(ckpt_path, device=device)
            if removed:
                print(f"[SDMatte] 运行结束，清理 {removed} 个缓存")

            if device.type == 'cuda':
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                except Exception:
                    pass

        return (out, matted_image)


class ClearSDMatteVRAM:
    """放到工作流任意位置。只要它被执行，就清空 SDMatte 缓存并释放显存。
    必须接一个输入才会被触发；输出原样透传。"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trigger": ("*", {"tooltip": "接任何输出（IMAGE/MASK/LATENT 等）都行，它经过时就触发清缓存"}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("passthrough",)
    FUNCTION = "run"
    CATEGORY = "Matting/SDMatte"

    def run(self, trigger):
        device = comfy.model_management.get_torch_device()
        n = len(_SDMATTE_CACHE)

        for k in list(_SDMATTE_CACHE.keys()):
            try:
                _SDMATTE_CACHE[k].to("cpu")
            except Exception:
                pass
            try:
                del _SDMATTE_CACHE[k]
            except Exception:
                pass

        if device.type == 'cuda':
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass

        if n:
            print(f"[SDMatte] ClearSDMatteVRAM: 已清理 {n} 个缓存条目，显存已释放")
        else:
            print("[SDMatte] ClearSDMatteVRAM: 缓存为空，无需清理")

        return (trigger,)


NODE_CLASS_MAPPINGS = {
    "SDMatteApply": SDMatteApply,
    "ClearSDMatteVRAM": ClearSDMatteVRAM,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDMatteApply": "Apply SDMatte",
    "ClearSDMatteVRAM": "清理SDMatte显存",
}
