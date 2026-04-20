from src.demo.core import SAETester
from src.sae_training.loaders import get_sae_and_vit
from src.data.utils import get_all_classnames, get_max_acts_and_images, load_datasets


def load_sae_tester(sae_path, include_imagenet=False):
    datasets   = load_datasets(include_imagenet=include_imagenet)
    classnames = get_all_classnames(datasets)

    root        = "./out/feature_data"
    sae_runname = "sae_base"
    vit_name    = "base"

    max_act_imgs, mean_acts = get_max_acts_and_images(
        datasets, root, sae_runname, vit_name
    )

    backbone = "openai/clip-vit-base-patch16"

    sae, vit, cfg = get_sae_and_vit(
        sae_path=sae_path,
        vit_type="base",
        device="cpu",
        backbone=backbone,
    )
    sae_clip = SAETester(vit, cfg, sae, mean_acts, max_act_imgs, datasets, classnames)

    sae, vit_maple, cfg_maple = get_sae_and_vit(
        sae_path=sae_path,
        vit_type="maple",
        device="cpu",
        backbone=backbone,
        model_path="./data/clip/maple/imagenet/model.pth.tar-2",
        config_path="./configs/models/maple/vit_b16_c2_ep5_batch4_2ctx.yaml",
        classnames=classnames.get("imagenet", []),
    )
    sae_maple = SAETester(
        vit_maple, cfg_maple, sae, mean_acts, max_act_imgs, datasets, classnames
    )

    return {"CLIP": sae_clip, "MaPLE-imagenet": sae_maple}
