#import timm
from .module import *
import timm
# -----------------------------------------------------------------------------------------------------------------------------------------------------

# Encoder information

encoder_cfg = {
    "efficientnetv2_l":{
        "out_channels" : (32, 64, 96, 224, 640),
        "stage_idx" : (1, 2, 3, 5),
        "group_batch" : (2, 4, 6, 14, 40)
    },
    "efficientnetv2_m":{
        "out_channels" : (24, 48, 80, 176, 512),
        "stage_idx" : (1, 2, 3, 5),
        "group_batch" : (2, 3, 4, 11, 32)
    },
    "efficientnetv2_rw_m":{
        "out_channels" : (32, 56, 80, 192, 328),
        "stage_idx" : (1, 2, 3, 5),
        "group_batch" : (2, 4, 4, 12, 8)
    },
    "efficientnetv2_rw_s":{
        "out_channels" : (24, 48, 64, 160, 272),
        "stage_idx" : (1, 2, 3, 5),
        "group_batch" : (2, 3, 4, 10, 17)
    },
    "efficientnetv2_s":{
        "out_channels" : (24, 48, 64, 160, 256),
        "stage_idx" : (1, 2, 3, 5),
        "group_batch" : (2, 3, 4, 10, 16)
    },
    "tf_efficientnetv2_b0":{
        "out_channels" : (16, 32, 48, 112, 192),
        "stage_idx" : (1, 2, 3, 5),
        "group_batch" : (1, 2, 3, 8, 12)
    },
    "tf_efficientnetv2_b1":{
        "out_channels" : (16, 32, 48, 112, 192),
        "stage_idx" : (1, 2, 3, 5),
        "group_batch" : (1, 2, 3, 8, 12)
    },
    "tf_efficientnetv2_b2":{
        "out_channels" : (16, 32, 56, 120, 208),
        "stage_idx" : (1, 2, 3, 5),
        "group_batch" : (1, 2, 4, 8, 8)
    },
    "tf_efficientnetv2_b3":{
        "out_channels" : (16, 40, 56, 136, 232),
        "stage_idx" : (1, 2, 3, 5),
        "group_batch" : (1, 2, 4, 8, 8)
    },
    "tf_efficientnetv2_l":{
        "out_channels" : (32, 64, 96, 224, 640),
        "stage_idx" : (1, 2, 3, 5),
        "group_batch" : (2, 4, 6, 14, 40)
    },
    "tf_efficientnetv2_l_in21ft1k":{
        "out_channels" : (32, 64, 96, 224, 640),
        "stage_idx" : (1, 2, 3, 5),
        "group_batch" : (2, 4, 6, 14, 40)
    },
    "tf_efficientnetv2_l_in21k":{
        "out_channels" : (32, 64, 96, 224, 640),
        "stage_idx" : (1, 2, 3, 5),
        "group_batch" : (2, 4, 6, 14, 40)
    },
    "tf_efficientnetv2_m":{
        "out_channels" : (24, 48, 80, 176, 512),
        "stage_idx" : (1, 2, 3, 5),
        "group_batch" : (2, 3, 4, 11, 32)
    },
    "tf_efficientnetv2_m_in21ft1k":{
        "out_channels" : (24, 48, 80, 176, 512),
        "stage_idx" : (1, 2, 3, 5),
        "group_batch" : (2, 3, 4, 11, 32)
    },
    "tf_efficientnetv2_m_in21k":{
        "out_channels" : (24, 48, 80, 176, 512),
        "stage_idx" : (1, 2, 3, 5),
        "group_batch" : (2, 3, 4, 11, 32)
    },
    "tf_efficientnetv2_s":{
        "out_channels" : (24, 48, 64, 160, 256),
        "stage_idx" : (1, 2, 3, 5),
        "group_batch" : (2, 3, 4, 10, 16)
    },
    "tf_efficientnetv2_s_in21ft1k":{
        "out_channels" : (24, 48, 64, 160, 256),
        "stage_idx" : (1, 2, 3, 5),
        "group_batch" : (2, 3, 4, 10, 16)
    },
    "tf_efficientnetv2_s_in21k":{
        "out_channels" : (24, 48, 64, 160, 256),
        "stage_idx" : (1, 2, 3, 5),
        "group_batch" : (2, 3, 4, 10, 16)
    }
}
def get_timm_block(module_name):
    if module_name not in encoder_cfg.keys():
        raise RuntimeError('Unknown model (%s)' % module_name)
    config = timm.models.efficientnet.default_cfgs[module_name]
    config_key = ['input_size', 'mean', 'std', 'interpolation', 'test_input_size']
    new_config = encoder_cfg[module_name]
    for key in config_key:
        new_config[key] = config[key]
    new_config["name"] = module_name
    return new_config
# -----------------------------------------------------------------------------------------------------------------------------------------------------

# Decoder information

        

block_config = {
    "convnet":{
        "module_class" : ConvNet,
        "parameters":{
        }
    },
    "InvertedResidual":{
        "module_class" : InvertedResidualNet,
        "parameters":{
            'expand_ratio':6,
            'stride':1
        }
    },
    "resnet":{
        "module_class":ResNet,
        "parameters":{
            
        }
    },
    "SalientAttention":{
        "module_class" : AttentionBlock,
        "parameters" : {
            "middle_channel" : 128
        }
    }
}

            
        
def get_block(name):
    if name not in block_config.keys():
        raise RuntimeError('Unknown decoder block (%s)' % name)
    
    block = block_config[name]["module_class"]
    kwargs = block_config[name]["parameters"]
    kwargs["name"] = name
    return block, kwargs


    
    
