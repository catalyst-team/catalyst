#!/usr/bin/env python
# coding: utf-8
# flake8: noqa

# In[1]:

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# In[2]:

from catalyst.rl.agent import StateNet, StateActionNet

# In[3]:

import torch
import torch.nn as nn


def _get_network_output(net: nn.Module, *input_shapes):
    inputs = []
    for input_shape in input_shapes:
        if isinstance(input_shape, dict):
            input_t = {}
            for key, input_shape_ in input_shape.items():
                input_t[key] = torch.Tensor(torch.randn((1, ) + input_shape_))
        else:
            input_t = torch.Tensor(torch.randn((1, ) + input_shape))
        inputs.append(input_t)
    output_t = net(*inputs)
    return output_t


# # FF base

# In[4]:

history_len = 5
observation_shape = (24, )

state_shape = (
    history_len,
    observation_shape[0],
)
observation_net_params = {
    "_network_type": "linear",
    "history_len": history_len,
    "features": [128],
    "use_bias": False,
    "use_normalization": True,
    "dropout_rate": 0.1,
    "activation": "LeakyReLU",
}
# observation_net_params = None

aggregation_net_params = {}
aggregation_net_params = None

main_net_params = {
    "features": [128],
    "use_bias": False,
    "use_normalization": True,
    "dropout_rate": 0.1,
    "activation": "LeakyReLU",
}

# In[5]:

net = StateNet.get_from_params(
    state_shape=state_shape,
    observation_net_params=observation_net_params,
    aggregation_net_params=aggregation_net_params,
    main_net_params=main_net_params,
)
print(net)

out = _get_network_output(net, state_shape)
print(out.shape, out.nelement())

# ---

# # FF no observation

# In[6]:

history_len = 5
observation_shape = (24, )

state_shape = (
    history_len,
    observation_shape[0],
)

observation_net_params = None
aggregation_net_params = None
main_net_params = {
    "features": [128],
    "use_bias": False,
    "use_normalization": True,
    "dropout_rate": 0.1,
    "activation": "LeakyReLU",
}

# In[7]:

net = StateNet.get_from_params(
    state_shape=state_shape,
    observation_net_params=observation_net_params,
    aggregation_net_params=aggregation_net_params,
    main_net_params=main_net_params,
)
print(net)

out = _get_network_output(net, state_shape)
print(out.shape, out.nelement())

# ---

# # CNN

# In[8]:

history_len = 5
observation_shape = (3, 80, 80)

state_shape = (history_len, ) + observation_shape

observation_net_params = {
    "_network_type": "convolution",
    "history_len": history_len,
    "channels": [16, 32, 16],
    "use_bias": False,
    "use_normalization": False,
    "dropout_rate": 0.1,
    "activation": "LeakyReLU",
}
# observation_net_params = None
aggregation_net_params = None
main_net_params = {
    "features": [128],
    "use_bias": False,
    "use_normalization": True,
    "dropout_rate": 0.1,
    "activation": "LeakyReLU",
}

# In[9]:

net = StateNet.get_from_params(
    state_shape=state_shape,
    observation_net_params=observation_net_params,
    aggregation_net_params=aggregation_net_params,
    main_net_params=main_net_params,
)
print(net)

out = _get_network_output(net, state_shape)
print(out.shape, out.nelement())

# ---

# # CNN - no observation

# In[10]:

history_len = 5
observation_shape = (3, 80, 80)

state_shape = (history_len, ) + observation_shape

observation_net_params = None
aggregation_net_params = None
main_net_params = {
    "features": [128],
    "use_bias": False,
    "use_normalization": True,
    "dropout_rate": 0.1,
    "activation": "LeakyReLU",
}

# In[11]:

net = StateNet.get_from_params(
    state_shape=state_shape,
    observation_net_params=observation_net_params,
    aggregation_net_params=aggregation_net_params,
    main_net_params=main_net_params,
)
print(net)

out = _get_network_output(net, state_shape)
print(out.shape, out.nelement())

# ---

# # FF - LAMA

# In[12]:

history_len = 5
observation_shape = (24, )

state_shape = (
    history_len,
    observation_shape[0],
)
observation_net_params = {
    "_network_type": "linear",
    "history_len": 1,
    "features": [128],
    "use_bias": False,
    "use_normalization": True,
    "dropout_rate": 0.1,
    "activation": "LeakyReLU",
}
# observation_net_params = None

aggregation_net_params = {
    "groups": ["last", "avg_droplast", "max_droplast", "softmax_droplast"]
}
# aggregation_net_params=None

main_net_params = {
    "features": [128],
    "use_bias": False,
    "use_normalization": True,
    "dropout_rate": 0.1,
    "activation": "LeakyReLU",
}

# In[13]:

net = StateNet.get_from_params(
    state_shape=state_shape,
    observation_net_params=observation_net_params,
    aggregation_net_params=aggregation_net_params,
    main_net_params=main_net_params,
)
print(net)

out = _get_network_output(net, state_shape)
print(out.shape, out.nelement())

# # CNN - LAMA

# In[14]:

history_len = 5
observation_shape = (3, 80, 80)

state_shape = (history_len, ) + observation_shape

observation_net_params = {
    "_network_type": "convolution",
    "history_len": 1,
    "channels": [16, 32, 16],
    "use_bias": False,
    "use_normalization": False,
    "dropout_rate": 0.1,
    "activation": "LeakyReLU",
}
# observation_net_params = None
aggregation_net_params = {
    "groups": ["last", "avg_droplast", "max_droplast", "softmax_droplast"]
}
# aggregation_net_params=None
main_net_params = {
    "features": [128],
    "use_bias": False,
    "use_normalization": True,
    "dropout_rate": 0.1,
    "activation": "LeakyReLU",
}

# In[15]:

net = StateNet.get_from_params(
    state_shape=state_shape,
    observation_net_params=observation_net_params,
    aggregation_net_params=aggregation_net_params,
    main_net_params=main_net_params,
)
print(net)

out = _get_network_output(net, state_shape)
print(out.shape, out.nelement())

# ---

# # FF - KV

# In[16]:

history_len = 5
observation_shape1, observation_shape2 = 24, 42

state_shape = {
    "obs1": (
        history_len,
        observation_shape1,
    ),
    "obs2": (
        history_len,
        observation_shape2,
    ),
}

observation_net_params = {
    "_key_value": True,
    "obs1": {
        "_network_type": "linear",
        "history_len": history_len,
        "features": [128],
        "use_bias": False,
        "use_normalization": True,
        "dropout_rate": 0.1,
        "activation": "LeakyReLU",
    },
    "obs2": {
        "_network_type": "linear",
        "history_len": history_len,
        "features": [128],
        "use_bias": False,
        "use_normalization": True,
        "dropout_rate": 0.1,
        "activation": "LeakyReLU",
    }
}
# observation_net_params = None

aggregation_net_params = {}
aggregation_net_params = None

main_net_params = {
    "features": [128],
    "use_bias": False,
    "use_normalization": True,
    "dropout_rate": 0.1,
    "activation": "LeakyReLU",
}

# In[17]:

net = StateNet.get_from_params(
    state_shape=state_shape,
    observation_net_params=observation_net_params,
    aggregation_net_params=aggregation_net_params,
    main_net_params=main_net_params,
)
print(net)

out = _get_network_output(net, state_shape)
print(out.shape, out.nelement())

# ---

# # FF - LAMA - KV

# In[18]:

history_len = 5
observation_shape1, observation_shape2 = 24, 42

state_shape = {
    "obs1": (
        history_len,
        observation_shape1,
    ),
    "obs2": (
        history_len,
        observation_shape2,
    ),
}

observation_net_params = {
    "_key_value": True,
    "obs1": {
        "_network_type": "linear",
        "history_len": 1,
        "features": [128],
        "use_bias": False,
        "use_normalization": True,
        "dropout_rate": 0.1,
        "activation": "LeakyReLU",
    },
    "obs2": {
        "_network_type": "linear",
        "history_len": 1,
        "features": [128],
        "use_bias": False,
        "use_normalization": True,
        "dropout_rate": 0.1,
        "activation": "LeakyReLU",
    }
}
# observation_net_params = None

aggregation_net_params = {
    "groups": ["last", "avg_droplast", "max_droplast", "softmax_droplast"]
}
# aggregation_net_params=None

main_net_params = {
    "features": [128],
    "use_bias": False,
    "use_normalization": True,
    "dropout_rate": 0.1,
    "activation": "LeakyReLU",
}

# In[19]:

net = StateNet.get_from_params(
    state_shape=state_shape,
    observation_net_params=observation_net_params,
    aggregation_net_params=aggregation_net_params,
    main_net_params=main_net_params,
)
print(net)

out = _get_network_output(net, state_shape)
print(out.shape, out.nelement())

# ----

# # CNN - KV

# In[20]:

history_len = 5

observation_shape1, observation_shape2 = (3, 80, 80), (3, 42, 42)

state_shape = {
    "obs1": (history_len, ) + observation_shape1,
    "obs2": (history_len, ) + observation_shape2,
}

observation_net_params = {
    "_key_value": True,
    "obs1": {
        "_network_type": "convolution",
        "history_len": history_len,
        "channels": [16, 32, 16],
        "use_bias": False,
        "use_normalization": False,
        "dropout_rate": 0.1,
        "activation": "LeakyReLU",
    },
    "obs2": {
        "_network_type": "convolution",
        "history_len": history_len,
        "channels": [16, 32, 16],
        "use_bias": False,
        "use_normalization": False,
        "dropout_rate": 0.1,
        "activation": "LeakyReLU",
    }
}
# observation_net_params = None

aggregation_net_params = {}
aggregation_net_params = None

main_net_params = {
    "features": [128],
    "use_bias": False,
    "use_normalization": True,
    "dropout_rate": 0.1,
    "activation": "LeakyReLU",
}

# In[21]:

net = StateNet.get_from_params(
    state_shape=state_shape,
    observation_net_params=observation_net_params,
    aggregation_net_params=aggregation_net_params,
    main_net_params=main_net_params,
)
print(net)

out = _get_network_output(net, state_shape)
print(out.shape, out.nelement())

# ---

# # CNN - LAMA - KV

# In[22]:

history_len = 5

observation_shape1, observation_shape2 = (3, 80, 80), (3, 42, 42)

state_shape = {
    "obs1": (history_len, ) + observation_shape1,
    "obs2": (history_len, ) + observation_shape2,
}

observation_net_params = {
    "_key_value": True,
    "obs1": {
        "_network_type": "convolution",
        "history_len": 1,
        "channels": [16, 32, 16],
        "use_bias": False,
        "use_normalization": False,
        "dropout_rate": 0.1,
        "activation": "LeakyReLU",
    },
    "obs2": {
        "_network_type": "convolution",
        "history_len": 1,
        "channels": [16, 32, 16],
        "use_bias": False,
        "use_normalization": False,
        "dropout_rate": 0.1,
        "activation": "LeakyReLU",
    }
}
# observation_net_params = None

aggregation_net_params = {
    "groups": ["last", "avg_droplast", "max_droplast", "softmax_droplast"]
}
# aggregation_net_params=None

main_net_params = {
    "features": [128],
    "use_bias": False,
    "use_normalization": True,
    "dropout_rate": 0.1,
    "activation": "LeakyReLU",
}

# In[23]:

net = StateNet.get_from_params(
    state_shape=state_shape,
    observation_net_params=observation_net_params,
    aggregation_net_params=aggregation_net_params,
    main_net_params=main_net_params,
)
print(net)

out = _get_network_output(net, state_shape)
print(out.shape, out.nelement())

# # FF+CNN - KV

# In[24]:

history_len = 5

observation_shape1, observation_shape2 = (3, 80, 80), 42

state_shape = {
    "obs1": (history_len, ) + observation_shape1,
    "obs2": (
        history_len,
        observation_shape2,
    ),
}

observation_net_params = {
    "_key_value": True,
    "obs1": {
        "_network_type": "convolution",
        "history_len": history_len,
        "channels": [16, 32, 16],
        "use_bias": False,
        "use_normalization": False,
        "dropout_rate": 0.1,
        "activation": "LeakyReLU",
    },
    "obs2": {
        "_network_type": "linear",
        "history_len": history_len,
        "features": [128],
        "use_bias": False,
        "use_normalization": True,
        "dropout_rate": 0.1,
        "activation": "LeakyReLU",
    }
}
# observation_net_params = None

aggregation_net_params = {}
aggregation_net_params = None

main_net_params = {
    "features": [128],
    "use_bias": False,
    "use_normalization": True,
    "dropout_rate": 0.1,
    "activation": "LeakyReLU",
}

# In[25]:

net = StateNet.get_from_params(
    state_shape=state_shape,
    observation_net_params=observation_net_params,
    aggregation_net_params=aggregation_net_params,
    main_net_params=main_net_params,
)
print(net)

out = _get_network_output(net, state_shape)
print(out.shape, out.nelement())

# ----

# # FF+CNN - LAMA - KV

# In[26]:

history_len = 5

observation_shape1, observation_shape2 = (3, 80, 80), 42

state_shape = {
    "obs1": (history_len, ) + observation_shape1,
    "obs2": (
        history_len,
        observation_shape2,
    ),
}

observation_net_params = {
    "_key_value": True,
    "obs1": {
        "_network_type": "convolution",
        "history_len": 1,
        "channels": [16, 32, 16],
        "use_bias": False,
        "use_normalization": False,
        "dropout_rate": 0.1,
        "activation": "LeakyReLU",
    },
    "obs2": {
        "_network_type": "linear",
        "history_len": 1,
        "features": [128],
        "use_bias": False,
        "use_normalization": True,
        "dropout_rate": 0.1,
        "activation": "LeakyReLU",
    }
}
# observation_net_params = None

aggregation_net_params = {
    "groups": ["last", "avg_droplast", "max_droplast", "softmax_droplast"]
}
# aggregation_net_params=None

main_net_params = {
    "features": [128],
    "use_bias": False,
    "use_normalization": True,
    "dropout_rate": 0.1,
    "activation": "LeakyReLU",
}

# In[27]:

net = StateNet.get_from_params(
    state_shape=state_shape,
    observation_net_params=observation_net_params,
    aggregation_net_params=aggregation_net_params,
    main_net_params=main_net_params,
)
print(net)

out = _get_network_output(net, state_shape)
print(out.shape, out.nelement())

# ---

# ---

# # FF - base

# In[28]:

history_len = 5
observation_shape = (24, )
action_shape = (5, )

state_shape = (
    history_len,
    observation_shape[0],
)
observation_net_params = {
    "_network_type": "linear",
    "history_len": history_len,
    "features": [128],
    "use_bias": False,
    "use_normalization": True,
    "dropout_rate": 0.1,
    "activation": "LeakyReLU",
}
# observation_net_params = None

aggregation_net_params = {}
aggregation_net_params = None

action_net_params = {
    "_network_type": "linear",
    "history_len": 1,
    "features": [128],
    "use_bias": False,
    "use_normalization": True,
    "dropout_rate": 0.1,
    "activation": "LeakyReLU",
}
# action_net_params = None

main_net_params = {
    "features": [128],
    "use_bias": False,
    "use_normalization": True,
    "dropout_rate": 0.1,
    "activation": "LeakyReLU",
}

# In[29]:

net = StateActionNet.get_from_params(
    state_shape=state_shape,
    action_shape=action_shape,
    observation_net_params=observation_net_params,
    aggregation_net_params=aggregation_net_params,
    action_net_params=action_net_params,
    main_net_params=main_net_params,
)
print(net)

out = _get_network_output(net, state_shape, action_shape)
print(out.shape, out.nelement())

# # FF - no observation/action

# In[30]:

history_len = 5
observation_shape = (24, )
action_shape = (5, )

state_shape = (
    history_len,
    observation_shape[0],
)
observation_net_params = None
aggregation_net_params = None
action_net_params = None

main_net_params = {
    "features": [128],
    "use_bias": False,
    "use_normalization": True,
    "dropout_rate": 0.1,
    "activation": "LeakyReLU",
}

# In[31]:

net = StateActionNet.get_from_params(
    state_shape=state_shape,
    action_shape=action_shape,
    observation_net_params=observation_net_params,
    aggregation_net_params=aggregation_net_params,
    action_net_params=action_net_params,
    main_net_params=main_net_params,
)
print(net)

out = _get_network_output(net, state_shape, action_shape)
print(out.shape, out.nelement())

# # FF+CNN - LAMA - KV

# In[32]:

history_len = 5
observation_shape1, observation_shape2 = (3, 80, 80), 42
action_shape = (5, )
state_shape = {
    "obs1": (history_len, ) + observation_shape1,
    "obs2": (
        history_len,
        observation_shape2,
    ),
}

observation_net_params = {
    "_key_value": True,
    "obs1": {
        "_network_type": "convolution",
        "history_len": 1,
        "channels": [16, 32, 16],
        "use_bias": False,
        "use_normalization": False,
        "dropout_rate": 0.1,
        "activation": "LeakyReLU",
    },
    "obs2": {
        "_network_type": "linear",
        "history_len": 1,
        "features": [128],
        "use_bias": False,
        "use_normalization": True,
        "dropout_rate": 0.1,
        "activation": "LeakyReLU",
    }
}
# observation_net_params = None

aggregation_net_params = {
    "groups": [
        "last", "avg_droplast", "max_droplast", {
            "key": "softmax_droplast",
            "kernel_size": 3,
            "padding": 1,
            "bias": False,
        }
    ]
}
# aggregation_net_params=None

action_net_params = {
    "_network_type": "linear",
    "history_len": 1,
    "features": [128],
    "use_bias": False,
    "use_normalization": True,
    "dropout_rate": 0.1,
    "activation": "LeakyReLU",
}
# action_net_params = None

main_net_params = {
    "features": [128],
    "use_bias": False,
    "use_normalization": True,
    "dropout_rate": 0.1,
    "activation": "LeakyReLU",
}

# In[33]:

net = StateActionNet.get_from_params(
    state_shape=state_shape,
    action_shape=action_shape,
    observation_net_params=observation_net_params,
    aggregation_net_params=aggregation_net_params,
    action_net_params=action_net_params,
    main_net_params=main_net_params,
)
print(net)

out = _get_network_output(net, state_shape, action_shape)
print(out.shape, out.nelement())

# In[ ]:
